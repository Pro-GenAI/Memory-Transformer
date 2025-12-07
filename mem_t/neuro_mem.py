# A memory model that stores memories in neurons resembling human brain.

"""
Neural-only Hierarchical Memory Model
------------------------------------
This single-file implementation stores and retrieves memories using *neural
parameters* (PyTorch). There is no external embedding model or vector DB: raw
text is encoded by a learned character-level encoder (NeuralEncoder), and each
memory is written into a trainable MemoryBank slot using gradient updates.

Features:
- Add memories by writing into a parametric memory bank (trainable weights).
- Query by encoding text and doing cosine similarity against memory bank slots.
- Recall, forget, compact, replay, and save/load (torch) using neural weights.
- All "memory" content is recoverable from stored text (kept for human readability)
  but the retrieval signal comes from trained neurons.

Note: This file requires PyTorch. Keep in mind storing the actual text alongside
neural weights is pragmatic — the retrieval relies on neurons whereas the text
is useful for inspection, debugging, and serialization.

Usage:
    model = HierarchicalMemoryModel(max_slots=4096)
    key = model.add_memory("I like coffee", user_id="alice")
    results = model.query("coffee", top_k=3)
    model.save("memories.pt")

"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ----------------------------- Config ---------------------------------
DEFAULT_USER = "default_user"

# Character vocabulary for simple char-level encoder
# Keep printable ASCII subset (space + 33..126)
VOCAB = [" "] + [chr(i) for i in range(33, 127)]
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Hyperparams
ENCODE_DIM = 128  # encoder output dim
MEMORY_DIM = 128  # memory vector dimensionality
DEFAULT_MAX_SLOTS = 2048
WRITE_ITERS = 200
WRITE_LR = 5e-2
RECALL_TEMPERATURE = 0.05
NEUROGENESIS_GROW_BY = 128  # number of slots to add when growing
MIN_WRITE_SIM = 0.85  # minimum cosine similarity required after writing

# --------------------------- Data classes ------------------------------
@dataclass
class MemoryItem:
    key: str
    text: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 1.0
    synaptic_strength: float = 1.0
    memory_type: str = "episodic"
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------- Neural modules ----------------------------
class NeuralEncoder(nn.Module):
    """Simple char-level encoder:

    - embeds characters
    - uses a 1D conv stack + global pooling to produce fixed-length vector
    - final vector normalized to unit length
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, out_dim: int = ENCODE_DIM):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, 128)
        # larger conv stack for better discrimination
        self.conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        # use max pooling over sequence to make short distinctive tokens stand out
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.char_emb(x)  # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        h = F.relu(self.conv1(emb))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # global max pool over seq
        h = h.max(dim=2).values
        out = self.fc(h)
        out = F.normalize(out, dim=-1)
        return out


class MemoryBank(nn.Module):
    """A parametric memory bank: a fixed number of slots, each is a trainable vector.

    - `slots` is a Parameter of shape (max_slots, memory_dim)
    - `used` mask tracks which slots have been allocated
    - writing a memory optimizes a selected slot to match the encoded vector
    """

    def __init__(self, max_slots: int, memory_dim: int):
        super().__init__()
        # Initialize slots small random to avoid symmetry
        self.max_slots = max_slots
        self.memory_dim = memory_dim
        self.slots = nn.Parameter(torch.randn(max_slots, memory_dim) * 0.01)
        # persistent tensor mask for used slots (bool). Not a Parameter.
        self.used = torch.zeros(max_slots, dtype=torch.bool)

    def allocate_slot(self) -> int:
        """Return the index of the first free slot, or raise if full."""
        free_idx = ((~self.used.bool())).nonzero(as_tuple=False)
        if free_idx.numel() == 0:
            raise RuntimeError("MemoryBank is full — compact or increase max_slots")
        idx = int(free_idx[0].item())
        self.used[idx] = torch.tensor(True)
        return idx

    def grow(self, n_new_slots: int):
        """Increase the capacity of the memory bank by `n_new_slots`.

        This replaces the `slots` Parameter with a larger Parameter and
        preserves the existing slot contents and used mask.
        """
        if n_new_slots <= 0:
            return
        new_max = self.max_slots + n_new_slots
        # create new parameter and copy existing data
        with torch.no_grad():
            device = self.slots.device
            new_slots = nn.Parameter(torch.randn(new_max, self.memory_dim, device=device) * 0.01)
            # copy old data
            new_slots.data[: self.max_slots].copy_(self.slots.data)
        # replace parameter (registers properly)
        self.slots = new_slots
        # expand used mask
        new_used = torch.zeros(new_max, dtype=torch.bool, device=self.used.device)
        new_used[: self.max_slots] = self.used
        self.used = new_used
        self.max_slots = new_max

    def free_slot(self, idx: int):
        if 0 <= idx < self.max_slots:
            self.used[idx] = False
            # zero out the slot to forget
            with torch.no_grad():
                self.slots.data[idx].zero_()

    def active_slots(self) -> torch.Tensor:
        """Return the indices of active slots."""
        return self.used.bool().nonzero(as_tuple=False).squeeze(1)

    def forward(self) -> torch.Tensor:
        """Return normalized active memory vectors (N_active, dim)
        For convenience we return all slots (normalized)."""
        slots = F.normalize(self.slots, dim=-1)
        return slots


# ------------------------ HierarchicalMemoryModel ----------------------
class HierarchicalMemoryModel:
    """Neural-only memory model.

    All retrieval signals come from neural modules: NeuralEncoder and MemoryBank.
    Text is stored for inspection and serialization, but discovery relies on
    trained neurons.
    """

    def __init__(
        self,
        max_slots: int = DEFAULT_MAX_SLOTS,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.encoder = NeuralEncoder(VOCAB_SIZE).to(self.device)
        self.memory_bank = MemoryBank(max_slots, MEMORY_DIM).to(self.device)

        # Projection head to map encoder dim -> memory dim if they differ
        if ENCODE_DIM != MEMORY_DIM:
            self.proj = nn.Linear(ENCODE_DIM, MEMORY_DIM).to(self.device)
        else:
            self.proj = None

        # bookkeeping: map slot_idx -> MemoryItem and key
        self.slot_to_item: Dict[int, MemoryItem] = {}
        self.key_to_slot: Dict[str, int] = {}

        # optimizer for writing operations (we'll create optim per-write)
        self.global_write_lr = WRITE_LR

    # --------------------- utilities: encoding & similarity ----------------
    def _text_to_indices(self, texts: List[str], max_len: int = 256) -> torch.Tensor:
        """Convert a list of texts to a (batch, seq_len) Tensor of char indices."""
        batch_size = len(texts)
        seqs = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, t in enumerate(texts):
            # clip / pad
            s = t[:max_len]
            idxs = [CHAR2IDX.get(ch, 0) for ch in s]
            seqs[i, : len(idxs)] = torch.tensor(idxs, dtype=torch.long)
        return seqs

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """Return normalized memory-dim vectors for texts."""
        self.encoder.eval()
        with torch.no_grad():
            idx = self._text_to_indices(texts).to(self.device)
            z = self.encoder(idx)
            if self.proj is not None:
                z = self.proj(z)
                z = F.normalize(z, dim=-1).to(self.device)
        return z

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarities between a (Nq, D) and b (Nb, D) -> (Nq, Nb)"""
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return a @ b.t()

    # ------------------------ core API -----------------------------------
    def add_memory(
        self,
        text: str,
        user_id: str = DEFAULT_USER,
        key: Optional[str] = None,
        importance: float = 1.0,
        memory_type: str = "episodic",
        synaptic_strength: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a memory into the bank using gradient descent.

        The method allocates a new slot and trains that slot to match the
        encoded representation of `text`. This means the information is stored
        in the memory bank (neural weights).
        """
        key = key or f"m_{int(time.time() * 1000)}"

        # allocate slot (with one retry via neurogenesis if bank is full)
        try:
            slot_idx = self.memory_bank.allocate_slot()
        except RuntimeError:
            # try neurogenesis: grow and allocate again
            try:
                self.memory_bank.grow(NEUROGENESIS_GROW_BY)
                slot_idx = self.memory_bank.allocate_slot()
            except Exception:
                raise

        # create bookkeeping MemoryItem
        item = MemoryItem(
            key=key,
            text=text,
            user_id=user_id,
            importance=importance,
            memory_type=memory_type,
            synaptic_strength=synaptic_strength,
            meta=meta or {},
        )
        self.slot_to_item[slot_idx] = item
        self.key_to_slot[key] = slot_idx

        # train the slot to match encoded vector
        target = self._encode([text])[0:1]  # (1, D)

        # token-level boosting: include token encodings to emphasize keywords
        try:
            tokens = [t for t in text.split() if t]
            if tokens:
                # limit tokens to avoid huge batches
                tokens = tokens[:32]
                token_encs = self._encode(tokens)  # (n_tokens, D)
                token_avg = token_encs.mean(dim=0, keepdim=True)
                # weight token signal moderately
                target = F.normalize(target + 0.8 * token_avg, dim=-1)
        except:  # fallback: keep original target
            target = target

        # We'll optimize only the single slot vector
        slots_param = self.memory_bank.slots
        optimizer = optim.SGD([slots_param], lr=self.global_write_lr, momentum=0.9)

        # training loop: minimize cosine-distance to target
        for it in range(WRITE_ITERS):
            optimizer.zero_grad()
            # normalize slot and target
            # we want to maximize cosine similarity -> minimize (1 - cos)
            slots_param = self.memory_bank.slots
            slot_vec = slots_param[slot_idx:slot_idx+1]
            slot_norm = F.normalize(slot_vec, dim=-1)
            loss = 1.0 - (slot_norm * target).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            # small weight decay to keep values stable
            with torch.no_grad():
                slots_param.mul_(0.999)

        # ensure slot normalized numerically and check write quality
        with torch.no_grad():
            try:
                tgt = target[0].to(self.memory_bank.slots.data.device)
                self.memory_bank.slots.data[slot_idx] = F.normalize(tgt, dim=-1)
            except Exception:
                # fallback: normalize existing slot
                self.memory_bank.slots.data[slot_idx] = F.normalize(
                    self.memory_bank.slots.data[slot_idx], dim=-1
                )

        # verify the written slot achieves minimum similarity; if not, try neurogenesis once
        try:
            with torch.no_grad():
                written = F.normalize(self.memory_bank.slots.data[slot_idx:slot_idx+1], dim=-1)
                sim = float((written @ F.normalize(target, dim=-1).t()).item())
            if sim < MIN_WRITE_SIM:
                # free the partially written slot and grow bank, then retry one more time
                self.memory_bank.free_slot(slot_idx)
                # grow
                self.memory_bank.grow(NEUROGENESIS_GROW_BY)
                # allocate a fresh slot and write target directly (fast path)
                new_slot = self.memory_bank.allocate_slot()
                with torch.no_grad():
                    self.memory_bank.slots.data[new_slot] = F.normalize(
                        target[0].to(self.memory_bank.slots.data.device), dim=-1
                    )
                # update bookkeeping to point to new slot
                del self.slot_to_item[slot_idx]
                self.slot_to_item[new_slot] = item
                self.key_to_slot[key] = new_slot
                slot_idx = new_slot
        except Exception:
            # ignore failures in the verification/grow path
            pass

        return key

    def query(
        self,
        query_text: str,
        user_id: str = DEFAULT_USER,
        top_k: int = 10,
        threshold: float = 0.1,
    ) -> List[Tuple[float, MemoryItem]]:
        """Return top-k memories by cosine similarity (neural retrieval).

        Only slots whose stored MemoryItem.user_id matches the requested user_id
        will be returned.
        """
        if len(self.slot_to_item) == 0:
            return []

        q = self._encode([query_text])  # (1, D)
        slots = F.normalize(self.memory_bank.slots, dim=-1)  # (max_slots, D)
        sims = (q @ slots.t()).squeeze(0)  # (max_slots,)

        # mask out unused slots
        used_mask = self.memory_bank.used.to(sims.device)
        sims = sims.masked_fill(~used_mask, -10.0)

        # Collect candidates for the given user_id only
        candidate_indices = [idx for idx, item in self.slot_to_item.items() if item.user_id == user_id]
        if not candidate_indices:
            return []

        cand_tensor = torch.tensor(candidate_indices, device=sims.device, dtype=torch.long)
        cand_sims = sims[cand_tensor]

        # select top-k
        topk = min(top_k, len(candidate_indices))
        vals, idxs = torch.topk(cand_sims, topk)

        results: List[Tuple[float, MemoryItem]] = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            slot = candidate_indices[i]
            if v < threshold:
                continue
            item = self.slot_to_item.get(slot)
            if item is None:
                continue
            # apply synaptic strength multiplier
            score = float(v) * (0.8 + (item.synaptic_strength - 1.0) * 0.2)
            # lexical token overlap boost: raise score when query tokens appear in memory text
            try:
                q_tokens = [t.lower() for t in query_text.split() if t]
                if q_tokens:
                    mem_tokens = set([t.lower() for t in item.text.split() if t])
                    matches = sum(1 for t in q_tokens if t in mem_tokens)
                    lexical_boost = 0.25 * (matches / max(1, len(q_tokens)))
                    score = score + lexical_boost
            except Exception:
                pass
            # update access stats
            item.access_count += 1
            item.last_accessed = time.time()
            results.append((score, item))

        # sort results
        results.sort(key=lambda x: -x[0])
        return results

    def recall_by_key(self, key: str) -> Optional[MemoryItem]:
        slot = self.key_to_slot.get(key)
        if slot is None:
            return None
        item = self.slot_to_item.get(slot)
        if item:
            item.access_count += 1
            item.last_accessed = time.time()
        return item

    def forget(self, key: str) -> bool:
        slot = self.key_to_slot.get(key)
        if slot is None:
            return False
        # free slot and remove bookkeeping
        self.memory_bank.free_slot(slot)
        del self.slot_to_item[slot]
        del self.key_to_slot[key]
        return True

    def delete_all(self, user_id: str) -> int:
        """Delete all memories belonging to `user_id`.

        Returns the number of memories removed.
        """
        # collect slots to remove to avoid modifying dict during iteration
        to_remove = [slot for slot, item in self.slot_to_item.items() if item.user_id == user_id]
        for slot in to_remove:
            item = self.slot_to_item.get(slot)
            if item:
                # remove key mapping
                if item.key in self.key_to_slot:
                    try:
                        del self.key_to_slot[item.key]
                    except KeyError:
                        pass
            # remove slot bookkeeping and free slot
            if slot in self.slot_to_item:
                try:
                    del self.slot_to_item[slot]
                except KeyError:
                    pass
            self.memory_bank.free_slot(slot)
        return len(to_remove)

    def compact(self, max_keep: int = 1024):
        """Simple compaction: keep top `max_keep` memories by importance+recency.

        We will identify slots to keep and re-assign remaining ones by freeing them.
        """
        items = list(self.slot_to_item.items())  # (slot, item)
        if len(items) <= max_keep:
            return

        # score by importance and recency
        scored = []
        now = time.time()
        for slot, it in items:
            age = now - it.last_accessed
            score = it.importance + 1.0 / (1.0 + age / (60.0 * 60.0)) + 0.01 * it.access_count
            scored.append((score, slot, it))
        scored.sort(key=lambda x: -x[0])

        # keep = set([s for _, s, _ in scored[:max_keep]])
        remove = [s for _, s, _ in scored[max_keep:]]

        for slot in remove:
            item = self.slot_to_item.get(slot)
            if item:
                del self.key_to_slot[item.key]
            if slot in self.slot_to_item:
                del self.slot_to_item[slot]
            self.memory_bank.free_slot(slot)

    # -------------------- sleep/replay style consolidation ----------------
    def memory_replay(self, replay_cycles: int = 3, focus_on_recent: bool = True):
        """Simulate replay by re-training memory bank slots on their texts.

        This strengthens the slot vectors (neural weights) and slightly boosts
        importance/synaptic strength.
        """
        slots = list(self.slot_to_item.keys())
        if not slots:
            return

        for cycle in range(replay_cycles):
            for slot in slots:
                item = self.slot_to_item.get(slot)
                if item is None:
                    continue
                # slight prioritization of recent/important
                if focus_on_recent and item.access_count < 1 and cycle < replay_cycles - 1:
                    continue
                # retrain this specific slot briefly (avoid pulling all slots toward target)
                target = self._encode([item.text])[0:1]
                slots_param = self.memory_bank.slots
                optimizer = optim.SGD([slots_param], lr=self.global_write_lr, momentum=0.9)
                for it in range(10):
                    optimizer.zero_grad()
                    slot_vec = slots_param[slot:slot+1]
                    slot_norm = F.normalize(slot_vec, dim=-1)
                    loss = 1.0 - (slot_norm * target).sum(dim=-1).mean()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    slots_param.data[slot] = F.normalize(slots_param.data[slot], dim=-1)
                # boost synapses slightly
                item.synaptic_strength = min(2.0, item.synaptic_strength * 1.02)
                item.importance = min(1.0, item.importance * 1.01)

    # --------------------- persistence ----------------------------------
    def save(self, path: str):
        """Save neural weights and metadata to a single torch file."""
        path1 = Path(path)
        path1.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_state": self.encoder.state_dict(),
            "memory_slots": self.memory_bank.state_dict(),
            "proj_state": self.proj.state_dict() if self.proj is not None else None,
            "slot_to_item": {str(k): self._serialize_item(v) for k, v in self.slot_to_item.items()},
            "key_to_slot": self.key_to_slot,
        }
        torch.save(payload, str(path1))

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "HierarchicalMemoryModel":
        path1 = Path(path)
        data = torch.load(str(path1), map_location=device)
        max_slots = data["memory_slots"]["slots"].shape[0]
        model = cls(max_slots=max_slots, device=device)
        model.encoder.load_state_dict(data["encoder_state"])
        model.memory_bank.load_state_dict(data["memory_slots"]) 
        if data.get("proj_state") is not None and model.proj is not None:
            model.proj.load_state_dict(data["proj_state"])
        # restore items
        slot_to_item = {}
        for k, v in data["slot_to_item"].items():
            slot = int(k)
            slot_to_item[slot] = cls._deserialize_item(v)
            model.key_to_slot[slot_to_item[slot].key] = slot
        model.slot_to_item = slot_to_item
        return model

    def _serialize_item(self, item: MemoryItem) -> Dict[str, Any]:
        return {
            "key": item.key,
            "text": item.text,
            "user_id": item.user_id,
            "created_at": item.created_at,
            "access_count": item.access_count,
            "last_accessed": item.last_accessed,
            "importance": item.importance,
            "synaptic_strength": item.synaptic_strength,
            "memory_type": item.memory_type,
        }

    @classmethod
    def _deserialize_item(cls, d: Dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            key=d.get("key", ""),
            text=d.get("text", ""),
            user_id=d.get("user_id", DEFAULT_USER),
            created_at=d.get("created_at", time.time()),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", time.time()),
            importance=d.get("importance", 1.0),
            synaptic_strength=d.get("synaptic_strength", 1.0),
            memory_type=d.get("memory_type", "episodic"),
        )

memory_model = HierarchicalMemoryModel()


# --------------------------- quick smoke test --------------------------
if __name__ == "__main__":
    m = HierarchicalMemoryModel(max_slots=512)
    print("Adding memories...")
    m.add_memory("I had coffee with Alice yesterday", user_id="alice")
    m.add_memory("Alice likes hiking in the Alps", user_id="alice")
    m.add_memory("Bob loves tea and chess", user_id="bob")

    print("Query: 'coffee' for alice")
    res = m.query("coffee", user_id="alice", top_k=5)
    for s, item in res:
        print(f"score={s:.3f}  key={item.key}  text={item.text}")

    print("Saving to /tmp/memories.pt")
    m.save("/tmp/memories.pt")

    print("Reloading and re-query")
    m2 = HierarchicalMemoryModel.load("/tmp/memories.pt")
    res2 = m2.query("tea", user_id="bob", top_k=3)
    for s, item in res2:
        print(f"score={s:.3f} key={item.key} text={item.text}")
