"""
Memory Transformer
------------------
Transformer-inspired memory system that stores text into learned memory
slots and retrieves via attention/cosine similarity. This model avoids
external embeddings/DBs: a small tokenizer/encoder feeds a
TransformerEncoder; a set of trainable memory slot vectors represent
stored memories.

Core ideas:
- Tokenize raw text (char-level) and encode with a lightweight
  TransformerEncoder.
- Memory slots: trainable vectors. Adding a memory writes (optimizes)
  one slot towards the encoded text representation.
- Query: encode query text, compute cosine similarity against active slots
  and return top matches per user.
- Save/Load via torch to persist both neural weights and metadata.

Usage:
	mt = MemoryTransformer(max_slots=1024)
	k = mt.add_memory("Alice loves coffee", user_id="alice")
	res = mt.query("coffee", user_id="alice")

"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
	# Optional: use external tokenizer if available
	from mem_t.utils.embed_utils import tokenize_text as external_tokenize_text
except Exception:
	external_tokenize_text = None


# ----------------------------- Config ---------------------------------
DEFAULT_USER = "default_user"

# Char vocabulary (printable ASCII subset)
VOCAB = [" "] + [chr(i) for i in range(33, 127)]
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Model sizes
MODEL_DIM = 128
FF_DIM = 256
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 256
MEMORY_DIM = MODEL_DIM
DEFAULT_MAX_SLOTS = 1024

# Write/query behavior
WRITE_ITERS = 150
WRITE_LR = 5e-2
MIN_WRITE_SIM = 0.80
NEUROGENESIS_GROW_BY = 128


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


# --------------------------- Modules -----------------------------------
class CharTokenizer(nn.Module):
	def __init__(self, vocab_size: int):
		super().__init__()
		self.emb = nn.Embedding(vocab_size, MODEL_DIM)
		self.pos = nn.Embedding(MAX_SEQ_LEN, MODEL_DIM)

	def forward(self, idx: torch.LongTensor) -> torch.Tensor:
		# idx: (B, L)
		b, l = idx.shape
		pos_ids = torch.arange(l, device=idx.device).unsqueeze(0).expand(b, l)
		x = self.emb(idx) + self.pos(pos_ids)
		return x  # (B, L, D)


class DynamicTokenEmbedding(nn.Module):
	"""Embedding that grows to fit incoming token ids from external tokenizer."""
	def __init__(self, init_vocab_size: int = 4096, dim: int = MODEL_DIM):
		super().__init__()
		self.dim = dim
		self.vocab_size = init_vocab_size
		self.emb = nn.Embedding(self.vocab_size, dim)

	def maybe_grow(self, max_id: int):
		if max_id < self.vocab_size:
			return
		new_size = max_id + 1024  # grow with margin
		with torch.no_grad():
			new_emb = nn.Embedding(new_size, self.dim)
			new_emb.weight[: self.vocab_size].copy_(self.emb.weight)
		self.emb = new_emb
		self.vocab_size = new_size

	def forward(self, idx: torch.LongTensor) -> torch.Tensor:
		self.maybe_grow(int(idx.max().item()))
		return self.emb(idx)


class SmallTransformerEncoder(nn.Module):
	def __init__(self, d_model: int = MODEL_DIM, n_heads: int = N_HEADS, ff_dim: int = FF_DIM, n_layers: int = N_LAYERS):
		super().__init__()
		encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# x: (B, L, D)
		h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
		h = self.norm(h)
		# global mean+max pooling for a robust sentence representation
		mean_pool = h.mean(dim=1)
		max_pool = h.max(dim=1).values
		out = F.normalize(mean_pool + max_pool, dim=-1)
		return out  # (B, D)


class MemorySlots(nn.Module):
	def __init__(self, max_slots: int, dim: int):
		super().__init__()
		self.max_slots = max_slots
		self.dim = dim
		self.slots = nn.Parameter(torch.randn(max_slots, dim) * 0.01)
		self.used = torch.zeros(max_slots, dtype=torch.bool)

	def allocate(self) -> int:
		free = ((~self.used.bool())).nonzero(as_tuple=False)
		if free.numel() == 0:
			raise RuntimeError("MemorySlots full; grow and retry")
		idx = int(free[0].item())
		self.used[idx] = torch.tensor(True)
		return idx

	def free(self, idx: int):
		if 0 <= idx < self.max_slots:
			self.used[idx] = False
			with torch.no_grad():
				self.slots.data[idx].zero_()

	def grow(self, n_new: int):
		if n_new <= 0:
			return
		new_max = self.max_slots + n_new
		device = self.slots.device
		with torch.no_grad():
			new_param = nn.Parameter(torch.randn(new_max, self.dim, device=device) * 0.01)
			new_param.data[: self.max_slots].copy_(self.slots.data)
		self.slots = new_param
		new_used = torch.zeros(new_max, dtype=torch.bool, device=self.used.device)
		new_used[: self.max_slots] = self.used
		self.used = new_used
		self.max_slots = new_max

	def forward(self) -> torch.Tensor:
		return F.normalize(self.slots, dim=-1)


# ------------------------ MemoryTransformer ----------------------------
class MemoryTransformer:
	def __init__(self, max_slots: int = DEFAULT_MAX_SLOTS, device: Optional[torch.device] = None):
		self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		# choose tokenizer path
		self.use_external = external_tokenize_text is not None
		if self.use_external:
			self.tokenizer_char = None
			self.tokenizer_dyn = DynamicTokenEmbedding().to(self.device)
			self.pos_emb = nn.Embedding(MAX_SEQ_LEN, MODEL_DIM).to(self.device)
		else:
			self.tokenizer_char = CharTokenizer(VOCAB_SIZE).to(self.device)
			self.tokenizer_dyn = None
			self.pos_emb = None
		self.encoder = SmallTransformerEncoder().to(self.device)
		self.memory = MemorySlots(max_slots, MEMORY_DIM).to(self.device)

		self.global_write_lr = WRITE_LR

		self.slot_to_item: Dict[int, MemoryItem] = {}
		self.key_to_slot: Dict[str, int] = {}

	# ---------------- utilities -----------------
	def _text_to_indices(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
		b = len(texts)
		seqs = torch.zeros((b, max_len), dtype=torch.long)
		for i, t in enumerate(texts):
			s = t[:max_len]
			idxs = [CHAR2IDX.get(ch, 0) for ch in s]
			seqs[i, : len(idxs)] = torch.tensor(idxs, dtype=torch.long)
		return seqs

	def _encode(self, texts: List[str]) -> torch.Tensor:
		self.encoder.eval()
		with torch.no_grad():
			if self.use_external and external_tokenize_text:
				# tokenize each text to variable-length ids, then pad/truncate
				ids_list: List[torch.Tensor] = []
				for t in texts:
					tid = external_tokenize_text(t)
					ids_list.append(tid)
				max_len = MAX_SEQ_LEN
				b = len(ids_list)
				idx = torch.zeros((b, max_len), dtype=torch.long)
				for i, tid in enumerate(ids_list):
					s = tid[:max_len]
					idx[i, : s.shape[0]] = s
				idx = idx.to(self.device)
				tok = self.tokenizer_dyn(idx)  # type: ignore
				# add positions
				pos_ids = torch.arange(max_len, device=self.device).unsqueeze(0).expand(b, max_len)
				x = tok + self.pos_emb(pos_ids)  # type: ignore
				pad_mask = (idx == 0)
				z = self.encoder(x, src_key_padding_mask=pad_mask)
			else:
				if self.tokenizer_char is None:
					raise RuntimeError("Tokenizer not initialized")
				idx = self._text_to_indices(texts).to(self.device)
				x = self.tokenizer_char(idx)
				pad_mask = (idx == 0)
				z = self.encoder(x, src_key_padding_mask=pad_mask)
		return z  # (B, D)

	def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		a = F.normalize(a, dim=-1)
		b = F.normalize(b, dim=-1)
		return a @ b.t()

	# ---------------- core API ------------------
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
		key = key or f"mt_{int(time.time() * 1000)}"
		try:
			slot = self.memory.allocate()
		except RuntimeError:
			self.memory.grow(NEUROGENESIS_GROW_BY)
			slot = self.memory.allocate()

		item = MemoryItem(
			key=key,
			text=text,
			user_id=user_id,
			importance=importance,
			memory_type=memory_type,
			synaptic_strength=synaptic_strength,
			meta=meta or {},
		)
		self.slot_to_item[slot] = item
		self.key_to_slot[key] = slot

		target = self._encode([text])[0:1]
		# token emphasis via subword averaging
		try:
			tokens = text.split()
			if tokens:
				tokens = tokens[:32]
				tok_enc = self._encode(tokens)
				target = F.normalize(target + 0.7 * tok_enc.mean(dim=0, keepdim=True), dim=-1)
		except Exception:
			pass

		slots_param = self.memory.slots
		opt = torch.optim.SGD([slots_param], lr=self.global_write_lr, momentum=0.9)
		for _ in range(WRITE_ITERS):
			opt.zero_grad()
			vec = slots_param[slot:slot+1]
			vec_n = F.normalize(vec, dim=-1)
			loss = 1.0 - (vec_n * target).sum(dim=-1).mean()
			loss.backward()
			opt.step()
			with torch.no_grad():
				slots_param.mul_(0.999)

		# finalize write and verify
		with torch.no_grad():
			self.memory.slots.data[slot] = F.normalize(target[0], dim=-1)
			sim = float((self.memory.slots.data[slot:slot+1] @ F.normalize(target, dim=-1).t()).item())
		if sim < MIN_WRITE_SIM:
			# retry with neurogenesis and direct write
			self.memory.free(slot)
			self.memory.grow(NEUROGENESIS_GROW_BY)
			new_slot = self.memory.allocate()
			with torch.no_grad():
				self.memory.slots.data[new_slot] = F.normalize(target[0], dim=-1)
			del self.slot_to_item[slot]
			self.slot_to_item[new_slot] = item
			self.key_to_slot[key] = new_slot
			slot = new_slot

		return key

	def query(self, query_text: str, user_id: str = DEFAULT_USER, top_k: int = 10, threshold: float = 0.1) -> List[Tuple[float, MemoryItem]]:
		if not self.slot_to_item:
			return []
		q = self._encode([query_text])  # (1, D)
		slots = F.normalize(self.memory.slots, dim=-1)
		sims = (q @ slots.t()).squeeze(0)
		sims = sims.masked_fill(~self.memory.used.to(sims.device), -10.0)

		cand = [idx for idx, it in self.slot_to_item.items() if it.user_id == user_id]
		if not cand:
			return []
		ct = torch.tensor(cand, device=sims.device, dtype=torch.long)
		cs = sims[ct]
		k = min(top_k, len(cand))
		vals, idxs = torch.topk(cs, k)

		out: List[Tuple[float, MemoryItem]] = []
		for v, i in zip(vals.tolist(), idxs.tolist()):
			slot = cand[i]
			if v < threshold:
				continue
			it = self.slot_to_item.get(slot)
			if it is None:
				continue
			score = float(v) * (0.8 + (it.synaptic_strength - 1.0) * 0.2)
			try:
				qtok = [t.lower() for t in query_text.split() if t]
				mtok = set([t.lower() for t in it.text.split() if t])
				matches = sum(1 for t in qtok if t in mtok)
				score += 0.25 * (matches / max(1, len(qtok)))
			except Exception:
				pass
			it.access_count += 1
			it.last_accessed = time.time()
			out.append((score, it))

		out.sort(key=lambda x: -x[0])
		return out

	def recall_by_key(self, key: str) -> Optional[MemoryItem]:
		slot = self.key_to_slot.get(key)
		if slot is None:
			return None
		it = self.slot_to_item.get(slot)
		if it:
			it.access_count += 1
			it.last_accessed = time.time()
		return it

	def forget(self, key: str) -> bool:
		slot = self.key_to_slot.get(key)
		if slot is None:
			return False
		self.memory.free(slot)
		self.slot_to_item.pop(slot, None)
		self.key_to_slot.pop(key, None)
		return True

	def delete_all(self, user_id: str) -> int:
		to_remove = [s for s, it in self.slot_to_item.items() if it.user_id == user_id]
		for s in to_remove:
			it = self.slot_to_item.get(s)
			if it:
				self.key_to_slot.pop(it.key, None)
			self.slot_to_item.pop(s, None)
			self.memory.free(s)
		return len(to_remove)

	def save(self, path: str):
		p = Path(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		payload: Dict[str, Any] = {
			"encoder": self.encoder.state_dict(),
			"memory": self.memory.state_dict(),
			"slot_to_item": {str(k): self._ser(it) for k, it in self.slot_to_item.items()},
			"key_to_slot": self.key_to_slot,
			"use_external": self.use_external,
		}
		if self.use_external:
			if self.tokenizer_dyn is not None:
				payload["tokenizer_dyn"] = self.tokenizer_dyn.state_dict()
			if self.pos_emb is not None:
				payload["pos_emb"] = self.pos_emb.state_dict()
		else:
			if self.tokenizer_char is not None:
				payload["tokenizer_char"] = self.tokenizer_char.state_dict()
		torch.save(payload, str(p))

	@classmethod
	def load(cls, path: str, device: Optional[torch.device] = None) -> "MemoryTransformer":
		p = Path(path)
		data = torch.load(str(p), map_location=device)
		max_slots = data["memory"]["slots"].shape[0]
		mt = cls(max_slots=max_slots, device=device)
		mt.encoder.load_state_dict(data["encoder"])
		mt.memory.load_state_dict(data["memory"])
		# restore tokenizer path
		if data.get("use_external"):
			mt.use_external = True
			mt.tokenizer_char = None
			if mt.tokenizer_dyn is not None:
				mt.tokenizer_dyn.load_state_dict(data.get("tokenizer_dyn", {}))
			if mt.pos_emb is not None:
				mt.pos_emb.load_state_dict(data.get("pos_emb", {}))
		else:
			mt.use_external = False
			mt.tokenizer_dyn = None
			mt.pos_emb = None
			if mt.tokenizer_char is not None:
				mt.tokenizer_char.load_state_dict(data.get("tokenizer_char", {}))
		mt.slot_to_item = {int(k): cls._deser(v) for k, v in data["slot_to_item"].items()}
		mt.key_to_slot = data.get("key_to_slot", {})
		return mt

	def _ser(self, it: MemoryItem) -> Dict[str, Any]:
		return {
			"key": it.key,
			"text": it.text,
			"user_id": it.user_id,
			"created_at": it.created_at,
			"access_count": it.access_count,
			"last_accessed": it.last_accessed,
			"importance": it.importance,
			"synaptic_strength": it.synaptic_strength,
			"memory_type": it.memory_type,
			"meta": it.meta,
		}

	@classmethod
	def _deser(cls, d: Dict[str, Any]) -> MemoryItem:
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
			meta=d.get("meta", {}),
		)


# Convenience instance
memory_model = MemoryTransformer()


# --------------------------- smoke test --------------------------------
if __name__ == "__main__":
	mt = MemoryTransformer(max_slots=256)
	print("Writing memories...")
	start = time.time()
	mt.add_memory("Alice had coffee yesterday", user_id="alice")
	mt.add_memory("Alice enjoys hiking and coffee", user_id="alice")
	mt.add_memory("Alice went hiking last weekend", user_id="alice")
	mt.add_memory("Remember I hate tea", user_id="alice")
	mt.add_memory("My favorite drink is not tea", user_id="alice")
	mt.add_memory("Bob prefers tea and chess", user_id="bob")
	end = time.time()
	print(f"Wrote memories in {end - start:.2f} seconds")

	print("Query 'coffee' for alice")
	start = time.time()
	res = mt.query("coffee", user_id="alice", top_k=5)
	end = time.time()
	print(f"Query done in {end - start:.2f} seconds")
	for s, it in res:
		print(f"score={s:.3f} key={it.key} text={it.text}")

	path = "/tmp/memory_transformer.pt"
	mt.save(path)
	# print("Reloading...")
	# mt2 = MemoryTransformer.load(path)
	# res2 = mt2.query("tea", user_id="bob", top_k=3)
	# for s, it in res2:
	# 	print(f"score={s:.3f} key={it.key} text={it.text}")

