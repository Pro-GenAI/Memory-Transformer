# Memory Transformer: Neural Memory Storage for Textual Data

Memory Transformer is a compact, practical library for storing and retrieving textual memories directly
inside small neural modules — no external vector DB required. It brings transformer-inspired
encoders together with trainable memory slots so applications can write, query, and manage memories
using pure PyTorch objects.

### Why it matters
- Store memory signals in neural weights for tight, offline retrieval.
- Lightweight: designed to run locally without cloud embedding services.
- Flexible: supports a char-level transformer encoder (`MemoryTransformer`) and a conv-based 
    neural memory (`HierarchicalMemoryModel`) for different trade-offs of speed and accuracy.

### Key features
- __Neurogenesis__: create new neurons to grow memory capacity automatically when banks fill up.
- Write memories by optimizing trainable memory slots to match an encoded representation of the text.
- Query by cosine similarity with optional token-overlap boosts and synaptic-strength weighting.
- Save and load full model state (weights + human-readable metadata) via PyTorch serialization.
- Simple REST API wrapper included (`mem_t/mem0_server.py`) for quick service deployment.
- Compatible with mem0 to switch from mem0-compatible projects.

### Quick start
1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (PyTorch must be installed according to your
     platform/GPU):

```bash
pip install -r requirements.txt
```

3) Run the bundled server (example):

```bash
export MEM0_BASE_URL=http://0.0.0.0:8123
python -m mem_t.mem0_server
# then POST /v1/memories/ and /v1/memories/search/ to add/search memories
```

### Examples (Python)

- Memory Transformer (transformer-based encoder + slot bank):

```python
from mem_t.memo_tra_model import MemoryTransformer

mt = MemoryTransformer(max_slots=1024)
key = mt.add_memory("Alice enjoys coffee and hiking", user_id="alice")
results = mt.query("coffee", user_id="alice", top_k=5)
for score, item in results:
        print(score, item.text)
```

- Neural Hierarchical Memory (conv encoder + memory bank):

```python
from mem_t.neuro_mem import HierarchicalMemoryModel

hmm = HierarchicalMemoryModel(max_slots=2048)
key = hmm.add_memory("Remember to call Bob tomorrow", user_id="me")
print(hmm.query("call Bob", user_id="me", top_k=3))
```

### Design principles
- Local-first: run fully offline without external embedding providers.
- Interpretable: textual memory items are stored alongside the learned
    vectors for inspection and debugging.
- Minimal ops: writing is implemented as a short gradient loop to sculpt a
    slot vector, keeping the system simple and robust.

### When to use Memory Transformer
- Personal assistants that keep short/long-term user memories locally.
- Research prototypes exploring learned memory storage and neuro-inspired
    mechanisms like replay and capacity growth.
- Embedded systems and privacy-sensitive applications where cloud
    embeddings or vector DBs are undesirable.

If you'd like help with suggestions, open an issue.

Enjoy building with Memory Transformer — small neural memories, big impact.
