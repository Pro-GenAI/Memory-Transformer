import os
from typing import List

from dotenv import load_dotenv
import numpy as np
import openai
import torch
from transformers import AutoTokenizer

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OpenAI API key required. Set OPENAI_API_KEY environment variable."
    )

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
if not EMBED_MODEL_NAME:
    raise ValueError("EMBED_MODEL_NAME environment variable not set.")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
if not EMBED_BASE_URL:
    raise ValueError("EMBED_BASE_URL environment variable not set.")

EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

embed_client = openai.OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)


def embed(texts: List[str]) -> np.ndarray:
    """Generate embeddings using OpenAI."""

    # Handle batching for OpenAI (they have token limits)
    all_embeddings = []
    batch_size = 200  # Increased from 100 for better throughput with vLLM

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = embed_client.embeddings.create(
                input=batch, model=EMBED_MODEL_NAME
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms

    return embeddings


embed_sample = embed(["test"])

# If the first row doesn't exist, raise an error
if embed_sample is None or embed_sample.shape[0] == 0:
    raise ValueError("Failed to generate sample embedding.")

EMBED_DIM = int(embed_sample.shape[1])


tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

def tokenize_text(text: str) -> torch.Tensor:
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return encodings.input_ids[0]


if __name__ == "__main__":
    print("Embedding dimension:", EMBED_DIM)
    test_texts = [
        "Hello world",
        "Neural networks are fascinating",
        "OpenAI provides powerful APIs",
    ]
    embeddings = embed(test_texts)
    print("Embeddings shape:", embeddings.shape)
    print("First embedding vector:", embeddings[0][:5])  # Print first 5 dimensions of first embedding
