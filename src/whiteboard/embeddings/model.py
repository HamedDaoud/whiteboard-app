# src/whiteboard/embeddings/model.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


class Embedder:
    """
    Thin wrapper over Sentence-Transformers with sensible defaults:

    - Model name from EMBED_MODEL (default: all-MiniLM-L6-v2)
    - Automatic device: cuda if available, else cpu (can override via EMBED_DEVICE)
    - Optional vector L2-normalization (recommended for cosine search in Milvus)
    - Batch encoding with float32 output

    Public attributes:
      - model_name: str
      - max_seq_length: int
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        normalize: bool = True,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> None:
        # Resolve config from environment with sane defaults
        self.model_name: str = model_name or _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Device selection
        env_device = _env("EMBED_DEVICE")
        self.device = device or env_device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate model
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Respect max sequence length:
        # - If max_seq_length is provided (arg or env), set it explicitly.
        # - Else keep model default (many ST BERT-family models default to 256/384/512).
        env_max_len = _env("EMBED_MAX_SEQ_LENGTH")
        if max_seq_length is None and env_max_len:
            try:
                max_seq_length = int(env_max_len)
            except ValueError:
                pass
        if isinstance(max_seq_length, int) and max_seq_length > 0:
            # sentence-transformers exposes this property for truncation during encode()
            self.model.max_seq_length = max_seq_length

        self.max_seq_length: int = int(getattr(self.model, "max_seq_length", 256))

        # Encoding params
        env_bs = _env("EMBED_BATCH_SIZE")
        if batch_size is None and env_bs:
            try:
                batch_size = int(env_bs)
            except ValueError:
                pass
        self.batch_size = int(batch_size) if isinstance(batch_size, int) and batch_size > 0 else 32

        env_norm = _env("EMBED_NORMALIZE")
        if env_norm is not None:
            # Accept "0/1", "true/false", "yes/no"
            s = env_norm.strip().lower()
            normalize = s in ("1", "true", "yes", "y", "on")
        self.normalize = bool(normalize)

    # ------------- public API -------------

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode a batch of texts -> np.ndarray [n, d] float32.
        Applies optional L2 normalization (recommended for cosine).
        """
        # sentence-transformers can take iterables directly
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll handle normalization explicitly
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        if self.normalize:
            vectors = _l2_normalize(vectors)
        return vectors

    def encode_one(self, text: str) -> np.ndarray:
        """
        Encode a single text -> np.ndarray [d] float32.
        """
        vec = self.encode([text])
        return vec[0]


# ------------- utils -------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: each vector becomes unit length.
    Safe for zero vectors.
    """
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) + eps)
        return (x / denom).astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / norms).astype(np.float32, copy=False)