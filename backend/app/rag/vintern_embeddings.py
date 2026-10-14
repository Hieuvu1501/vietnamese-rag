import gc
import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class VinternEmbedder:
    """
    Text embeddings from the local 5CD-AI/Vintern-Embedding-1B model (text path only).
    Each text yields a [num_tokens, dim] tensor; VectorStore mean-pools it to one vector.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv(
            "VINTERN_MODEL", "5CD-AI/Vintern-Embedding-1B"
        )
        self.device = self._resolve_device(device)

        # Lazy-loaded on first use
        self._model = None
        self._processor = None

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> str:
        """Resolves the device string, supporting 'auto' detection."""
        requested = device or os.getenv("VINTERN_DEVICE", "auto")
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return requested

    def _load_model(self):
        """Lazily loads the model and processor on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        logger.info(
            "Loading Vintern-Embedding-1B model '%s' on device '%s'...",
            self.model_name,
            self.device,
        )

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Text embedding never runs the image encoder (forward only touches vision_model/mlp1 when
        # pixel_values is passed) or the next-token head (it reads language_model.model's hidden
        # states directly). Dropping them before moving to the device frees ~0.9 GB of the ~1.9 GB model.
        del model.vision_model, model.mlp1, model.language_model.lm_head
        gc.collect()
        self._model = model.eval().to(self.device)

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        logger.info("Vintern-Embedding-1B model loaded successfully.")

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def processor(self):
        self._load_model()
        return self._processor

    def embed_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Generates multi-vector embeddings for a list of text documents.

        Uses the processor's `process_docs` method to tokenize text documents,
        then runs through the model to produce per-token embeddings.

        Args:
            texts: List of document text strings.

        Returns:
            List of 2D tensors, each of shape [num_tokens, hidden_dim].
        """
        if not texts:
            return []

        batch = self.processor.process_docs(texts)
        batch = self._move_to_device(batch)

        with torch.no_grad():
            embeddings = self.model(**batch)

        # embeddings is a tensor of shape [batch, num_tokens, hidden_dim]
        # Split into list of per-document tensors
        return list(embeddings)

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Generates multi-vector embedding for a single query.

        Uses the processor's `process_queries` method for query-specific tokenization.

        Args:
            query: The query string.

        Returns:
            A 2D tensor of shape [num_tokens, hidden_dim].
        """
        batch = self.processor.process_queries([query])
        batch = self._move_to_device(batch)

        with torch.no_grad():
            embeddings = self.model(**batch)

        return embeddings[0]

    def _move_to_device(self, batch: dict) -> dict:
        """
        Moves a batch dictionary of tensors to the configured device.

        Per the model's official usage example, `input_ids` stay integer while
        `attention_mask` is cast to the model's compute dtype (not left as int) -
        this model consumes the mask as a multiplicative float weight.
        """
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        moved = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                if key == "input_ids":
                    moved[key] = val.to(device=self.device)
                elif val.is_floating_point() or key == "attention_mask":
                    moved[key] = val.to(device=self.device, dtype=dtype)
                else:
                    moved[key] = val.to(device=self.device)
            else:
                moved[key] = val
        return moved

