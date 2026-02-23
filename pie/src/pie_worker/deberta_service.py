"""
DeBERTa NLI Service for PIE.

A lightweight model service that loads a DeBERTa-based NLI model
and handles classify_batch RPC requests for entailment classification.

This service is registered as a second model alongside the LLM,
enabling inferlets to perform batched NLI classification via the
classify WIT interface.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import message

logger = logging.getLogger(__name__)

# NLI label mapping for cross-encoder/nli-deberta-v3-large
LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}


class DeBERTaService:
    """DeBERTa NLI classification service.

    Loads the model on initialization and handles RPC requests
    from the Rust runtime.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large",
                 device: str = "cpu"):
        """Initialize and load the DeBERTa NLI model.

        Args:
            model_name: HuggingFace model ID for the NLI model.
            device: Device to run inference on ("cpu" or "cuda:N").
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading DeBERTa NLI model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"DeBERTa NLI model loaded ({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)")

    def handshake(self, req: message.HandshakeRequest) -> message.HandshakeResponse:
        """Handle handshake — return model info with NLI traits."""
        return message.HandshakeResponse(
            version=req.version,
            model_name=self.model_name,
            model_traits=["nli", "classification"],
            model_description="DeBERTa-v3-large NLI classifier",
            prompt_template="",
            prompt_template_type="none",
            prompt_stop_tokens=[],
            kv_page_size=0,
            max_batch_tokens=0,
            max_batch_size=0,
            resources={},
            tokenizer_num_vocab=0,
            tokenizer_merge_table={},
            tokenizer_special_tokens={},
            tokenizer_split_regex="",
            tokenizer_escape_non_printable=False,
            tokenizer_sentencepiece_space=False,
        )

    def classify_batch(self, req: message.ClassifyBatchRequest) -> message.ClassifyBatchResponse:
        """Classify a batch of (premise, hypothesis) pairs.

        Uses a single batched forward pass for efficiency.
        """
        if not req.pairs:
            return message.ClassifyBatchResponse(results=[])

        premises = [p for p, _ in req.pairs]
        hypotheses = [h for _, h in req.pairs]

        # Batch tokenize all pairs at once
        inputs = self.tokenizer(
            premises, hypotheses,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)

        # Single batched forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        # Extract predictions
        results = []
        for i in range(len(req.pairs)):
            pred_idx = probs[i].argmax().item()
            score = probs[i][pred_idx].item()
            results.append({
                "label": LABEL_MAP.get(pred_idx, "unknown"),
                "score": float(score),
            })

        return message.ClassifyBatchResponse(results=results)

    def query(self, req: message.QueryRequest) -> message.QueryResponse:
        """Handle query request (for debug/introspection)."""
        match req.query:
            case "ping":
                return message.QueryResponse(value="pong")
            case _:
                return message.QueryResponse(value=f"DeBERTa NLI service: {self.model_name}")
