from __future__ import annotations

"""Vocabulary builders and adaptive vocabulary utilities for AHOT.

This module provides a small, production oriented implementation of the
advanced vocabulary system described in the task instructions.  The goal of
this implementation is not to be a state of the art tokenizer but to supply a
clean interface that can later be extended.  The builders expose a common
API so that the rest of the AHOT stack can request a vocabulary without
knowing which algorithm is used under the hood.

The design intentionally keeps the algorithms lightweight – the focus of the
project is hardware awareness and adaptability rather than perfect
sub‑word segmentation.  Each builder produces a simple token->id mapping and
reports the algorithm it implements.  The mapping can then be consumed by the
`AHOTTokenizer` or any other component requiring a vocabulary.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from abc import ABC, abstractmethod

# The HardwareProfile dataclass is defined in ``src/ahot/__init__.py`` and is
# re‑used here for hardware aware logic.
try:  # pragma: no cover - import convenience
    from . import HardwareProfile  # type: ignore
except Exception:  # pragma: no cover - during docs generation etc.
    HardwareProfile = object  # type: ignore


@dataclass
class TokenizerVocabulary:
    """A tiny container for vocabularies produced by builders.

    Attributes
    ----------
    token_to_id: Mapping from token string to integer id.
    algorithm: Name of the algorithm that produced this vocabulary.
    domain: Optional domain of the vocabulary.  ``None`` indicates a generic
        vocabulary.
    """

    token_to_id: Dict[str, int]
    algorithm: str
    domain: Optional[str] = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.token_to_id)


class VocabularyBuilder(ABC):
    """Base class for building vocabularies from text corpora."""

    #: human readable algorithm name, e.g. ``"bpe"`` or ``"wordpiece"``
    algorithm: str = "unknown"

    def __init__(self, hardware_profile: Optional[HardwareProfile] = None):
        self.hardware_profile = hardware_profile

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_hardware_profile(cls, profile: HardwareProfile) -> "VocabularyBuilder":
        """Select an appropriate builder for the provided hardware profile.

        The heuristic is intentionally straightforward:

        * Servers or machines with a GPU use ``SentencePiece`` because it tends
          to perform well for multilingual data and benefits from additional
          resources.
        * Desktops/laptops use ``WordPiece``.
        * Very small devices fall back to a light‑weight ``BPE`` implementation.
        """

        if getattr(profile, "gpu_available", False) and getattr(profile, "memory_gb", 0) >= 16:
            return SentencePieceVocabularyBuilder(profile)
        if getattr(profile, "memory_gb", 0) >= 8:
            return WordPieceVocabularyBuilder(profile)
        return BPEVocabularyBuilder(profile)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    @abstractmethod
    def build_from_corpus(self, texts: Iterable[str], target_vocab_size: int) -> TokenizerVocabulary:
        """Build a vocabulary from ``texts``.

        Implementations should respect ``target_vocab_size`` and return a
        :class:`TokenizerVocabulary` instance.  The returned vocabulary **must**
        contain no more than ``target_vocab_size`` tokens.
        """

    def max_vocab_size_for_memory(self, embedding_dim: int = 256) -> int:
        """Best effort upper bound of vocabulary size for the given hardware."""
        if not self.hardware_profile:
            return 32000
        sizer = MemoryAwareVocabSizer()
        return sizer.calculate_optimal_vocab_size(
            self.hardware_profile.memory_gb,
            embedding_dim,
            getattr(self.hardware_profile, "gpu_available", False),
        )


# ----------------------------------------------------------------------
# Concrete builders
# ----------------------------------------------------------------------

class _BaseAlgorithm(VocabularyBuilder):
    """Utility class containing trivial implementations used by builders."""

    def _base_build(self, texts: Iterable[str], target_vocab_size: int) -> Dict[str, int]:
        # Basic algorithm: collect characters and common pairs until the target
        # size is met.  This is *not* an exact implementation of the respective
        # algorithms but provides deterministic and inexpensive behaviour for
        # demonstration and testing purposes.
        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for text in texts:
            for ch in text:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
                    if len(vocab) >= target_vocab_size:
                        return vocab
        # Optionally add simple bigrams for a little more variety.
        for text in texts:
            for i in range(len(text) - 1):
                pair = text[i : i + 2]
                if pair not in vocab:
                    vocab[pair] = len(vocab)
                    if len(vocab) >= target_vocab_size:
                        return vocab
        return vocab


class BPEVocabularyBuilder(_BaseAlgorithm):
    algorithm = "bpe"

    def build_from_corpus(self, texts: Iterable[str], target_vocab_size: int) -> TokenizerVocabulary:
        vocab = self._base_build(texts, target_vocab_size)
        return TokenizerVocabulary(vocab, self.algorithm)


class WordPieceVocabularyBuilder(_BaseAlgorithm):
    algorithm = "wordpiece"

    def build_from_corpus(self, texts: Iterable[str], target_vocab_size: int) -> TokenizerVocabulary:
        # WordPiece normally starts with word level tokens.  We mimic that by
        # first splitting on whitespace and adding full words before applying the
        # base algorithm for residual capacity.
        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
                    if len(vocab) >= target_vocab_size:
                        return TokenizerVocabulary(vocab, self.algorithm)
        remaining = target_vocab_size - len(vocab)
        if remaining > 0:
            extra = self._base_build(texts, target_vocab_size)
            for token, idx in extra.items():
                if token not in vocab and len(vocab) < target_vocab_size:
                    vocab[token] = len(vocab)
        return TokenizerVocabulary(vocab, self.algorithm)


class SentencePieceVocabularyBuilder(_BaseAlgorithm):
    algorithm = "sentencepiece"

    def build_from_corpus(self, texts: Iterable[str], target_vocab_size: int) -> TokenizerVocabulary:
        # SentencePiece is byte based and language agnostic.  Our lightweight
        # variant simply applies the base algorithm; the distinction is mainly
        # exposed through the ``algorithm`` attribute.
        vocab = self._base_build(texts, target_vocab_size)
        return TokenizerVocabulary(vocab, self.algorithm)


# ----------------------------------------------------------------------
# Memory aware sizing
# ----------------------------------------------------------------------

class MemoryAwareVocabSizer:
    """Calculate an appropriate vocabulary size for a given memory budget."""

    def calculate_optimal_vocab_size(
        self,
        available_memory_gb: float,
        embedding_dim: int,
        gpu_available: bool,
    ) -> int:
        if available_memory_gb <= 0:
            raise ValueError("available_memory_gb must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        # Base limits depending on memory tiers.
        if available_memory_gb < 4:
            lower, upper = 8000, 16000
        elif available_memory_gb <= 16:
            lower, upper = 16000, 32000
        else:
            lower, upper = 32000, 64000

        # Estimate how many tokens can fit if we spend ~10% of memory on the
        # embedding table.  Each embedding uses 4 bytes (float32).
        bytes_available = available_memory_gb * (1024 ** 3) * 0.10
        token_bytes = embedding_dim * 4
        mem_cap = int(bytes_available // token_bytes)
        size = max(lower, min(upper, mem_cap))

        # GPU memory is generally faster; when available we allow a small bonus
        # (10%).
        if gpu_available:
            size = int(size * 1.1)
        return size


# ----------------------------------------------------------------------
# Domain specific adaptation
# ----------------------------------------------------------------------

class DomainClassifier:
    """Very small keyword based domain detector."""

    _KEYWORDS = {
        "medical": ["patient", "diagnosis", "treatment", "disease"],
        "legal": ["court", "law", "contract", "plaintiff"],
        "technical": ["algorithm", "hardware", "software", "data"],
    }

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        for domain, words in self._KEYWORDS.items():
            if any(w in text_lower for w in words):
                return domain
        return "general"


class DomainAdaptiveVocabulary:
    """Manage domain specific vocabularies built on demand."""

    def __init__(self, base_vocabulary: TokenizerVocabulary, hardware_profile: HardwareProfile):
        self.base_vocab = base_vocabulary
        self.hardware_profile = hardware_profile
        self.domain_vocabs: Dict[str, TokenizerVocabulary] = {}
        self.domain_detector = DomainClassifier()

    # Public API -------------------------------------------------------
    def adapt_to_text(self, text: str) -> TokenizerVocabulary:
        domain = self.domain_detector.classify(text)
        return self.get_or_create_domain_vocab(domain)

    # Internal helpers -------------------------------------------------
    def get_or_create_domain_vocab(self, domain: str) -> TokenizerVocabulary:
        if domain in self.domain_vocabs:
            return self.domain_vocabs[domain]

        # For new domains we extend the base vocabulary with a simple domain
        # token.  In a real system this is where a domain specific corpus would
        # be used.
        new_tokens = dict(self.base_vocab.token_to_id)
        domain_token = f"<domain:{domain}>"
        new_tokens[domain_token] = len(new_tokens)
        vocab = TokenizerVocabulary(new_tokens, self.base_vocab.algorithm, domain)
        self.domain_vocabs[domain] = vocab
        return vocab


__all__ = [
    "TokenizerVocabulary",
    "VocabularyBuilder",
    "BPEVocabularyBuilder",
    "WordPieceVocabularyBuilder",
    "SentencePieceVocabularyBuilder",
    "MemoryAwareVocabSizer",
    "DomainAdaptiveVocabulary",
    "DomainClassifier",
]
