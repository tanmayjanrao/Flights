"""
QA provider contract.

Mirrors services/base.py for flight providers: anything capable of analysing
an airline customer support transcript and returning a normalized QAAnalysis
implements this interface.

The router and orchestration layer never know (or care) whether the analysis
came from a local Ollama model, Groq, OpenAI, or any future provider.
Swapping providers is therefore a configuration change, not a rewrite.
"""
from abc import ABC, abstractmethod

from app.models.qa_schemas import ChatTranscript, QAAnalysis


class QAProviderError(Exception):
    """Base class for any QA provider failure."""


class QAProviderUnavailable(QAProviderError):
    """Network failure, timeout, or provider unavailable."""


class QAProviderAuthError(QAProviderError):
    """Authentication failed."""


class QAProviderRateLimited(QAProviderError):
    """The provider rejected the request due to quota or rate limits."""


class QAProviderBadResponse(QAProviderError):
    """Provider returned data that could not be parsed into QAAnalysis."""


class QAProvider(ABC):
    """
    Abstract interface implemented by every QA backend.
    """

    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Returns True if this provider is correctly configured and can
        accept requests.
        """

    @abstractmethod
    async def analyze(self, transcript: ChatTranscript) -> QAAnalysis:
        """
        Analyse a customer support transcript and return a normalized
        QAAnalysis object.
        """