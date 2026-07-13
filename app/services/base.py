"""
Provider contract.

Every flight-data provider (Aviationstack, AirLabs, or anything added later)
implements this interface. The rest of the app (routers, FlightService
orchestrator) only ever talks to this abstraction, which is what makes
swapping/adding providers a service-layer change rather than a rewrite.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models.schemas import FlightRecord


class ProviderError(Exception):
    """Base class for any failure talking to an upstream flight API."""


class ProviderRateLimited(ProviderError):
    """Raised when the provider's quota is exhausted (HTTP 429 or plan limit)."""


class ProviderUnavailable(ProviderError):
    """Raised for network errors, timeouts, or 5xx responses upstream."""


class ProviderAuthError(ProviderError):
    """Raised for bad/missing API keys (401/403)."""


@dataclass
class FlightSearchParams:
    """Common search filters, translated by each provider into its own query params."""
    flight_iata: str | None = None
    flight_icao: str | None = None
    dep_iata: str | None = None
    arr_iata: str | None = None
    airline_iata: str | None = None
    flight_status: str | None = None  # scheduled/active/landed/cancelled...
    limit: int = 10


class FlightProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has the API key it needs."""

    @abstractmethod
    async def search_flights(self, params: FlightSearchParams) -> list[FlightRecord]:
        """Search flights matching the given filters."""

    @abstractmethod
    async def get_flight_status(self, flight_iata: str) -> FlightRecord | None:
        """Look up the live/most-recent status of a single flight by IATA code."""
