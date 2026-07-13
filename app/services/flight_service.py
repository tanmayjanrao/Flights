"""
FlightService - the single entry point routers should call.

This is where the "compatible with both API keys" requirement actually
lives: it tries the configured primary provider first and, only if that
provider is unconfigured, out of quota, or unreachable, transparently
retries the same request against the other one. Callers (routers, tests)
never need to know which upstream answered - the response says so via
`source` / `fallback_used`.
"""
import logging

from app.config import settings
from app.models.schemas import FlightRecord
from app.services.airlabs import AirLabsProvider
from app.services.aviationstack import AviationstackProvider
from app.services.base import FlightProvider, FlightSearchParams, ProviderError

logger = logging.getLogger("flights.service")


class AllProvidersFailedError(Exception):
    """Raised when neither provider could answer the request."""


class FlightService:
    def __init__(self) -> None:
        self._providers: dict[str, FlightProvider] = {
            "airlabs": AirLabsProvider(),
            "aviationstack": AviationstackProvider(),
        }

    def _ordered_providers(self) -> list[FlightProvider]:
        """Primary provider first (from settings), then whatever's left, configured ones only."""
        order = [settings.primary_provider] + [
            name for name in self._providers if name != settings.primary_provider
        ]
        return [self._providers[name] for name in order if self._providers[name].is_configured()]

    async def search_flights(self, params: FlightSearchParams) -> tuple[list[FlightRecord], str, bool]:
        providers = self._ordered_providers()
        if not providers:
            raise AllProvidersFailedError("No flight providers are configured. Set an API key in .env")

        last_error: Exception | None = None
        for i, provider in enumerate(providers):
            try:
                results = await provider.search_flights(params)
                return results, provider.name, i > 0
            except ProviderError as exc:
                logger.warning("provider %s failed: %s", provider.name, exc)
                last_error = exc
                continue

        raise AllProvidersFailedError(str(last_error) if last_error else "all providers failed")

    async def get_flight_status(self, flight_iata: str) -> tuple[FlightRecord | None, str, bool]:
        providers = self._ordered_providers()
        if not providers:
            raise AllProvidersFailedError("No flight providers are configured. Set an API key in .env")

        last_error: Exception | None = None
        for i, provider in enumerate(providers):
            try:
                result = await provider.get_flight_status(flight_iata)
                return result, provider.name, i > 0
            except ProviderError as exc:
                logger.warning("provider %s failed: %s", provider.name, exc)
                last_error = exc
                continue

        raise AllProvidersFailedError(str(last_error) if last_error else "all providers failed")

    def provider_status(self) -> dict[str, bool]:
        return {name: p.is_configured() for name, p in self._providers.items()}


# Single shared instance - httpx clients are created per-request so this is safe to reuse.
flight_service = FlightService()
