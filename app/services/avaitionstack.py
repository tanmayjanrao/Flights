"""
Aviationstack provider.

Docs: https://aviationstack.com/documentation
Endpoint used: GET /v1/flights

Sample raw shape (trimmed) this maps from:
{
  "data": [{
    "flight_date": "2024-01-01",
    "flight_status": "active",
    "departure": {"airport": "...", "iata": "JFK", "terminal": "4", "gate": "B6",
                  "delay": 13, "scheduled": "...", "estimated": "...", "actual": "..."},
    "arrival":   {"airport": "...", "iata": "LHR", ... },
    "airline":   {"name": "American Airlines", "iata": "AA", "icao": "AAL"},
    "flight":    {"number": "1004", "iata": "AA1004", "icao": "AAL1004"},
    "aircraft":  {"registration": "...", "iata": "..."},
    "live":      {"latitude": ..., "longitude": ..., "altitude": ..., "is_ground": false}
  }]
}
"""
import httpx

from app.config import settings
from app.models.schemas import (
    Aircraft,
    Airline,
    AirportLeg,
    FlightRecord,
    FlightStatus,
    LivePosition,
)
from app.services.base import (
    FlightProvider,
    FlightSearchParams,
    ProviderAuthError,
    ProviderRateLimited,
    ProviderUnavailable,
)

_STATUS_MAP = {
    "scheduled": FlightStatus.SCHEDULED,
    "active": FlightStatus.ACTIVE,
    "landed": FlightStatus.LANDED,
    "cancelled": FlightStatus.CANCELLED,
    "incident": FlightStatus.INCIDENT,
    "diverted": FlightStatus.DIVERTED,
}


def _leg(raw: dict | None) -> AirportLeg:
    raw = raw or {}
    return AirportLeg(
        airport=raw.get("airport"),
        iata=raw.get("iata"),
        icao=raw.get("icao"),
        terminal=raw.get("terminal"),
        gate=raw.get("gate"),
        scheduled=raw.get("scheduled"),
        estimated=raw.get("estimated"),
        actual=raw.get("actual"),
        delay_minutes=raw.get("delay"),
    )


def _map_record(raw: dict) -> FlightRecord:
    flight = raw.get("flight") or {}
    live_raw = raw.get("live") or {}
    live = None
    if live_raw:
        live = LivePosition(
            latitude=live_raw.get("latitude"),
            longitude=live_raw.get("longitude"),
            altitude=live_raw.get("altitude"),
            speed_horizontal=live_raw.get("speed_horizontal"),
            direction=live_raw.get("direction"),
            is_ground=live_raw.get("is_ground"),
            updated=live_raw.get("updated"),
        )

    return FlightRecord(
        flight_number=flight.get("number"),
        flight_iata=flight.get("iata"),
        flight_icao=flight.get("icao"),
        status=_STATUS_MAP.get(raw.get("flight_status"), FlightStatus.UNKNOWN),
        airline=Airline(**(raw.get("airline") or {})),
        departure=_leg(raw.get("departure")),
        arrival=_leg(raw.get("arrival")),
        aircraft=Aircraft(
            registration=(raw.get("aircraft") or {}).get("registration"),
            iata_type=(raw.get("aircraft") or {}).get("iata"),
        ),
        live=live,
        source="aviationstack",
    )


class AviationstackProvider(FlightProvider):
    name = "aviationstack"

    def is_configured(self) -> bool:
        return bool(settings.aviationstack_api_key)

    async def _request(self, params: dict) -> list[dict]:
        query = {"access_key": settings.aviationstack_api_key, **params}
        url = f"{settings.aviationstack_base_url}/flights"

        try:
            async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
                resp = await client.get(url, params=query)
        except httpx.RequestError as exc:
            raise ProviderUnavailable(f"aviationstack network error: {exc}") from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError("aviationstack rejected the API key")
        if resp.status_code == 429:
            raise ProviderRateLimited("aviationstack monthly quota exceeded")
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"aviationstack upstream error {resp.status_code}")

        body = resp.json()

        # Aviationstack returns HTTP 200 with an "error" object for bad requests
        # (e.g. invalid key, plan limits) instead of a proper status code.
        if "error" in body:
            code = (body["error"] or {}).get("code", "")
            message = (body["error"] or {}).get("message", "unknown error")
            if code in ("rate_limit_reached", "usage_limit_reached"):
                raise ProviderRateLimited(f"aviationstack: {message}")
            if code in ("invalid_access_key", "missing_access_key"):
                raise ProviderAuthError(f"aviationstack: {message}")
            raise ProviderUnavailable(f"aviationstack: {message}")

        return body.get("data") or []

    async def search_flights(self, params: FlightSearchParams) -> list[FlightRecord]:
        query = {
            "flight_iata": params.flight_iata,
            "flight_icao": params.flight_icao,
            "dep_iata": params.dep_iata,
            "arr_iata": params.arr_iata,
            "airline_iata": params.airline_iata,
            "flight_status": params.flight_status,
            "limit": params.limit,
        }
        query = {k: v for k, v in query.items() if v is not None}
        raw_flights = await self._request(query)
        return [_map_record(f) for f in raw_flights]

    async def get_flight_status(self, flight_iata: str) -> FlightRecord | None:
        raw_flights = await self._request({"flight_iata": flight_iata, "limit": 1})
        if not raw_flights:
            return None
        return _map_record(raw_flights[0])
