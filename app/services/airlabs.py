"""
AirLabs provider.

Docs: https://airlabs.co/docs/schedules
Endpoint used: GET /schedules - chosen over /flights (real-time positions only)
because it carries scheduled/estimated/actual times, delay minutes, terminal
and gate for both legs, which is what "flight status" actually means for this
app. Live lat/long tracking from /flights could be layered on top later (see
README "possible improvements").

Sample raw shape (trimmed) this maps from:
{
  "response": [{
    "flight_iata": "AA1004", "flight_icao": "AAL1004", "flight_number": "1004",
    "airline_iata": "AA", "airline_icao": "AAL",
    "dep_iata": "JFK", "dep_icao": "KJFK", "dep_terminal": "4", "dep_gate": "B6",
    "dep_time_utc": "2024-01-01 04:20", "dep_estimated_utc": "...", "dep_actual_utc": "...",
    "dep_delayed": 13,
    "arr_iata": "LHR", "arr_icao": "EGLL", "arr_terminal": "5", "arr_gate": "A2",
    "arr_time_utc": "...", "arr_estimated_utc": "...", "arr_actual_utc": "...",
    "arr_delayed": 0,
    "aircraft_icao": "A321", "status": "scheduled"
  }]
}
"""
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.models.schemas import Aircraft, Airline, AirportLeg, FlightRecord, FlightStatus
from app.services.airport_tz import airport_timezone
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
    "en-route": FlightStatus.ACTIVE,
    "landed": FlightStatus.LANDED,
    "cancelled": FlightStatus.CANCELLED,
    "incident": FlightStatus.INCIDENT,
    "diverted": FlightStatus.DIVERTED,
}


def _parse_utc(value: str | None) -> datetime | None:
    """
    AirLabs's `*_utc` fields are naive strings like "2021-07-14 23:53" with
    no offset marker - unlike Aviationstack, which includes "+00:00". If we
    let pydantic parse this straight, it becomes a naive datetime with no
    timezone attached, which is exactly the kind of ambiguity that caused
    the original bug. Parse it ourselves and attach UTC explicitly.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _dep_leg(raw: dict) -> AirportLeg:
    dep_iata = raw.get("dep_iata")
    return AirportLeg(
        iata=dep_iata,
        icao=raw.get("dep_icao"),
        terminal=raw.get("dep_terminal"),
        gate=raw.get("dep_gate"),
        # AirLabs gives no timezone name at all for /schedules - always resolved via lookup.
        timezone=airport_timezone(dep_iata),
        scheduled=_parse_utc(raw.get("dep_time_utc")),
        estimated=_parse_utc(raw.get("dep_estimated_utc")),
        actual=_parse_utc(raw.get("dep_actual_utc")),
        delay_minutes=raw.get("dep_delayed"),
    )


def _arr_leg(raw: dict) -> AirportLeg:
    arr_iata = raw.get("arr_iata")
    return AirportLeg(
        iata=arr_iata,
        icao=raw.get("arr_icao"),
        terminal=raw.get("arr_terminal"),
        gate=raw.get("arr_gate"),
        timezone=airport_timezone(arr_iata),
        scheduled=_parse_utc(raw.get("arr_time_utc")),
        estimated=_parse_utc(raw.get("arr_estimated_utc")),
        actual=_parse_utc(raw.get("arr_actual_utc")),
        delay_minutes=raw.get("arr_delayed"),
    )


def _map_record(raw: dict) -> FlightRecord:
    status_raw = (raw.get("status") or "").lower()
    return FlightRecord(
        flight_number=raw.get("flight_number"),
        flight_iata=raw.get("flight_iata"),
        flight_icao=raw.get("flight_icao"),
        status=_STATUS_MAP.get(status_raw, FlightStatus.UNKNOWN),
        airline=Airline(iata=raw.get("airline_iata"), icao=raw.get("airline_icao")),
        departure=_dep_leg(raw),
        arrival=_arr_leg(raw),
        aircraft=Aircraft(iata_type=raw.get("aircraft_icao")),
        live=None,  # see module docstring
        source="airlabs",
    )


class AirLabsProvider(FlightProvider):
    name = "airlabs"

    def is_configured(self) -> bool:
        return bool(settings.airlabs_api_key)

    async def _request(self, params: dict) -> list[dict]:
        query = {"api_key": settings.airlabs_api_key, **params}
        url = f"{settings.airlabs_base_url}/schedules"

        try:
            async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
                resp = await client.get(url, params=query)
        except httpx.RequestError as exc:
            raise ProviderUnavailable(f"airlabs network error: {exc}") from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError("airlabs rejected the API key")
        if resp.status_code == 429:
            raise ProviderRateLimited("airlabs monthly quota exceeded")
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"airlabs upstream error {resp.status_code}")

        body = resp.json()

        if isinstance(body, dict) and "error" in body:
            message = body["error"].get("message", "unknown error") if isinstance(body["error"], dict) else str(body["error"])
            lowered = message.lower()
            if "limit" in lowered or "quota" in lowered:
                raise ProviderRateLimited(f"airlabs: {message}")
            if "key" in lowered or "auth" in lowered:
                raise ProviderAuthError(f"airlabs: {message}")
            raise ProviderUnavailable(f"airlabs: {message}")

        return body.get("response") or []

    async def search_flights(self, params: FlightSearchParams) -> list[FlightRecord]:
        query = {
            "flight_iata": params.flight_iata,
            "dep_iata": params.dep_iata,
            "arr_iata": params.arr_iata,
            "airline_iata": params.airline_iata,
        }
        query = {k: v for k, v in query.items() if v is not None}
        raw_flights = await self._request(query)
        records = [_map_record(f) for f in raw_flights]
        if params.flight_status:
            records = [r for r in records if r.status.value == params.flight_status]
        return records[: params.limit]

    async def get_flight_status(self, flight_iata: str) -> FlightRecord | None:
        raw_flights = await self._request({"flight_iata": flight_iata})
        if not raw_flights:
            return None
        return _map_record(raw_flights[0])
