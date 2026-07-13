"""
Provider-agnostic response schemas.

Aviationstack and AirLabs return completely different JSON shapes for the
same real-world concept (a flight's status). Every provider service maps its
raw response into these models, so routers and the frontend only ever see
one consistent shape regardless of which upstream answered the request.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FlightStatus(str, Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    LANDED = "landed"
    CANCELLED = "cancelled"
    INCIDENT = "incident"
    DIVERTED = "diverted"
    DELAYED = "delayed"
    UNKNOWN = "unknown"


class AirportLeg(BaseModel):
    """
    One end (departure or arrival) of a flight.

    `scheduled` / `estimated` / `actual` are always UTC - that's the one
    canonical value both providers can agree on. `timezone` is the IANA
    zone name for *this specific airport* (e.g. "Asia/Kolkata" for a
    departure from DEL, "Europe/London" for an arrival at LHR). A client
    renders each leg by formatting the UTC instant in that leg's own
    timezone - never the viewer's browser timezone - which is what makes
    a DEL -> LHR flight correctly show "10:00 PM IST" for departure and
    "6:30 AM BST" for arrival, in the same response.
    """
    airport: str | None = None
    iata: str | None = None
    icao: str | None = None
    terminal: str | None = None
    gate: str | None = None
    timezone: str | None = None
    scheduled: datetime | None = None
    estimated: datetime | None = None
    actual: datetime | None = None
    delay_minutes: int | None = None


class Airline(BaseModel):
    name: str | None = None
    iata: str | None = None
    icao: str | None = None


class Aircraft(BaseModel):
    registration: str | None = None
    iata_type: str | None = None


class LivePosition(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    speed_horizontal: float | None = None
    direction: float | None = None
    is_ground: bool | None = None
    updated: datetime | None = None


class FlightRecord(BaseModel):
    """The normalized, unified representation of a single flight."""
    flight_number: str | None = None
    flight_iata: str | None = None
    flight_icao: str | None = None
    status: FlightStatus = FlightStatus.UNKNOWN
    airline: Airline = Field(default_factory=Airline)
    departure: AirportLeg = Field(default_factory=AirportLeg)
    arrival: AirportLeg = Field(default_factory=AirportLeg)
    aircraft: Aircraft = Field(default_factory=Aircraft)
    live: LivePosition | None = None
    source: str | None = None  # which provider answered: "airlabs" | "aviationstack"


class FlightSearchResponse(BaseModel):
    count: int
    source: str
    fallback_used: bool = False
    results: list[FlightRecord]


class FlightStatusResponse(BaseModel):
    source: str
    fallback_used: bool = False
    flight: FlightRecord | None = None


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, bool]
