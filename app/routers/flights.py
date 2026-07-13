from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import FlightSearchResponse, FlightStatusResponse, HealthResponse
from app.services.flight_service import (
    AllProvidersFailedError,
    flight_service,
)
from app.services.base import FlightSearchParams

router = APIRouter(prefix="/api/flights", tags=["flights"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", providers=flight_service.provider_status())


@router.get("/search", response_model=FlightSearchResponse)
async def search_flights(
    dep_iata: str | None = Query(None, min_length=3, max_length=4, description="Departure airport IATA code, e.g. JFK"),
    arr_iata: str | None = Query(None, min_length=3, max_length=4, description="Arrival airport IATA code, e.g. LHR"),
    airline_iata: str | None = Query(None, min_length=2, max_length=3, description="Airline IATA code, e.g. AA"),
    flight_status: str | None = Query(None, description="scheduled | active | landed | cancelled"),
    limit: int = Query(10, ge=1, le=50),
):
    if not any([dep_iata, arr_iata, airline_iata, flight_status]):
        raise HTTPException(status_code=400, detail="Provide at least one filter: dep_iata, arr_iata, airline_iata, or flight_status")

    params = FlightSearchParams(
        dep_iata=dep_iata,
        arr_iata=arr_iata,
        airline_iata=airline_iata,
        flight_status=flight_status,
        limit=limit,
    )

    try:
        results, source, fallback_used = await flight_service.search_flights(params)
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return FlightSearchResponse(count=len(results), source=source, fallback_used=fallback_used, results=results)


@router.get("/status/{flight_iata}", response_model=FlightStatusResponse)
async def get_flight_status(flight_iata: str):
    try:
        record, source, fallback_used = await flight_service.get_flight_status(flight_iata.upper())
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if record is None:
        raise HTTPException(status_code=404, detail=f"No data found for flight {flight_iata.upper()}")

    return FlightStatusResponse(source=source, fallback_used=fallback_used, flight=record)
