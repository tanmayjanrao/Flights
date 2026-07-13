"""
Airport -> timezone lookup.

Aviationstack hands back an IANA timezone name directly on every departure/
arrival object. AirLabs's /schedules endpoint does not - it only gives local
wall-clock time and UTC time as separate strings, with no zone name attached.

Rather than trust each provider inconsistently, every leg's timezone is
resolved through this single lookup (backed by the offline `airportsdata`
package - no network call, no quota cost). This also means Aviationstack's
own `timezone` field is only used as a first choice; if it's ever missing
(has happened per provider bug reports), this fills the gap the same way.
"""
from functools import lru_cache

import airportsdata

_AIRPORTS = airportsdata.load("IATA")  # {"DEL": {"tz": "Asia/Kolkata", ...}, ...}


@lru_cache(maxsize=4096)
def airport_timezone(iata_code: str | None) -> str | None:
    """Return the IANA timezone name for an airport's IATA code, or None if unknown."""
    if not iata_code:
        return None
    airport = _AIRPORTS.get(iata_code.strip().upper())
    return airport["tz"] if airport else None
