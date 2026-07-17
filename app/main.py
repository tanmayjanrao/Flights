from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import flights, pages, qa

app = FastAPI(
    title="Flights API",
    description="Real-time flight tracking backed by AirLabs and Aviationstack, "
                 "with automatic fallback between providers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(flights.router)
app.include_router(qa.router)
app.include_router(pages.router)
