from fastapi import FastAPI

app = FastAPI(
    title="Flights API",
    description="Real-time flight tracking using Aviation Edge API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Flights API is running"
    }