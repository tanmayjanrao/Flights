# Centralized configuration (API keys, environment variables, settings).


from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("AVIATION_EDGE_API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    print("Warning: AVIATION_EDGE_API_KEY not found in .env")