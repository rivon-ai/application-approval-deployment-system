# API-specific configuration (e.g., host, port)
import os

# Configuration for the FastAPI app
class Config:
    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", 8000))

config = Config()
