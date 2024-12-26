# Main entry point for the FastAPI server
from fastapi import FastAPI
from .api import router
from .config import config

app = FastAPI()

# Include the API router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)

    