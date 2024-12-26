 # Schema for health-check endpoint
from  pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
