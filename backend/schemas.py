# backend/schemas.py

from pydantic import BaseModel

class NewsRequest(BaseModel):
    text: str