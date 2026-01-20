from pydantic import BaseModel

class SymptomRequest(BaseModel):
    symptoms: str

class SymptomResponse(BaseModel):
    symptoms: str
    ai_advice: str
    disclaimer: str