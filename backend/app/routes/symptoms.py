from fastapi import APIRouter
from app.core.config import settings
from app.services.huggingface import get_health_advice
from app.schemas.symptoms import SymptomRequest, SymptomResponse


router = APIRouter(
    prefix=f"{settings.API_V1_STR}/symptoms",
    tags=["symptoms"],
)

@router.post("", response_model=SymptomResponse)
async def analyze_symptoms(request: SymptomRequest):
    """
    Analyze symptoms using Hugging Face LLM
    """
    advice = await get_health_advice(request.symptoms)

    return {
        "symptoms": request.symptoms,
        "ai_advice": advice,
        "disclaimer": "This is not medical advice. Consult a healthcare professional."
    }