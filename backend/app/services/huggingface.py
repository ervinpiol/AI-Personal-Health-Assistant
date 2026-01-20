from transformers import pipeline, Text2TextGenerationPipeline
from fastapi.concurrency import run_in_threadpool

# Initialize pipeline once
generator: Text2TextGenerationPipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# Simple red-flag check
RED_FLAGS = ["chest pain", "severe bleeding", "difficulty breathing"]

def contains_red_flag(symptoms: str) -> bool:
    return any(flag in symptoms.lower() for flag in RED_FLAGS)

async def get_health_advice(symptoms: str) -> str:
    if contains_red_flag(symptoms):
        return "Red-flag symptoms detected. Please seek emergency medical attention immediately."

    prompt = (
        "You are a helpful medical assistant. "
        "Provide concise, general health advice for the symptoms described. "
        "Do NOT repeat the symptoms verbatim. "
        "Return your response in 2-3 sentences and always recommend consulting a doctor. "
        f"Symptoms: {symptoms}"
    )

    results = await run_in_threadpool(
        generator,
        prompt,
        max_new_tokens=150,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )

    return results[0]["generated_text"]

