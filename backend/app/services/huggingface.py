from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sweatSmile/Gemma-2-2B-MedicalQA-Assistant"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Simple red-flag check
RED_FLAGS = ["chest pain", "severe bleeding", "difficulty breathing"]

def contains_red_flag(symptoms: str) -> bool:
    return any(flag in symptoms.lower() for flag in RED_FLAGS)

async def get_health_advice(symptoms: str) -> str:
    if contains_red_flag(symptoms):
        return "Red-flag symptoms detected. Please seek emergency medical attention immediately."

    prompt = (
        "You are a medical assistant.\n"
        "Give general, non-diagnostic health advice.\n"
        "Do NOT list symptoms.\n"
        "Answer in 2-3 complete sentences.\n"
        "Always recommend seeing a doctor.\n\n"
        f"User complaint: {symptoms}\n"
        "Advice:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response
