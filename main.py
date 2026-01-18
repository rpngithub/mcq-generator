from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Multilingual Q&A Generator")

# Load open-source models
# Multilingual Q&A (mT5 or similar)
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")

# Multilingual question generation (optional, experimental)
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qa-qg-hl")

# Translation (OPUS-MT, example: English to French)
from transformers import MarianMTModel, MarianTokenizer
def get_translation_pipeline(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    def translate(text):
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    return translate

class QARequest(BaseModel):
    context: str
    question: str
    answer_language: str = "en"  # answer in English by default

class QGRequest(BaseModel):
    context: str
    question_language: str = "en"

@app.post("/answer")
def get_answer(q: QARequest):
    # Get answer in English
    result = qa_pipeline(question=q.question, context=q.context)
    answer = result.get("answer", "")
    # Translate if needed
    if q.answer_language != "en":
        try:
            translate = get_translation_pipeline("en", q.answer_language)
            answer = translate(answer)
        except Exception:
            return {"error": f"Translation not available for 'en' to '{q.answer_language}'"}
    return {"text": answer}

@app.post("/question")
def generate_question(req: QGRequest):
    # Generate question from context (returns English by default)
    result = qg_pipeline(f"generate question: {req.context}")
    question = result[0]['generated_text']
    # Translate if needed
    if req.question_language != "en":
        try:
            translate = get_translation_pipeline("en", req.question_language)
            question = translate(question)
        except Exception:
            return {"error": f"Translation not available for 'en' to '{req.question_language}'"}
    return {"question": question}