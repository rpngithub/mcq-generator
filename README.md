# Multilingual Q&A Generator (Backend, FastAPI)

This backend uses only open-source models to generate and answer questions in **multiple languages**.

## Features

- **Multilingual question answering** with [deepset/xlm-roberta-base-squad2](https://huggingface.co/deepset/xlm-roberta-base-squad2)
- **Multilingual question generation** with [givealittle/mT5-large-qa-qg-hl](https://huggingface.co/givealittle/mT5-large-qa-qg-hl)
- **Translation** for answer/question output via [Helsinki-NLP/OPUS-MT models](https://huggingface.co/Helsinki-NLP), supporting dozens of language pairs

## Usage

1. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the API:**
    ```bash
    uvicorn main:app --reload
    ```

3. **Endpoints:**
    - `/answer`: POST Q&A request. Returns answer in your requested language. Both `context` and `question` can be in any supported language.
    - `/question`: POST context, generates a question in your requested language. Context can be in any supported language.

### Example: `/answer` endpoint

POST
```json
{
    "context": "La Tour Eiffel est un monument célèbre à Paris construit en 1889.",
    "question": "En quelle année a été construite la Tour Eiffel ?",
    "answer_language": "fr"
}
```

### Example: `/question` endpoint

POST
```json
{
    "context": "La Tour Eiffel est un monument célèbre à Paris construit en 1889.",
    "question_language": "fr"
}
```

## Notes

- The models now support **both question answering and question generation from context in a wide variety of languages** (see model cards for complete lists).
- For language output not directly supported, translation is provided via OPUS-MT.
- All models and dependencies are **100% open source**.

---