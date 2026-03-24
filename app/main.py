from fastapi import FastAPI, HTTPException
from app.models import QueryRequest
from app.rag_engine import prepare_index, answer_query

app = FastAPI(title="RAG PDF Chatbot API")


@app.get("/")
def root():
    return {"message": "RAG PDF Chatbot API is running"}


@app.post("/build-index")
def build_index():
    try:
        return prepare_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        return answer_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))