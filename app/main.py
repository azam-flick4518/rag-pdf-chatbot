from fastapi import FastAPI, HTTPException, UploadFile, File
from app.models import QueryRequest, UploadResponse
from app.rag_engine import (
    prepare_index,
    answer_query,
    add_document,
    list_documents,
    reset_index
)

app = FastAPI(title="RAG PDF Chatbot API")


@app.get("/")
def root():
    return {"message": "RAG PDF Chatbot API is running"}


@app.post("/build-index")
def build_index():
    """Backward-compatible: index the default data/sample.pdf"""
    try:
        return prepare_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a new PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        message = add_document(contents, file.filename)
        return UploadResponse(message=message, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QueryRequest):
    """Query across all indexed documents."""
    try:
        return answer_query(request.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    """List all indexed document names."""
    return {"documents": list_documents()}


@app.delete("/reset")
def reset():
    """Clear the entire index."""
    try:
        reset_index()
        return {"message": "Index cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))