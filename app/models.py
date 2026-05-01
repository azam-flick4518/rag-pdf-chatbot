from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class UploadResponse(BaseModel):
    message: str
    filename: str