from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import List
from scripts.auto_processor import AutoDocumentProcessor
from config import OPENAI_API_KEY

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root():
    # Return the HTML file
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/process-documents/")
async def process_documents(files: List[UploadFile]):
    try:
        # Convert uploaded files to the format expected by AutoDocumentProcessor
        documents = {}
        for file in files:
            if file.filename.endswith('.txt'):
                content = await file.read()
                documents[file.filename] = content.decode('utf-8')
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid text files uploaded")
        
        # Initialize processor with your existing logic
        processor = AutoDocumentProcessor(api_key=OPENAI_API_KEY)
        
        # Process documents using your existing logic
        stories = await processor.process_documents(documents)
        
        return {
            "status": "success",
            "stories": stories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 