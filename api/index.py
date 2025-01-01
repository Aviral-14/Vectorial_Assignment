from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from typing import List
from scripts.auto_processor import AutoDocumentProcessor

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
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"Error loading template: {str(e)}", status_code=500)

@app.get("/health")
async def health_check():
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "templates_dir": os.path.exists("templates"),
        "index_html": os.path.exists("templates/index.html")
    }

@app.post("/process-documents/")
async def process_documents(files: List[UploadFile]):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        documents = {}
        for file in files:
            if file.filename.endswith('.txt'):
                content = await file.read()
                documents[file.filename] = content.decode('utf-8')
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid text files uploaded")
        
        processor = AutoDocumentProcessor(api_key=api_key)
        stories = await processor.process_documents(documents)
        
        return {
            "status": "success",
            "stories": stories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 