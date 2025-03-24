import time
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
import os
import sys
import uvicorn

# Add the current directory to sys.path so absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Absolute imports
from models import PromptRequest, PromptResponse
from services.llm_service import LLMService
from services.scorer import ScoringService

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Prompt Refiner API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/refine", response_model=PromptResponse)
async def refine_prompt(request: PromptRequest):
    """Process a prompt and generate improved variations."""
    
    start_time = time.time()
    
    # Generate prompt variations using selected providers
    variations = await LLMService.generate_prompt_variations(
        original_prompt=request.original_prompt,
        context=request.context,
        desired_output_type=request.desired_output_type,
        providers=request.providers,
        num_variations=request.num_variations
    )
    
    # Score prompt variations
    scored_variations = await ScoringService.score_prompt_variations(
        original_prompt=request.original_prompt,
        variations=variations
    )
    
    # Find best prompt
    best_prompt = ScoringService.find_best_prompt(scored_variations)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Create and return response
    response = PromptResponse(
        original_prompt=request.original_prompt,
        variations=scored_variations,
        best_prompt=best_prompt,
        processing_time=processing_time
    )
    
    return response

@app.get("/api/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

# Custom Uvicorn runner to show localhost
if __name__ == "__main__":
    # Define server config
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",  # Bind to all interfaces
        port=8000,
        reload=True,
        log_level="info"
    )
    
    # Create server instance
    server = uvicorn.Server(config)
    
    # Print custom message before starting
    print("Starting Prompt Refiner API...")
    print("Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)")
    
    # Run the server
    server.run()