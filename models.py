from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"  
    TOGETHER = "together"
    GEMINI = "gemini"
    

class PromptRequest(BaseModel):
    original_prompt: str
    desired_output_type: str = "General"
    context: Optional[str] = None
    providers: List[LLMProvider] = [LLMProvider.ANTHROPIC, LLMProvider.TOGETHER, LLMProvider.GEMINI]
    num_variations: int = Field(default=5, ge=3, le=10)
    
class PromptVariation(BaseModel):
    prompt_text: str
    provider: LLMProvider
    quality_score: float = 0.0
    strengths: List[str] = []
    weaknesses: List[str] = []
    
class PromptResponse(BaseModel):
    original_prompt: str
    variations: List[PromptVariation]
    best_prompt: PromptVariation
    processing_time: float