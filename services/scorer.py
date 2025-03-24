import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from anthropic import AsyncAnthropic  # Use Anthropic SDK
import os

# Absolute imports
from models import PromptVariation

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ScoringService:
    @staticmethod
    async def score_prompt_variations(
        original_prompt: str,
        variations: List[PromptVariation]
    ) -> List[PromptVariation]:
        """Score prompt variations based on quality criteria."""
        
        tasks = []
        for variation in variations:
            tasks.append(ScoringService._score_single_variation(original_prompt, variation))
        
        # Process all scoring tasks concurrently
        scored_variations = await asyncio.gather(*tasks)
        return scored_variations
    
    @staticmethod
    async def _score_single_variation(
        original_prompt: str,
        variation: PromptVariation
    ) -> PromptVariation:
        """Score a single prompt variation."""
        
        # Define the scoring system prompt
        system_prompt = """
        You are an expert prompt engineer evaluator. Analyze the given prompt based on the following criteria:
        
        1. Clarity (0-10): Is the prompt clear and unambiguous?
        2. Specificity (0-10): Does it provide specific instructions?
        3. Context (0-10): Does it include necessary background information?
        4. Structure (0-10): Is information organized logically?
        5. Constraints (0-10): Does it set appropriate boundaries?
        6. Examples (0-10): Does it include useful examples if needed?
        7. Improvement (0-10): How much better is it than the original prompt?
        
        Provide your evaluation as a JSON object with:
        - score: The overall score (calculated average from all criteria, multiplied by 10)
        - strengths: Array of 2-3 key strengths
        - weaknesses: Array of 2-3 areas for improvement
        
        Format your response as valid JSON only, with no additional text.
        """
        
        try:
            # Use Anthropic's Claude for scoring
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Adjust model as needed
                max_tokens=1024,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Original prompt: {original_prompt}\n\nImproved prompt: {variation.prompt_text}"}
                ]
            )
            
            # Extract JSON result from the response
            result = response.content[0].text.strip()
            import json
            evaluation = json.loads(result)  # Safer than eval()
            
            # Update the variation with the score and feedback
            variation.quality_score = float(evaluation.get("score", 0))
            variation.strengths = evaluation.get("strengths", [])
            variation.weaknesses = evaluation.get("weaknesses", [])
            
        except Exception as e:
            # Default scoring if evaluation fails
            variation.quality_score = 50.0
            variation.strengths = ["Generated successfully"]
            variation.weaknesses = [f"Evaluation failed: {str(e)}"]
        
        return variation
    
    @staticmethod
    def find_best_prompt(variations: List[PromptVariation]) -> PromptVariation:
        """Find the best prompt variation based on quality score."""
        if not variations:
            return None
        
        # Sort by quality score (descending) and return the best one
        return sorted(variations, key=lambda x: x.quality_score, reverse=True)[0]