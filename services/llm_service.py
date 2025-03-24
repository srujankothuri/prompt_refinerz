import os
import time
import httpx
import asyncio
from typing import List, Dict, Any
from anthropic import AsyncAnthropic
import google.generativeai as genai
from dotenv import load_dotenv

# Assuming these are defined elsewhere in your project
from models import LLMProvider, PromptVariation

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = "AIzaSyCNEWH98mW_gTa7N1mfz8uIZezuQdpvgkg"  # Hardcoded for now; consider moving to .env
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Configure Google API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Anthropic async client
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

class LLMService:
    @staticmethod
    def _get_system_prompt(desired_output_type: str, context: str = None) -> str:
        """Generate a tailored system prompt based on the desired output type."""
        base_prompt = """
        You are an expert prompt engineer. Your task is to improve the given prompt to create a more effective version.
        
        The improved prompt should:
        1. Be clear and specific
        2. Provide necessary context
        3. Structure information logically
        4. Use appropriate formatting
        5. Set the right constraints
        6. Include useful examples if needed
        """
        
        output_specific_instructions = {
            "General": "Optimize the prompt for general-purpose use, ensuring versatility and clarity.",
            "Summarization": """
            Optimize the prompt for summarizing content. Ensure it instructs to capture key points concisely,
            avoid unnecessary details, and maintain the original meaning.
            Example: 'Summarize the following article in 3 sentences, focusing on main ideas.'
            """,
            "Code Generation": """
            Optimize the prompt for generating code. Ensure it specifies the programming language, functionality,
            and any constraints (e.g., performance, style). Include an example if relevant.
            Example: 'Write a Python function to sort a list of integers in ascending order using bubble sort.'
            """,
            "Data Analysis": """
            Optimize the prompt for data analysis tasks. Ensure it defines the data source, analysis goal,
            and output format (e.g., table, insights). Include an example if helpful.
            Example: 'Analyze a CSV file with sales data and return the top 5 products by revenue in a table.'
            """,
            "Technical Explanation": """
            Optimize the prompt for explaining technical concepts. Ensure it requests clear, step-by-step explanations
            suitable for the target audience (e.g., beginner, expert). Include an example if applicable.
            Example: 'Explain how a binary search tree works to a beginner, with a simple example.'
            """,
            "Creative Writing": """
            Optimize the prompt for creative writing. Ensure it sets the tone, genre, and any specific elements
            (e.g., characters, setting). Include an example if useful.
            Example: 'Write a short fantasy story about a dragon rider in a war-torn kingdom.'
            """
        }
        
        specific_instruction = output_specific_instructions.get(desired_output_type, output_specific_instructions["General"])
        
        return f"""
        {base_prompt}
        
        The prompt will be used for: {desired_output_type}
        
        {specific_instruction}
        
        {f'Additional context: {context}' if context else ''}
        
        Generate an improved version of the prompt. Provide ONLY the refined prompt text with no explanations.
        """

    @staticmethod
    async def generate_prompt_variations(
        original_prompt: str,
        context: str = None,
        desired_output_type: str = "General",
        providers: List[LLMProvider] = [LLMProvider.ANTHROPIC, LLMProvider.TOGETHER, LLMProvider.GEMINI],
        num_variations: int = 5
    ) -> List[PromptVariation]:
        """Generate exactly num_variations prompt variations using multiple LLM providers."""
        
        system_prompt = LLMService._get_system_prompt(desired_output_type, context)
        
        tasks = []
        base_variations = num_variations // len(providers)  # Base number per provider
        extra_variations = num_variations % len(providers)   # Remainder to distribute
        
        print(f"Providers: {len(providers)}, Requested variations: {num_variations}, Base: {base_variations}, Extra: {extra_variations}")
        
        for i, provider in enumerate(providers):
            provider_variations = base_variations + (1 if i < extra_variations else 0)
            for _ in range(provider_variations):
                tasks.append(LLMService._generate_single_variation(provider, original_prompt, system_prompt))
        
        print(f"Generating {len(tasks)} variations across {len(providers)} providers")
        
        variations = await asyncio.gather(*tasks, return_exceptions=True)
        valid_variations = [v for v in variations if isinstance(v, PromptVariation)]
        
        print(f"Generated {len(valid_variations)} valid variations")
        
        while len(valid_variations) < num_variations:
            valid_variations.append(PromptVariation(
                prompt_text="Error: Insufficient valid variations generated",
                provider=providers[len(valid_variations) % len(providers)],
                quality_score=0.0
            ))
        
        return valid_variations[:num_variations]
    
    @staticmethod
    async def _generate_single_variation(
        provider: LLMProvider, 
        original_prompt: str,
        system_prompt: str
    ) -> PromptVariation:
        """Generate a single prompt variation using the specified provider."""
        
        start_time = time.time()
        prompt_text = ""
        
        try:
            if provider == LLMProvider.ANTHROPIC:
                if not ANTHROPIC_API_KEY:
                    raise ValueError("Anthropic API key is missing")
                response = await anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"Original prompt: {original_prompt}"}
                    ]
                )
                prompt_text = response.content[0].text.strip()
                
            elif provider == LLMProvider.TOGETHER:
                if not TOGETHER_API_KEY:
                    raise ValueError("Together API key is missing")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {TOGETHER_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Original prompt: {original_prompt}"}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 1024
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()
                    result = response.json()
                    prompt_text = result["choices"][0]["message"]["content"].strip()
                    
            elif provider == LLMProvider.GEMINI:
                if not GEMINI_API_KEY:
                    raise ValueError("Gemini API key is missing")
                model = genai.GenerativeModel('gemini-1.5-flash')
                print(f"Using Gemini model: gemini-1.5-flash with API Key: {GEMINI_API_KEY[:5]}...")
                response = await asyncio.to_thread(
                    model.generate_content,
                    f"{system_prompt}\n\nOriginal prompt: {original_prompt}"
                )
                prompt_text = response.text.strip()
                
        except Exception as e:
            prompt_text = f"Error generating variation with {provider}: {str(e)}"
        
        return PromptVariation(
            prompt_text=prompt_text,
            provider=provider,
            quality_score=0.0
        )

# Synchronous test function for Gemini
def test_gemini_sync():
    """Test Gemini synchronously to isolate the issue."""
    print(f"Testing Gemini synchronously with model: gemini-1.5-flash, API Key: {GEMINI_API_KEY[:5]}...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content("Test prompt: Write a short story")
        print("Synchronous Gemini Test:", response.text)
    except Exception as e:
        print(f"Synchronous Gemini Test Failed: {str(e)}")

# Example usage testing all output types
async def main():
    print("Running synchronous Gemini test...")
    test_gemini_sync()
    
    # List of all supported output types
    output_types = [
        "General",
        "Summarization",
        "Code Generation",
        "Data Analysis",
        "Technical Explanation",
        "Creative Writing"
    ]
    
    original_prompts = {
        "General": "Tell me about dogs",
        "Summarization": "Summarize this article",
        "Code Generation": "Write a sorting function",
        "Data Analysis": "Analyze sales data",
        "Technical Explanation": "Explain binary search",
        "Creative Writing": "Write a story"
    }
    
    providers = [LLMProvider.ANTHROPIC, LLMProvider.TOGETHER, LLMProvider.GEMINI]
    num_variations = 5
    
    for output_type in output_types:
        print(f"\n=== Testing Output Type: {output_type} ===")
        original_prompt = original_prompts[output_type]
        variations = await LLMService.generate_prompt_variations(
            original_prompt=original_prompt,
            context="A fantasy setting with dragons" if output_type == "Creative Writing" else None,
            desired_output_type=output_type,
            providers=providers,
            num_variations=num_variations
        )
        for i, variation in enumerate(variations, 1):
            print(f"Variation {i} - Provider: {variation.provider}, Prompt: {variation.prompt_text}")

if __name__ == "__main__":
    asyncio.run(main())