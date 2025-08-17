"""
OpenRouter LLM model implementation for BugDetectiveAI.
"""

import json
import os
from typing import Dict, Any, Optional, List
from tqdm import tqdm


class OpenRouterLLMModel:
    """OpenRouter LLM model implementation for code generation."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPEN_ROUTER_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.client = None

    def validate_config(self) -> bool:
        """Validate OpenRouter configuration."""
        return bool(self.api_key and self.model_name)

    async def _initialize_client(self):
        """Initialize OpenAI client with OpenRouter base URL."""
        if not self.client:
            try:
                import openai

                self.client = openai.AsyncOpenAI(
                    api_key=self.api_key, base_url="https://openrouter.ai/api/v1"
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: poetry add openai"
                )

    async def generate_code_output(
        self, prompt: str, show_progress: bool = True
    ) -> str:
        """Generate code output only - no explanations or other text."""
        try:
            await self._initialize_client()

            if self.client is None:
                raise RuntimeError("OpenAI client not initialized")

            # Enhanced prompt to ensure code-only output
            code_prompt = f"""
{prompt}

IMPORTANT: Return ONLY the corrected/requested code. Do not include any explanations, comments about the changes, or other text. Just return the pure code.
"""

            # Build messages array with optional system prompt
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": code_prompt})

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("No content returned from OpenRouter API")

            # Clean the response to ensure it's just code
            cleaned_content = self._extract_code_only(content)

            return cleaned_content

        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

    def _extract_code_only(self, content: str) -> str:
        """Extract only code from the response, removing explanations and markdown."""
        if not content:
            return ""

        # Remove markdown code blocks if present
        lines = content.strip().split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            # Skip markdown code block markers
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            # If we're in a code block or the line looks like code, include it
            if in_code_block or self._looks_like_code(line):
                code_lines.append(line)

        # If no code blocks found, try to extract code from the entire response
        if not code_lines:
            return content.strip()

        return "\n".join(code_lines)

    def _looks_like_code(self, line: str) -> bool:
        """Check if a line looks like code rather than explanation text."""
        line = line.strip()
        
        # Skip empty lines
        if not line:
            return False
            
        # Skip common explanation patterns
        if any(pattern in line.lower() for pattern in [
            "here's the corrected code",
            "the corrected version",
            "here's the fix",
            "the issue was",
            "to fix this",
            "explanation:",
            "note:",
            "comment:",
            "the problem is",
            "this error occurs"
        ]):
            return False
            
        # Consider it code if it contains code-like elements
        code_indicators = [
            "def ", "class ", "import ", "from ", "if ", "for ", "while ",
            "try:", "except:", "finally:", "with ", "return ", "yield ",
            "=", "(", ")", "[", "]", "{", "}", ":", ";", "\\"
        ]
        
        return any(indicator in line for indicator in code_indicators)

    async def generate_batch_outputs(
        self, prompts: List[str], show_progress: bool = True
    ) -> List[str]:
        """Generate outputs for multiple prompts with progress tracking."""
        results = []
        
        if show_progress:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating batch outputs",
                unit="prompts",
                ncols=100,
            )

        for i, prompt in enumerate(prompts):
            try:
                if show_progress:
                    pbar.set_description(f"Processing prompt {i + 1}/{len(prompts)}")

                # Generate output for this prompt
                output = await self.generate_code_output(prompt)
                results.append(output)

            except Exception as e:
                error_msg = f"Error processing prompt {i}: {str(e)}"
                results.append(error_msg)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        return results


def create_openrouter_model(
    model_name: str = "anthropic/claude-3.5-sonnet",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> OpenRouterLLMModel:
    """Create an OpenRouter model instance for code generation.

    Args:
        model_name: The model to use (e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4')
        api_key: OpenRouter API key (will use env var if not provided)
        temperature: Sampling temperature (default 0.0 for consistent code generation)
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt to guide the model behavior

    Returns:
        OpenRouterLLMModel instance optimized for code generation
    """
    return OpenRouterLLMModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


def create_apr_model(
    model_name: str = "anthropic/claude-3.5-sonnet",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> OpenRouterLLMModel:
    """Create an OpenRouter model instance specifically for Automatic Program Repair (APR).

    Args:
        model_name: The model to use (e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4')
        api_key: OpenRouter API key (will use env var if not provided)
        temperature: Sampling temperature (default 0.0 for consistent code generation)
        max_tokens: Maximum tokens to generate

    Returns:
        OpenRouterLLMModel instance configured for APR with appropriate system prompt
    """
    apr_system_prompt = """You are an expert Python programmer specializing in automatic program repair. Your task is to fix Python code based on error tracebacks.

Key principles:
1. Fix the specific error mentioned in the traceback
2. Maintain the original code structure and style
3. Return ONLY the corrected code, no explanations
4. Preserve function signatures and variable names when possible
5. Ensure the fixed code follows Python best practices

Focus on the exact error and provide a minimal, correct fix."""
    
    return OpenRouterLLMModel(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=apr_system_prompt,
    )
