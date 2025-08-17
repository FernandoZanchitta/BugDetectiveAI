"""
OpenRouter LLM model implementation for BugDetectiveAI.
"""

import json
import os
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from .base_model import BaseLLMModel, ModelConfig, StructuredOutput


class OpenRouterLLMModel(BaseLLMModel):
    """OpenRouter LLM model implementation for code generation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OPEN_ROUTER_KEY")
        self.client = None

    def validate_config(self) -> bool:
        """Validate OpenRouter configuration."""
        return bool(self.api_key and self.config.model_name)

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
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": code_prompt})
            # print(messages)
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
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
            return self._clean_code_response(content)

        return "\n".join(code_lines).strip()

    def _looks_like_code(self, line: str) -> bool:
        """Check if a line looks like code rather than explanation."""
        line = line.strip()

        # Skip empty lines
        if not line:
            return False

        # Skip common explanation patterns
        explanation_patterns = [
            "here is",
            "here's",
            "the corrected",
            "the fixed",
            "the solution",
            "this will",
            "this code",
            "the code",
            "here you",
            "here is the",
            "the issue",
            "the problem",
            "the error",
            "to fix",
            "to correct",
        ]

        line_lower = line.lower()
        for pattern in explanation_patterns:
            if pattern in line_lower:
                return False

        # Consider it code if it contains programming elements
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "if ",
            "for ",
            "while ",
            "return ",
            "print(",
            "def(",
            "class(",
            "import(",
            "from(",
            "=",
            "+",
            "-",
            "*",
            "/",
            "==",
            "!=",
            "<=",
            ">=",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ":",
            ";",
        ]

        for indicator in code_indicators:
            if indicator in line:
                return True

        return False

    def _clean_code_response(self, content: str) -> str:
        """Clean a response that might contain explanations mixed with code."""
        lines = content.strip().split("\n")
        code_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are clearly explanations
            if line.lower().startswith(("here", "the ", "this ", "to ", "you ")):
                continue

            # Include lines that look like code
            if self._looks_like_code(line):
                code_lines.append(line)

        return "\n".join(code_lines).strip()

    # Required abstract methods (simplified for your use case)
    async def generate_structured_output(
        self, prompt: str, output_schema: Dict[str, Any]
    ) -> StructuredOutput:
        """Generate structured output - simplified for code-only focus."""
        try:
            code = await self.generate_code_output(prompt)
            return StructuredOutput(success=True, content={"code": code})
        except Exception as e:
            return StructuredOutput(success=False, content={}, error_message=str(e))

    async def generate_basic_output(self, prompt: str) -> str:
        """Generate basic text output - simplified to use code generation."""
        return await self.generate_code_output(prompt)

    async def generate_batch_outputs(
        self, prompts: List[str], show_progress: bool = True
    ) -> List[str]:
        """
        Generate outputs for a batch of prompts with progress monitoring.

        Args:
            prompts: List of prompts to process
            show_progress: Whether to show progress bar

        Returns:
            List of generated outputs
        """
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
    config = ModelConfig(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    return OpenRouterLLMModel(config)


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
    # Read APR prompt directly from file
    prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.txt")
    apr_prompt = ""
    
    if os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract APR prompt (between [apr] and next [type] or end)
            start = content.find('[apr]')
            if start != -1:
                start = content.find('\n', start) + 1
                end = content.find('[', start)
                if end == -1:
                    end = len(content)
                apr_prompt = content[start:end].strip()
    
    config = ModelConfig(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=apr_prompt,
    )

    return OpenRouterLLMModel(config)
