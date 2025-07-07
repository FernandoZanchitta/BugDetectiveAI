"""
Prompt builder using Jinja2 templates for dynamic prompt generation.
"""

from typing import Dict, Any, List, Optional
from jinja2 import Template, Environment, BaseLoader
import os


class PromptBuilder:
    """A class for building prompts using Jinja2 templates."""

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the PromptBuilder.

        Args:
            template_dir (str, optional): Directory containing template files.
                If None, uses string templates only.
        """
        self.template_dir = template_dir
        if template_dir:
            self.env = Environment(loader=BaseLoader())
        else:
            self.env = Environment()

    def build_correction_prompt(
        self,
        buggy_code: str,
        traceback_error: str,
        retrieved_examples: Optional[List[Dict[str, str]]] = None,
        instruction_prompt: str = "Please correct the following buggy code based on the error message.",
        include_examples: bool = True,
    ) -> str:
        """
        Build a prompt for code correction using Jinja2 template.

        Args:
            buggy_code (str): The original buggy code.
            traceback_error (str): The traceback error associated with the code.
            retrieved_examples (List[Dict[str, str]], optional): List of examples in the form
                [{"buggy_code": str, "corrected_code": str}].
            instruction_prompt (str): The instruction prompt for the LLM.
            include_examples (bool): Whether to include retrieved examples in the prompt.

        Returns:
            str: The formatted prompt.
        """
        template_str = """
{{ instruction_prompt }}

### BUGGY CODE:
{{ buggy_code }}

### ERROR:
{{ traceback_error }}

{% if retrieved_examples and include_examples %}
### RETRIEVED EXAMPLES:
{% for example in retrieved_examples %}
#### EXAMPLE:
Buggy:
{{ example.buggy_code }}
Corrected:
{{ example.corrected_code }}

{% endfor %}
{% endif %}

### RETURN ONLY THE CORRECTED CODE BELOW:
"""

        template = Template(template_str)
        return template.render(
            instruction_prompt=instruction_prompt,
            buggy_code=buggy_code,
            traceback_error=traceback_error,
            retrieved_examples=retrieved_examples or [],
            include_examples=include_examples,
        )

    def build_structured_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Build a prompt using a named template file.

        Args:
            template_name (str): Name of the template file (without extension).
            **kwargs: Variables to pass to the template.

        Returns:
            str: The formatted prompt.
        """
        if not self.template_dir:
            raise ValueError(
                "Template directory not set. Use string templates instead."
            )

        template_path = os.path.join(self.template_dir, f"{template_name}.jinja")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r") as f:
            template_content = f.read()

        template = Template(template_content)
        return template.render(**kwargs)

    def build_custom_prompt(self, template_string: str, **kwargs: Any) -> str:
        """
        Build a prompt using a custom template string.

        Args:
            template_string (str): The Jinja2 template string.
            **kwargs: Variables to pass to the template.

        Returns:
            str: The formatted prompt.
        """
        template = Template(template_string)
        return template.render(**kwargs)


# Convenience function for backward compatibility
def build_correction_prompt(
    buggy_code: str,
    traceback_error: str,
    retrieved_examples: Optional[List[Dict[str, str]]] = None,
    instruction_prompt: str = "Please correct the following buggy code based on the error message.",
) -> str:
    """
    Convenience function to build a correction prompt.

    Args:
        buggy_code (str): The original buggy code.
        traceback_error (str): The traceback error associated with the code.
        retrieved_examples (List[Dict[str, str]], optional): List of examples.
        instruction_prompt (str): The instruction prompt for the LLM.

    Returns:
        str: The formatted prompt.
    """
    builder = PromptBuilder()
    return builder.build_correction_prompt(
        buggy_code=buggy_code,
        traceback_error=traceback_error,
        retrieved_examples=retrieved_examples,
        instruction_prompt=instruction_prompt,
    )
