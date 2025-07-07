import ast
import difflib
import logging
from typing import Dict, Tuple
from codebleu import calc_codebleu

# Constants
DEFAULT_CODEBLEU_WEIGHTS = (0.1, 0.4, 0.1, 0.4)
LOG_FILENAME = "codebleu_dataflow_warnings.log"
PY_KEYWORDS = {
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
}

# Configure logging for problematic cases
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def exact_match(a: str, b: str) -> float:
    """Check if two strings are exactly equal.

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        1.0 if strings are identical, 0.0 otherwise
    """
    return 1.0 if a == b else 0.0


def codebleu(
    candidate: str,
    reference: str,
    weights: Tuple[float, float, float, float] = DEFAULT_CODEBLEU_WEIGHTS,
) -> Dict[str, float]:
    """Calculate CodeBLEU metrics between candidate and reference code.

    Args:
        candidate: The generated/fixed code to evaluate
        reference: The original/correct code to compare against
        weights: Tuple of weights for (ngram_match, weighted_ngram_match, syntax_match, dataflow_match)

    Returns:
        Dictionary containing CodeBLEU metrics

    Raises:
        AssertionError: If weights are invalid
    """
    if len(weights) != 4 or sum(weights) != 1.0 or not all(w >= 0 for w in weights):
        raise ValueError(
            "Weights must be a tuple of 4 non-negative floats that sum to 1.0"
        )

    metric = calc_codebleu([reference], [candidate], lang="python", weights=weights)

    # Log dataflow extraction failures for debugging
    if metric.get("dataflow_match", 1.0) == 0.0:
        logging.info("Dataflow extraction failed:")
        logging.info(f"REFERENCE:\n{reference}")
        logging.info(f"CANDIDATE:\n{candidate}")
        logging.info("-" * 60)

    return metric


class Normalizer(ast.NodeTransformer):
    """Normalizes identifiers and constants in an AST to make comparisons robust to naming changes."""

    def __init__(self):
        self.var_count = 0
        self.func_count = 0
        self.class_count = 0
        self.name_mapping = {}

    def _get_or_create_mapping(self, name: str, prefix: str, counter: int) -> str:
        """Helper method to get or create normalized name mapping."""
        if name not in self.name_mapping:
            counter += 1
            self.name_mapping[name] = f"{prefix}_{counter}"
        return self.name_mapping[name]

    def visit_FunctionDef(self, node):
        """Rename function definitions to normalized names."""
        node.name = self._get_or_create_mapping(node.name, "_func", self.func_count)
        self.func_count += 1
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        """Rename class definitions to normalized names."""
        node.name = self._get_or_create_mapping(node.name, "_class", self.class_count)
        self.class_count += 1
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        """Rename variable identifiers to normalized names."""
        node.id = self._get_or_create_mapping(node.id, "_var", self.var_count)
        self.var_count += 1
        return node

    def visit_Constant(self, node):
        """Replace literal constants with placeholder."""
        return ast.copy_location(ast.Constant(value="_const_"), node)


def get_normalized_ast(code: str) -> str:
    """Generate a normalized AST dump for the given code.

    Args:
        code: Python code string to normalize

    Returns:
        Normalized AST dump as string, or empty string if parsing fails
    """
    try:
        parsed = ast.parse(code)
        normalizer = Normalizer()
        normalized_tree = normalizer.visit(parsed)
        ast.fix_missing_locations(normalized_tree)
        return ast.dump(normalized_tree, indent=4)
    except SyntaxError:
        return ""


def _calculate_ast_similarity(before_code: str, after_code: str) -> float:
    """Calculate AST similarity between two code strings.

    Args:
        before_code: Original code
        after_code: Modified code

    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        before_ast = ast.dump(ast.parse(before_code))
        after_ast = ast.dump(ast.parse(after_code))
        return difflib.SequenceMatcher(None, before_ast, after_ast).ratio()
    except SyntaxError:
        return 0.0


def diff_score(before_code: str, after_code: str) -> Dict[str, float]:
    """Calculate comprehensive similarity scores between before and after code.

    This is the main function that calculates multiple similarity metrics:
    - AST structure similarity
    - Normalized AST similarity (ignoring variable/function names)
    - Text similarity
    - CodeBLEU metrics

    Args:
        before_code: Original code string
        after_code: Modified code string

    Returns:
        Dictionary containing all similarity metrics
    """


    # Calculate exact match
    exact_match_score = exact_match(before_code, after_code)

    # Calculate AST similarity
    ast_score = _calculate_ast_similarity(before_code, after_code)

    # Calculate normalized AST similarity
    before_normalized = get_normalized_ast(before_code)
    after_normalized = get_normalized_ast(after_code)
    ast_score_normalized = difflib.SequenceMatcher(
        None, before_normalized, after_normalized
    ).ratio()

    # Calculate text similarity
    text_score = difflib.SequenceMatcher(None, before_code, after_code).ratio()

    # Calculate CodeBLEU metrics
    codebleu_metrics = codebleu(after_code, before_code)

    return {
        "exact_match": exact_match_score,
        "ast_score": ast_score,
        "text_score": text_score,
        "ast_score_normalized": ast_score_normalized,
        **codebleu_metrics,
    }
