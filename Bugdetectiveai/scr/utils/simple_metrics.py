import ast
import difflib
import re
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

def display_asts(before_code: str, after_code: str) -> None:
    """Display both ASTs side by side for visual comparison.
    
    Args:
        before_code: The original code string
        after_code: The modified code string
    """
    
    # Parse and display before AST
    print("\n📄 BEFORE CODE AST:")
    print("-" * 40)
    try:
        before_ast = ast.parse(before_code)
        print(ast.dump(before_ast, indent=2))
    except SyntaxError as e:
        print(f"❌ Syntax Error in before_code: {e}")
    
    # Parse and display after AST
    print("\n📄 AFTER CODE AST:")
    print("-" * 40)
    try:
        after_ast = ast.parse(after_code)
        print(ast.dump(after_ast, indent=2))
    except SyntaxError as e:
        print(f"❌ Syntax Error in after_code: {e}")
    
    # Show the diff score for reference
    scores = diff_score(before_code, after_code)
    print(f"\n📊 AST Similarity Score: {scores['ast_score']:.3f}")
    print(f"📊 Text Similarity Score: {scores['text_score']:.3f}")
    print("=" * 80)

class Normalizer(ast.NodeTransformer):
    """Normalizes identifiers and constants in an AST."""
    def __init__(self):
        self.var_count = 0
        self.func_count = 0
        self.class_count = 0
        self.name_mapping = {}

    def generic_visit(self, node):
        """Default visit method."""
        return super().generic_visit(node)

    def visit_FunctionDef(self, node):
        """Rename function definitions."""
        if node.name not in self.name_mapping:
            self.func_count += 1
            self.name_mapping[node.name] = f"_func_{self.func_count}"
        node.name = self.name_mapping[node.name]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        """Rename class definitions."""
        if node.name not in self.name_mapping:
            self.class_count += 1
            self.name_mapping[node.name] = f"_class_{self.class_count}"
        node.name = self.name_mapping[node.name]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        """Rename variable identifiers."""
        if node.id not in self.name_mapping:
            self.var_count += 1
            self.name_mapping[node.id] = f"_var_{self.var_count}"
        node.id = self.name_mapping[node.id]
        return node

    def visit_Constant(self, node):
        """Replace literal constants with placeholder."""
        return ast.copy_location(ast.Constant(value="_const_"), node)

def get_normalized_ast(code: str) -> str:
    """
    Returns a normalized AST dump for the given code,
    making it robust to variable names and literal changes.
    """
    try:
        parsed = ast.parse(code)
        normalizer = Normalizer()
        normalized_tree = normalizer.visit(parsed)
        ast.fix_missing_locations(normalized_tree)
        return ast.dump(normalized_tree, indent=4)
    except SyntaxError:
        return ""
def diff_score(before_code: str, after_code: str) -> dict:
    """Calculate a diff_score between before_code and after_code.

    Returns:
        A float in [0.0, 1.0], where 1.0 means identical.
    """
    # Try comparing AST structure
    try:
        before_ast = ast.dump(ast.parse(before_code))
        after_ast = ast.dump(ast.parse(after_code))
        ast_score = difflib.SequenceMatcher(None, before_ast, after_ast).ratio()
    except SyntaxError:
        ast_score = 0.0  # If parse fails, AST match isn't valid.

    # Normalize the AST
    before_ast = get_normalized_ast(before_code)
    after_ast = get_normalized_ast(after_code)
    ast_score_normalized = difflib.SequenceMatcher(None, before_ast, after_ast).ratio()

    # Fallback: raw text similarity
    text_score = difflib.SequenceMatcher(None, before_code, after_code).ratio()
    
    return {"ast_score": ast_score, "text_score": text_score, "ast_score_normalized": ast_score_normalized}


PY_KEYWORDS = {
    "and","as","assert","break","class","continue","def","del","elif","else",
    "except","False","finally","for","from","global","if","import","in","is",
    "lambda","None","nonlocal","not","or","pass","raise","return","True",
    "try","while","with","yield"
}

def get_ngrams(text, n=4):
    """Get character-level n-grams for text."""
    return [text[i:i + n] for i in range(len(text) - n + 1)]

def codebleu(candidate, reference, alpha=0.5, beta=0.5):
    """Very simplified CodeBLEU focusing on text overlap and keyword match."""
    # 1. N-gram overlap (similar to BLEU)
    candidate_ngrams = Counter(get_ngrams(candidate))
    reference_ngrams = Counter(get_ngrams(reference))
    common = sum((candidate_ngrams & reference_ngrams).values())
    total = max(sum(candidate_ngrams.values()), 1)
    ngram_score = common / total

    # 2. Keyword overlap
    candidate_keywords = set(re.findall(r'\b\w+\b', candidate)) & PY_KEYWORDS
    reference_keywords = set(re.findall(r'\b\w+\b', reference)) & PY_KEYWORDS
    keyword_score = len(candidate_keywords & reference_keywords) / max(len(reference_keywords), 1)

    # Final weighted score
    return alpha * ngram_score + beta * keyword_score