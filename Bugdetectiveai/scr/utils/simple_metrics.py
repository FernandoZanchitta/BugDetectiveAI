import ast
import difflib

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

    # Fallback: raw text similarity
    text_score = difflib.SequenceMatcher(None, before_code, after_code).ratio()

    return {"ast_score": ast_score, "text_score": text_score}


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

