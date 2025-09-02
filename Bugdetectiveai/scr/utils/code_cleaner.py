#!/usr/bin/env python3
"""
Módulo para limpeza e correção automática de código Python.
Inclui correção de indentação, formatação e validação de sintaxe.
"""

import ast
import re
import textwrap
from typing import Tuple, Optional, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_indentation_issues(code: str) -> Dict[str, Any]:
    """
    Detecta problemas de indentação no código Python.
    
    Args:
        code: String contendo código Python
        
    Returns:
        Dicionário com informações sobre problemas detectados
    """
    issues = {
        'has_issues': False,
        'error_type': None,
        'error_message': None,
        'line_number': None,
        'suggested_fix': None
    }
    
    try:
        # Tentar fazer parse do código
        ast.parse(code)
        return issues  # Sem problemas
    except IndentationError as e:
        issues['has_issues'] = True
        issues['error_type'] = 'IndentationError'
        issues['error_message'] = str(e)
        
        # Extrair número da linha do erro
        line_match = re.search(r'line (\d+)', str(e))
        if line_match:
            issues['line_number'] = int(line_match.group(1))
        
        # Detectar tipo específico de problema de indentação
        if 'unexpected indent' in str(e):
            issues['suggested_fix'] = 'remove_extra_indentation'
        elif 'expected an indented block' in str(e):
            issues['suggested_fix'] = 'add_missing_indentation'
        else:
            issues['suggested_fix'] = 'fix_indentation'
            
    except SyntaxError as e:
        # Outros erros de sintaxe (não indentação)
        issues['has_issues'] = True
        issues['error_type'] = 'SyntaxError'
        issues['error_message'] = str(e)
        
        line_match = re.search(r'line (\d+)', str(e))
        if line_match:
            issues['line_number'] = int(line_match.group(1))
    
    return issues


def fix_extra_indentation(code: str) -> str:
    """
    Remove indentação extra no início do código.
    
    Args:
        code: Código com indentação extra
        
    Returns:
        Código com indentação corrigida
    """
    lines = code.split('\n')
    
    # Encontrar a menor indentação não-zero
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Linha não vazia
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                min_indent = min(min_indent, indent)
    
    # Se não há indentação ou se é muito pequena, não fazer nada
    if min_indent == float('inf') or min_indent <= 2:
        return code
    
    # Remover a indentação extra
    fixed_lines = []
    for line in lines:
        if line.strip():  # Linha não vazia
            # Remover apenas a indentação extra, mantendo a estrutura
            if len(line) - len(line.lstrip()) >= min_indent:
                fixed_lines.append(line[min_indent:])
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_missing_indentation(code: str) -> str:
    """
    Adiciona indentação faltante onde necessário.
    
    Args:
        code: Código com indentação faltante
        
    Returns:
        Código com indentação corrigida
    """
    lines = code.split('\n')
    fixed_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if not stripped:
            fixed_lines.append(line)
            continue
        
        # Verificar se a linha deve ter indentação
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'else:', 'elif ')):
            # Nova estrutura - resetar indentação
            if stripped.endswith(':'):
                indent_level += 1
            fixed_lines.append('    ' * indent_level + stripped)
        elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise', 'yield')):
            # Comandos que podem estar no nível atual
            fixed_lines.append('    ' * indent_level + stripped)
        else:
            # Linha normal - manter indentação atual
            fixed_lines.append('    ' * indent_level + stripped)
        
        # Reduzir indentação se necessário
        if stripped.endswith(('return', 'break', 'continue', 'pass', 'raise', 'yield')):
            # Comando de saída - pode reduzir indentação
            pass
    
    return '\n'.join(fixed_lines)


def auto_fix_indentation(code: str) -> str:
    """
    Tenta corrigir automaticamente problemas de indentação.
    
    Args:
        code: Código com problemas de indentação
        
    Returns:
        Código corrigido ou original se não conseguir corrigir
    """
    issues = detect_indentation_issues(code)
    
    if not issues['has_issues']:
        return code  # Sem problemas
    
    if issues['error_type'] == 'IndentationError':
        if issues['suggested_fix'] == 'remove_extra_indentation':
            try:
                fixed_code = fix_extra_indentation(code)
                # Verificar se a correção funcionou
                ast.parse(fixed_code)
                logger.info(f"Corrigido problema de indentação extra na linha {issues['line_number']}")
                return fixed_code
            except Exception as e:
                logger.warning(f"Falha ao corrigir indentação extra: {e}")
                
        elif issues['suggested_fix'] == 'add_missing_indentation':
            try:
                fixed_code = fix_missing_indentation(code)
                # Verificar se a correção funcionou
                ast.parse(fixed_code)
                logger.info(f"Corrigido problema de indentação faltante na linha {issues['line_number']}")
                return fixed_code
            except Exception as e:
                logger.warning(f"Falha ao corrigir indentação faltante: {e}")
    
    # Se não conseguiu corrigir, retorna o código original
    logger.warning(f"Não foi possível corrigir automaticamente: {issues['error_message']}")
    return code


def clean_code_for_ast(code: str, traceback_type: Optional[str] = None) -> str:
    """
    Limpa o código para análise AST, considerando o tipo de erro.
    
    Args:
        code: Código a ser limpo
        traceback_type: Tipo de erro do traceback (opcional)
        
    Returns:
        Código limpo e válido para análise AST
    """
    # Se não é erro de indentação, tentar corrigir automaticamente
    if traceback_type and traceback_type not in ['IndentationError', 'SyntaxError']:
        # Tentar corrigir problemas de indentação mesmo que não seja o erro principal
        cleaned_code = auto_fix_indentation(code)
        
        # Verificar se a correção funcionou
        try:
            ast.parse(cleaned_code)
            return cleaned_code
        except Exception:
            # Se não funcionou, retorna o código original
            return code
    
    # Se é erro de sintaxe/indentação, tentar corrigir
    return auto_fix_indentation(code)


def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Valida se o código tem sintaxe válida.
    
    Args:
        code: Código a ser validado
        
    Returns:
        Tupla (é_válido, mensagem_erro)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Erro inesperado: {e}"


def get_code_metrics(code: str) -> Dict[str, Any]:
    """
    Calcula métricas básicas do código.
    
    Args:
        code: Código a ser analisado
        
    Returns:
        Dicionário com métricas do código
    """
    try:
        tree = ast.parse(code)
        
        # Contar diferentes tipos de nós
        node_counts = {
            'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            'calls': len([n for n in ast.walk(tree) if isinstance(n, ast.Call)]),
            'assignments': len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)]),
            'if_statements': len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
            'loops': len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]),
            'try_blocks': len([n for n in ast.walk(tree) if isinstance(n, ast.Try)]),
        }
        
        return {
            'is_valid': True,
            'line_count': len(code.split('\n')),
            'char_count': len(code),
            'node_counts': node_counts
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'error': str(e),
            'line_count': len(code.split('\n')),
            'char_count': len(code)
        }


def batch_clean_codes(codes: list, traceback_types: Optional[list] = None) -> list:
    """
    Limpa uma lista de códigos em lote.
    
    Args:
        codes: Lista de códigos a serem limpos
        traceback_types: Lista de tipos de traceback correspondentes
        
    Returns:
        Lista de códigos limpos
    """
    cleaned_codes = []
    
    for i, code in enumerate(codes):
        traceback_type = traceback_types[i] if traceback_types else None
        cleaned_code = clean_code_for_ast(code, traceback_type)
        cleaned_codes.append(cleaned_code)
        
        # Log de progresso
        if i % 100 == 0:
            logger.info(f"Processados {i}/{len(codes)} códigos")
    
    return cleaned_codes


if __name__ == "__main__":
    # Testes
    test_codes = [
        # Código com indentação extra
        """    def test():
        return True""",
        
        # Código com indentação faltante
        """def test():
    if True:
    return True""",
        
        # Código válido
        """def test():
    if True:
        return True""",
    ]
    
    print("=== TESTE DE LIMPEZA DE CÓDIGO ===\n")
    
    for i, code in enumerate(test_codes):
        print(f"Teste {i+1}:")
        print(f"Código original:\n{repr(code)}")
        
        issues = detect_indentation_issues(code)
        print(f"Problemas detectados: {issues}")
        
        cleaned = clean_code_for_ast(code)
        print(f"Código limpo:\n{repr(cleaned)}")
        
        is_valid, error = validate_code_syntax(cleaned)
        print(f"Válido após limpeza: {is_valid}")
        if not is_valid:
            print(f"Erro: {error}")
        
        print("-" * 50)
