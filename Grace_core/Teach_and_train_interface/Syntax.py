#!/usr/bin/env python3
"""
Syntax Specialist Module

The Syntax Specialist is Grace's logic mechanic. It reads code, logic blocks, or structured
language and repairs, restructures, or refactors it. It ensures inputs are syntactically valid,
structurally sound, and ready for execution or further refinement.
"""

import ast
import logging
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

# Mock imports for external dependencies
try:
    from cryptography.fernet import Fernet
except ImportError:
    # Mock Fernet class if cryptography module is not available
    class Fernet:
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            return b"mock_encrypted_" + data
        
        def decrypt(self, token):
            if token.startswith(b"mock_encrypted_"):
                return token[len(b"mock_encrypted_"):]
            return b"mock_decrypted_data"

try:
    from flask import Flask, request, jsonify
    from werkzeug.exceptions import BadRequest
except ImportError:
    # Mock Flask and related classes if flask module is not available
    class Flask:
        def __init__(self, name):
            self.name = name
            self.routes = {}
        
        def route(self, route_str, **options):
            def decorator(f):
                self.routes[route_str] = f
                return f
            return decorator
    
    class MockRequest:
        def __init__(self):
            self.json = {}
            self.args = {}
            self.form = {}
    
    request = MockRequest()
    
    def jsonify(obj):
        return obj
    
    class BadRequest(Exception):
        pass

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import config_loader, logger_setup

# Setup logging
logger = logging.getLogger(__name__)

class SyntaxSpecialist:
    """
    The Syntax Specialist class is responsible for analyzing, repairing, and optimizing code
    syntax across various programming languages.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Syntax Specialist with configuration settings.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = config_loader.load_config(config_path)
        self.logger = logger_setup.setup_logger(
            self.config.get("logging", {}).get("level", "INFO"),
            __name__
        )
        self.supported_languages = self.config.get("supported_languages", ["python", "javascript"])
        self.logger.info(f"Syntax Specialist initialized with {len(self.supported_languages)} languages")
    
    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze code for syntax errors and structural issues.
        
        Args:
            code: The code to analyze
            language: The programming language of the code
            
        Returns:
            Dictionary containing analysis results
        """
        if language not in self.supported_languages:
            self.logger.warning(f"Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._analyze_python(code)
            elif language == "javascript":
                return self._analyze_javascript(code)
            else:
                return {"error": f"Analysis for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error analyzing {language} code: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code for syntax errors and structural issues.
        
        Args:
            code: The Python code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        results = {"syntax_valid": False, "issues": []}
        
        # Check for syntax errors
        try:
            ast.parse(code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["issues"].append({
                "type": "syntax_error",
                "line": e.lineno,
                "column": e.offset,
                "message": str(e)
            })
            return results
        
        # Check for style issues
        results["style_issues"] = self._check_python_style(code)
        
        # Check for potential logical issues
        results["logical_issues"] = self._check_python_logic(code)
        
        return results
    
    def _check_python_style(self, code: str) -> List[Dict[str, Any]]:
        """
        Check Python code for style issues.
        
        Args:
            code: The Python code to check
            
        Returns:
            List of style issues found
        """
        issues = []
        
        # Check line length
        for i, line in enumerate(code.split('\n')):
            if len(line) > 100:
                issues.append({
                    "type": "line_too_long",
                    "line": i + 1,
                    "message": f"Line exceeds 100 characters ({len(line)})"
                })
        
        # Check for trailing whitespace
        for i, line in enumerate(code.split('\n')):
            if line.rstrip() != line:
                issues.append({
                    "type": "trailing_whitespace",
                    "line": i + 1,
                    "message": "Line contains trailing whitespace"
                })
        
        return issues
    
    def _check_python_logic(self, code: str) -> List[Dict[str, Any]]:
        """
        Check Python code for potential logical issues.
        
        Args:
            code: The Python code to check
            
        Returns:
            List of logical issues found
        """
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for unused imports
            imported = set()
            used = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imported.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imported.add(name.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.add(node.id)
            
            unused_imports = imported - used
            for name in unused_imports:
                issues.append({
                    "type": "unused_import",
                    "message": f"Unused import: {name}"
                })
            
            # Check for bare except clauses
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append({
                        "type": "bare_except",
                        "line": node.lineno,
                        "message": "Bare except clause used"
                    })
        except Exception as e:
            self.logger.error(f"Error checking Python logic: {str(e)}")
        
        return issues
    
    def _analyze_javascript(self, code: str) -> Dict[str, Any]:
        """
        Analyze JavaScript code for syntax errors and structural issues.
        
        Args:
            code: The JavaScript code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # This is a placeholder implementation
        results = {"syntax_valid": False, "issues": []}
        
        # Basic syntax check using regex patterns
        try:
            # Check for unbalanced braces
            if code.count('{') != code.count('}'):
                results["issues"].append({
                    "type": "unbalanced_braces",
                    "message": "Unbalanced curly braces"
                })
            
            # Check for unbalanced parentheses
            if code.count('(') != code.count(')'):
                results["issues"].append({
                    "type": "unbalanced_parentheses",
                    "message": "Unbalanced parentheses"
                })
            
            # Check for unbalanced brackets
            if code.count('[') != code.count(']'):
                results["issues"].append({
                    "type": "unbalanced_brackets",
                    "message": "Unbalanced square brackets"
                })
            
            if not results["issues"]:
                results["syntax_valid"] = True
        except Exception as e:
            results["issues"].append({
                "type": "analysis_error",
                "message": str(e)
            })
        
        return results
    
    def repair_code(self, code: str, language: str, issues: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Attempt to repair code based on identified issues.
        
        Args:
            code: The code to repair
            language: The programming language of the code
            issues: Optional list of issues to fix
            
        Returns:
            Dictionary containing the repaired code and repair notes
        """
        if language not in self.supported_languages:
            return {"error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._repair_python(code, issues)
            elif language == "javascript":
                return self._repair_javascript(code, issues)
            else:
                return {"error": f"Repair for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error repairing {language} code: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _repair_python(self, code: str, issues: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Attempt to repair Python code based on identified issues.
        
        Args:
            code: The Python code to repair
            issues: Optional list of issues to fix
            
        Returns:
            Dictionary containing the repaired code and repair notes
        """
        if not issues:
            # If no issues provided, analyze the code first
            analysis = self._analyze_python(code)
            if not analysis.get("syntax_valid", False):
                issues = analysis.get("issues", [])
            else:
                return {"repaired_code": code, "notes": ["No syntax issues to repair"]}
        
        repaired_code = code
        repair_notes = []
        
        # Fix indentation issues
        repaired_code, indent_notes = self._fix_python_indentation(repaired_code)
        repair_notes.extend(indent_notes)
        
        # Fix missing colons in control structures
        repaired_code, colon_notes = self._fix_python_missing_colons(repaired_code)
        repair_notes.extend(colon_notes)
        
        # Fix unbalanced parentheses, brackets, and braces
        repaired_code, balance_notes = self._fix_python_unbalanced_delimiters(repaired_code)
        repair_notes.extend(balance_notes)
        
        # Fix incomplete try-except blocks
        repaired_code, try_except_notes = self._fix_python_try_except(repaired_code)
        repair_notes.extend(try_except_notes)
        
        # Validate the repaired code
        try:
            ast.parse(repaired_code)
            repair_notes.append("Syntax is now valid")
        except SyntaxError as e:
            repair_notes.append(f"Could not fully repair syntax: {str(e)}")
        
        return {"repaired_code": repaired_code, "notes": repair_notes}
    
    def _fix_python_indentation(self, code: str) -> Tuple[str, List[str]]:
        """
        Fix indentation issues in Python code.
        
        Args:
            code: The Python code to fix
            
        Returns:
            Tuple of (fixed_code, notes)
        """
        lines = code.split('\n')
        fixed_lines = []
        notes = []
        current_indent = 0
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Calculate current line's indentation
            indent = len(line) - len(stripped)
            
            # Check for indentation increase (new block)
            if stripped.endswith(':'):
                fixed_lines.append(' ' * current_indent + stripped)
                current_indent += 4
                indent_stack.append(current_indent)
                continue
            
            # Check for block end (dedent)
            if indent < indent_stack[-1]:
                while indent_stack and indent < indent_stack[-1]:
                    indent_stack.pop()
                    current_indent -= 4
                
                if not indent_stack:
                    indent_stack = [0]
                    current_indent = 0
                    notes.append(f"Fixed unexpected dedent at line {i+1}")
            
            # Apply the correct indentation
            fixed_lines.append(' ' * current_indent + stripped)
        
        return '\n'.join(fixed_lines), notes
    
    def _fix_python_missing_colons(self, code: str) -> Tuple[str, List[str]]:
        """
        Fix missing colons in Python control structures.
        
        Args:
            code: The Python code to fix
            
        Returns:
            Tuple of (fixed_code, notes)
        """
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally']
        lines = code.split('\n')
        fixed_lines = []
        notes = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Check if line starts with a control keyword but doesn't end with a colon
            for keyword in control_keywords:
                if (stripped.startswith(keyword + ' ') or stripped == keyword) and not stripped.endswith(':'):
                    # Don't add colon to 'else' if it's part of a ternary expression
                    if keyword == 'else' and ' if ' in stripped:
                        fixed_lines.append(line)
                        continue
                    
                    fixed_lines.append(line + ':')
                    notes.append(f"Added missing colon at line {i+1}")
                    break
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), notes
    
    def _fix_python_unbalanced_delimiters(self, code: str) -> Tuple[str, List[str]]:
        """
        Fix unbalanced parentheses, brackets, and braces in Python code.
        
        Args:
            code: The Python code to fix
            
        Returns:
            Tuple of (fixed_code, notes)
        """
        notes = []
        
        # Check and fix parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        
        if open_parens > close_parens:
            code += ')' * (open_parens - close_parens)
            notes.append(f"Added {open_parens - close_parens} missing closing parentheses")
        if open_parens > close_parens:
            code += ')' * (open_parens - close_parens)
            notes.append(f"Added {open_parens - close_parens} missing closing parentheses")
        elif close_parens > open_parens:
            code = '(' * (close_parens - open_parens) + code
            notes.append(f"Added {close_parens - open_parens} missing opening parentheses")
        
        # Check and fix brackets
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        
        if open_brackets > close_brackets:
            code += ']' * (open_brackets - close_brackets)
            notes.append(f"Added {open_brackets - close_brackets} missing closing brackets")
        elif close_brackets > open_brackets:
            code = '[' * (close_brackets - open_brackets) + code
            notes.append(f"Added {close_brackets - open_brackets} missing opening brackets")
        
        # Check and fix braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        
        if open_braces > close_braces:
            code += '}' * (open_braces - close_braces)
            notes.append(f"Added {open_braces - close_braces} missing closing braces")
        elif close_braces > open_braces:
            code = '{' * (close_braces - open_braces) + code
            notes.append(f"Added {close_braces - open_braces} missing opening braces")
        
        return code, notes
    
    def _fix_python_try_except(self, code: str) -> Tuple[str, List[str]]:
        """
        Fix incomplete try-except blocks in Python code.
        
        Args:
            code: The Python code to fix
            
        Returns:
            Tuple of (fixed_code, notes)
        """
        lines = code.split('\n')
        fixed_lines = []
        notes = []
        in_try_block = False
        has_except_or_finally = False
        try_line_number = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Check for try statement
            if stripped == 'try:' or stripped.startswith('try:'):
                in_try_block = True
                try_line_number = i + 1
                fixed_lines.append(line)
                continue
            
            # Check for except or finally statement
            if in_try_block and (stripped.startswith('except') or stripped == 'finally:'):
                has_except_or_finally = True
                fixed_lines.append(line)
                continue
            
            # If we're at the end of a try block without except or finally, add a generic except
            if in_try_block and not has_except_or_finally:
                # Check if the current line has less indentation than the try block
                # which would indicate the end of the try block
                try_indent = len(lines[try_line_number - 1]) - len(lines[try_line_number - 1].lstrip())
                current_indent = len(line) - len(stripped)
                
                if current_indent <= try_indent and i > try_line_number:
                    # Add a generic except before the current line
                    indent = ' ' * try_indent
                    fixed_lines.append(f"{indent}except Exception as e:")
                    fixed_lines.append(f"{indent}    pass  # Added by syntax specialist")
                    notes.append(f"Added missing except clause after try at line {try_line_number}")
                    in_try_block = False
                    has_except_or_finally = False
            
            fixed_lines.append(line)
            
            # Reset flags if we're moving to a new block
            if in_try_block and has_except_or_finally and stripped.endswith(':'):
                in_try_block = False
                has_except_or_finally = False
        
        # If we're still in a try block at the end of the file, add a generic except
        if in_try_block and not has_except_or_finally:
            try_indent = len(lines[try_line_number - 1]) - len(lines[try_line_number - 1].lstrip())
            indent = ' ' * try_indent
            fixed_lines.append(f"{indent}except Exception as e:")
            fixed_lines.append(f"{indent}    pass  # Added by syntax specialist")
            notes.append(f"Added missing except clause after try at line {try_line_number}")
        
        return '\n'.join(fixed_lines), notes
    
    def _repair_javascript(self, code: str, issues: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Attempt to repair JavaScript code based on identified issues.
        
        Args:
            code: The JavaScript code to repair
            issues: Optional list of issues to fix
            
        Returns:
            Dictionary containing the repaired code and repair notes
        """
        # This is a placeholder implementation
        repaired_code = code
        repair_notes = ["JavaScript repair is limited to basic delimiter balancing"]
        
        # Fix unbalanced delimiters
        delimiters = [
            ('(', ')'),
            ('[', ']'),
            ('{', '}')
        ]
        
        for open_delim, close_delim in delimiters:
            open_count = repaired_code.count(open_delim)
            close_count = repaired_code.count(close_delim)
            
            if open_count > close_count:
                repaired_code += close_delim * (open_count - close_count)
                repair_notes.append(f"Added {open_count - close_count} missing {close_delim}")
            elif close_count > open_count:
                repaired_code = open_delim * (close_count - open_count) + repaired_code
                repair_notes.append(f"Added {close_count - open_count} missing {open_delim}")
        
        # Fix missing semicolons (very basic approach)
        lines = repaired_code.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if (stripped and 
                not stripped.endswith(';') and 
                not stripped.endswith('{') and 
                not stripped.endswith('}') and 
                not stripped.endswith(':') and
                not stripped.startswith('//')):
                fixed_lines.append(line + ';')
                repair_notes.append("Added missing semicolons")
            else:
                fixed_lines.append(line)
        
        repaired_code = '\n'.join(fixed_lines)
        
        return {"repaired_code": repaired_code, "notes": repair_notes}
    
    def optimize_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Optimize code for better performance or readability.
        
        Args:
            code: The code to optimize
            language: The programming language of the code
            
        Returns:
            Dictionary containing the optimized code and optimization notes
        """
        if language not in self.supported_languages:
            return {"error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._optimize_python(code)
            elif language == "javascript":
                return self._optimize_javascript(code)
            else:
                return {"error": f"Optimization for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error optimizing {language} code: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _optimize_python(self, code: str) -> Dict[str, Any]:
        """
        Optimize Python code for better performance or readability.
        
        Args:
            code: The Python code to optimize
            
        Returns:
            Dictionary containing the optimized code and optimization notes
        """
        optimized_code = code
        optimization_notes = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Remove unused imports
            optimized_code, import_notes = self._remove_unused_imports(optimized_code)
            optimization_notes.extend(import_notes)
            
            # Simplify boolean expressions
            optimized_code, bool_notes = self._simplify_boolean_expressions(optimized_code)
            optimization_notes.extend(bool_notes)
            
            # Format long lines
            optimized_code, format_notes = self._format_long_lines(optimized_code)
            optimization_notes.extend(format_notes)
        except Exception as e:
            optimization_notes.append(f"Optimization error: {str(e)}")
        
        return {"optimized_code": optimized_code, "notes": optimization_notes}
    
    def _remove_unused_imports(self, code: str) -> Tuple[str, List[str]]:
        """
        Remove unused imports from Python code.
        
        Args:
            code: The Python code to process
            
        Returns:
            Tuple of (optimized_code, notes)
        """
        try:
            tree = ast.parse(code)
            
            # Find all imports
            imports = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports[name.name] = name.asname or name.name
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        full_name = f"{node.module}.{name.name}" if node.module else name.name
                        imports[full_name] = name.asname or name.name
            
            # Find all used names
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
            
            # Find unused imports
            unused_imports = []
            for import_name, alias in imports.items():
                if alias not in used_names:
                    unused_imports.append(import_name)
            
            if not unused_imports:
                return code, []
            
            # Remove unused imports
            lines = code.split('\n')
            filtered_lines = []
            notes = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # Very basic check - this won't catch all cases
                    should_remove = False
                    for unused in unused_imports:
                        if unused in stripped:
                            should_remove = True
                            notes.append(f"Removed unused import: {unused}")
                            break
                    
                    if not should_remove:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines), notes
        except Exception as e:
            return code, [f"Error removing unused imports: {str(e)}"]
    
    def _simplify_boolean_expressions(self, code: str) -> Tuple[str, List[str]]:
        """
        Simplify boolean expressions in Python code.
        
        Args:
            code: The Python code to process
            
        Returns:
            Tuple of (optimized_code, notes)
        """
        # This is a placeholder with some basic regex-based simplifications
        notes = []
        
        # Replace "if x == True:" with "if x:"
        pattern = r'if\s+(\w+)\s*==\s*True\s*:'
        replacement = r'if \1:'
        new_code, count = re.subn(pattern, replacement, code)
        if count > 0:
            notes.append(f"Simplified {count} boolean comparisons with True")
            code = new_code
        
        # Replace "if x == False:" with "if not x:"
        pattern = r'if\s+(\w+)\s*==\s*False\s*:'
        replacement = r'if not \1:'
        new_code, count = re.subn(pattern, replacement, code)
        if count > 0:
            notes.append(f"Simplified {count} boolean comparisons with False")
            code = new_code
        
        return code, notes
    
    def _format_long_lines(self, code: str) -> Tuple[str, List[str]]:
        """
        Format long lines in Python code to improve readability.
        
        Args:
            code: The Python code to process
            
        Returns:
            Tuple of (optimized_code, notes)
        """
        lines = code.split('\n')
        formatted_lines = []
        notes = []
        long_lines_count = 0
        
        for line in lines:
            if len(line) > 100:
                # Try to break at a comma for function calls or lists
                if '(' in line and ')' in line and ',' in line:
                    indent = len(line) - len(line.lstrip())
                    parts = []
                    current_part = ""
                    in_string = False
                    string_char = None
                    
                    for char in line:
                        current_part += char
                        
                        # Track string literals to avoid breaking inside them
                        if char in ('"', "'") and (not in_string or string_char == char):
                            if in_string:
                                in_string = False
                                string_char = None
                            else:
                                in_string = True
                                string_char = char
                        
                        # Break at commas, but not inside strings
                        if char == ',' and not in_string and len(current_part) > 50:
                            parts.append(current_part)
                            current_part = ' ' * (indent + 4)  # Indent continuation lines
                    
                    if current_part:
                        parts.append(current_part)
                    
                    formatted_lines.extend(parts)
                    long_lines_count += 1
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        if long_lines_count > 0:
            notes.append(f"Formatted {long_lines_count} long lines")
        
        return '\n'.join(formatted_lines), notes
    
    def _optimize_javascript(self, code: str) -> Dict[str, Any]:
        """
        Optimize JavaScript code for better performance or readability.
        
        Args:
            code: The JavaScript code to optimize
            
        Returns:
            Dictionary containing the optimized code and optimization notes
        """
        # This is a placeholder implementation
        return {
            "optimized_code": code,
            "notes": ["JavaScript optimization not fully implemented yet"]
        }
    
    def format_code(self, code: str, language: str, style_guide: Optional[str] = None) -> Dict[str, Any]:
        """
        Format code according to a specified style guide.
        
        Args:
            code: The code to format
            language: The programming language of the code
            style_guide: Optional style guide to follow (e.g., "pep8" for Python)
            
        Returns:
            Dictionary containing the formatted code and formatting notes
        """
        if language not in self.supported_languages:
            return {"error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._format_python(code, style_guide)
            elif
            if language == "python":
                return self._format_python(code, style_guide)
            elif language == "javascript":
                return self._format_javascript(code, style_guide)
            else:
                return {"error": f"Formatting for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error formatting {language} code: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _format_python(self, code: str, style_guide: Optional[str] = None) -> Dict[str, Any]:
        """
        Format Python code according to a specified style guide.
        
        Args:
            code: The Python code to format
            style_guide: Optional style guide to follow (default: "pep8")
            
        Returns:
            Dictionary containing the formatted code and formatting notes
        """
        if not style_guide:
            style_guide = "pep8"
        
        formatted_code = code
        formatting_notes = [f"Applied {style_guide} style guide"]
        
        # Apply basic PEP 8 formatting
        if style_guide.lower() == "pep8":
            # Fix indentation (use 4 spaces)
            lines = formatted_code.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('\t'):
                    lines[i] = line.replace('\t', '    ')
            
            formatted_code = '\n'.join(lines)
            formatting_notes.append("Replaced tabs with 4 spaces")
            
            # Add two blank lines before top-level function and class definitions
            formatted_code, count = re.subn(
                r'([^\n])\n(def |class )',
                r'\1\n\n\n\2',
                formatted_code
            )
            if count > 0:
                formatting_notes.append(f"Added proper spacing before {count} function/class definitions")
            
            # Add a single blank line before method definitions
            formatted_code, count = re.subn(
                r'(\n    [^\n]+\n)    def ',
                r'\1\n    def ',
                formatted_code
            )
            if count > 0:
                formatting_notes.append(f"Added proper spacing before {count} method definitions")
            
            # Ensure single space after commas
            formatted_code, count = re.subn(r',\s*', ', ', formatted_code)
            if count > 0:
                formatting_notes.append("Fixed spacing after commas")
            
            # Ensure single space around operators
            for op in ['=', '+', '-', '*', '/', '//', '%', '**', '==', '!=', '<', '>', '<=', '>=']:
                formatted_code, count = re.subn(r'\s*' + re.escape(op) + r'\s*', f' {op} ', formatted_code)
                if count > 0:
                    formatting_notes.append(f"Fixed spacing around {op} operator")
        
        return {"formatted_code": formatted_code, "notes": formatting_notes}
    
    def _format_javascript(self, code: str, style_guide: Optional[str] = None) -> Dict[str, Any]:
        """
        Format JavaScript code according to a specified style guide.
        
        Args:
            code: The JavaScript code to format
            style_guide: Optional style guide to follow
            
        Returns:
            Dictionary containing the formatted code and formatting notes
        """
        # This is a placeholder implementation
        return {
            "formatted_code": code,
            "notes": ["JavaScript formatting not fully implemented yet"]
        }
    
    def validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validate the syntax of code without executing it.
        
        Args:
            code: The code to validate
            language: The programming language of the code
            
        Returns:
            Dictionary containing validation results
        """
        if language not in self.supported_languages:
            return {"valid": False, "error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._validate_python_syntax(code)
            elif language == "javascript":
                return self._validate_javascript_syntax(code)
            else:
                return {"valid": False, "error": f"Validation for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error validating {language} code: {str(e)}")
            return {"valid": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate the syntax of Python code without executing it.
        
        Args:
            code: The Python code to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error_type": "SyntaxError",
                "message": str(e),
                "line": e.lineno,
                "column": e.offset,
                "text": e.text
            }
        except Exception as e:
            return {
                "valid": False,
                "error_type": type(e).__name__,
                "message": str(e)
            }
    
    def _validate_javascript_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate the syntax of JavaScript code without executing it.
        
        Args:
            code: The JavaScript code to validate
            
        Returns:
            Dictionary containing validation results
        """
        # This is a placeholder implementation with basic checks
        try:
            # Check for unbalanced delimiters
            delimiters = [
                ('(', ')'),
                ('[', ']'),
                ('{', '}')
            ]
            
            for open_delim, close_delim in delimiters:
                if code.count(open_delim) != code.count(close_delim):
                    return {
                        "valid": False,
                        "error_type": "SyntaxError",
                        "message": f"Unbalanced delimiters: {open_delim} and {close_delim}"
                    }
            
            # Check for unterminated strings
            in_string = False
            string_char = None
            escaped = False
            
            for i, char in enumerate(code):
                if not in_string:
                    if char in ('"', "'"):
                        in_string = True
                        string_char = char
                else:
                    if escaped:
                        escaped = False
                    elif char == '\\':
                        escaped = True
                    elif char == string_char:
                        in_string = False
                        string_char = None
            
            if in_string:
                return {
                    "valid": False,
                    "error_type": "SyntaxError",
                    "message": f"Unterminated string literal"
                }
            
            return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "error_type": type(e).__name__,
                "message": str(e)
            }
    
    def extract_functions(self, code: str, language: str) -> Dict[str, Any]:
        """
        Extract function definitions from code.
        
        Args:
            code: The code to analyze
            language: The programming language of the code
            
        Returns:
            Dictionary containing extracted functions
        """
        if language not in self.supported_languages:
            return {"error": f"Unsupported language: {language}"}
        
        try:
            if language == "python":
                return self._extract_python_functions(code)
            elif language == "javascript":
                return self._extract_javascript_functions(code)
            else:
                return {"error": f"Function extraction for {language} not implemented yet"}
        except Exception as e:
            self.logger.error(f"Error extracting functions from {language} code: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _extract_python_functions(self, code: str) -> Dict[str, Any]:
        """
        Extract function definitions from Python code.
        
        Args:
            code: The Python code to analyze
            
        Returns:
            Dictionary containing extracted functions
        """
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    
                    # Extract default values
                    defaults = []
                    for default in node.args.defaults:
                        if isinstance(default, ast.Constant):
                            defaults.append(default.value)
                        else:
                            defaults.append("complex_default")
                    
                    # Extract docstring if available
                    docstring = None
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        docstring = node.body[0].value.s
                    
                    # Extract function source code
                    function_lines = []
                    for i in range(node.lineno, node.end_lineno + 1):
                        function_lines.append(code.split('\n')[i - 1])
                    
                    functions.append({
                        "name": node.name,
                        "args": args,
                        "defaults": defaults,
                        "docstring": docstring,
                        "source": '\n'.join(function_lines),
                        "line_start": node.lineno,
                        "line_end": node.end_lineno
                    })
        except Exception as e:
            return {"error": str(e), "functions": []}
        
        return {"functions": functions}
    
    def _extract_javascript_functions(self, code: str) -> Dict[str, Any]:
        """
        Extract function definitions from JavaScript code.
        
        Args:
            code: The JavaScript code to analyze
            
        Returns:
            Dictionary containing extracted functions
        """
        # This is a placeholder implementation with basic regex extraction
        functions = []
        
        # Match function declarations
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        matches = re.finditer(func_pattern, code)
        
        for match in matches:
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(',') if p.strip()]
            body = match.group(3)
            
            functions.append({
                "name": name,
                "params": params,
                "source": match.group(0),
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        # Match arrow functions with names (const/let/var assignments)
        arrow_pattern = r'(const|let|var)\s+(\w+)\s*=\s*(\([^)]*\)|[^=]*)\s*=>\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\}|[^;]*)'
        matches = re.finditer(arrow_pattern, code)
        
        for match in matches:
            name = match.group(2)
            params_str = match.group(3)
            if params_str.startswith('(') and params_str.endswith(')'):
                params = [p.strip() for p in params_str[1:-1].split(',') if p.strip()]
            else:
                params = [params_str]
            
            functions.append({
                "name": name,
                "params": params,
                "source": match.group(0),
                "start_pos": match.start(),
                "end_pos": match.end(),
                "type": "arrow"
            })
        
        return {"functions": functions}


# API Server for the Syntax Specialist
def create_app(specialist=None):
    """
    Create a Flask application for the Syntax Specialist API.
    
    Args:
        specialist: Optional SyntaxSpecialist instance
        
    Returns:
        Flask application
    """
    app = Flask(__name__)
    
    if specialist is None:
        specialist = SyntaxSpecialist()
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            result = specialist.analyze_code(data['code'], data['language'])
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/repair', methods=['POST'])
    def repair():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            issues = data.get('issues')
            result = specialist.repair_code(data['code'], data['language'], issues)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/optimize', methods=['POST'])
    def optimize():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            result = specialist.optimize_code(data['code'], data['language'])
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/format', methods=['POST'])
    def format_code():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            style_guide = data.get('style_guide')
            result = specialist.format_code(data['code'], data['language'], style_guide)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/validate', methods=['POST'])
    def validate():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            result = specialist.validate_syntax(data['code'], data['language'])
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/extract-functions', methods=['POST'])
    def extract_functions():
    @app.route('/extract-functions', methods=['POST'])
    def extract_functions():
        try:
            data = request.json
            if not data or 'code' not in data or 'language' not in data:
                return jsonify({
                    "error": "Missing required fields: code and language"
                }), 400
            
            result = specialist.extract_functions(data['code'], data['language'])
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "supported_languages": specialist.supported_languages
        })
    
    return app


if __name__ == "__main__":
    # Setup command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Syntax Specialist API Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000, 
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the server on"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run in debug mode"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize the Syntax Specialist
    specialist = SyntaxSpecialist(args.config)
    
    # Create and run the Flask app
    app = create_app(specialist)
    app.run(host=args.host, port=args.port, debug=args.debug)
