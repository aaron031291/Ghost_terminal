"""Self-Learning Loop for Grace.

This module implements the feedback loop for code improvement and learning.
"""

import logging
import re
import time
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import json
from pathlib import Path

from grace.config import Config
from grace.execution import CodeExecutionKernel, ExecutionResult

logger = logging.getLogger("grace.learning")


class ErrorPattern:
    """Pattern for recognizing and fixing common errors."""

    def __init__(self, pattern: str, description: str, fix_function: Callable[[str, re.Match], str]):
        """Initialize error pattern.
        
        Args:
            pattern: Regular expression pattern to match error
            description: Human-readable description of the error
            fix_function: Function that takes code and match object and returns fixed code
        """
        self.pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        self.description = description
        self.fix_function = fix_function
        
    def match(self, error_message: str) -> Optional[re.Match]:
        """Check if error message matches this pattern.
        
        Args:
            error_message: Error message to check
            
        Returns:
            Match object if pattern matches, None otherwise
        """
        return self.pattern.search(error_message)
        
    def apply_fix(self, code: str, match: re.Match) -> str:
        """Apply fix to code based on error match.
        
        Args:
            code: Original code
            match: Match object from error pattern
            
        Returns:
            Fixed code
        """
        return self.fix_function(code, match)


class LearningResult:
    """Result of a learning iteration."""

    def __init__(
        self,
        original_code: str,
        improved_code: str,
        execution_results: List[ExecutionResult],
        errors_fixed: List[str],
        improvement_score: float,
    ):
        """Initialize learning result.
        
        Args:
            original_code: Original code
            improved_code: Improved code after learning
            execution_results: List of execution results from attempts
            errors_fixed: List of error descriptions that were fixed
            improvement_score: Score indicating improvement (0-1)
        """
        self.original_code = original_code
        self.improved_code = improved_code
        self.execution_results = execution_results
        self.errors_fixed = errors_fixed
        self.improvement_score = improvement_score
        
    def __str__(self) -> str:
        """String representation of learning result."""
        return (
            f"Learning Result:\n"
            f"Improvement Score: {self.improvement_score:.2f}\n"
            f"Errors Fixed: {len(self.errors_fixed)}\n"
            f"Execution Attempts: {len(self.execution_results)}\n"
            f"Final Success: {self.execution_results[-1].success if self.execution_results else False}"
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_code": self.original_code,
            "improved_code": self.improved_code,
            "execution_results": [result.to_dict() for result in self.execution_results],
            "errors_fixed": self.errors_fixed,
            "improvement_score": self.improvement_score,
        }


class SelfLearningLoop:
    """Learning loop that improves code through execution and feedback."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize self-learning loop.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.kernel = CodeExecutionKernel(config)
        self.max_attempts = self.config.get("learning", "max_attempts")
        self.error_threshold = self.config.get("learning", "error_threshold")
        self.success_threshold = self.config.get("learning", "success_threshold")
        self.learning_rate = self.config.get("learning", "learning_rate")
        
        # Initialize error patterns
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize common error patterns and fixes.
        
        Returns:
            List of ErrorPattern objects
        """
        patterns = []
        
        # Syntax error: Missing closing parenthesis
        patterns.append(
            ErrorPattern(
                r"SyntaxError: unexpected EOF while parsing|SyntaxError: \(\'EOF\'",
                "Missing closing parenthesis, bracket, or brace",
                lambda code, match: self._fix_missing_closing_delimiter(code)
            )
        )
        
        # Syntax error: Invalid syntax
        patterns.append(
            ErrorPattern(
                r"SyntaxError: invalid syntax",
                "Invalid syntax",
                lambda code, match: self._fix_invalid_syntax(code)
            )
        )
        
        # NameError: Name not defined
        patterns.append(
            ErrorPattern(
                r"NameError: name \'(\w+)\' is not defined",
                "Undefined variable",
                lambda code, match: self._fix_undefined_variable(code, match)
            )
        )
        
        # ImportError: Module not found
        patterns.append(
            ErrorPattern(
                r"ImportError: No module named \'([^\']+)\'",
                "Missing import",
                lambda code, match: self._fix_missing_import(code, match)
            )
        )
        
        # IndentationError
        patterns.append(
            ErrorPattern(
                r"IndentationError: ([^\n]+)",
                "Indentation error",
                lambda code, match: self._fix_indentation(code)
            )
        )
        
        # Add more patterns as needed
        
        return patterns
        
    def _fix_missing_closing_delimiter(self, code: str) -> str:
        """Fix missing closing delimiters (parentheses, brackets, braces).
        
        Args:
            code: Original code
            
        Returns:
            Fixed code
        """
        # Count opening and closing delimiters
        delimiters = {
            '(': ')',
            '[': ']',
            '{': '}',
        }
        
        counts = {char: 0 for char in delimiters.keys() | delimiters.values()}
        
        for char in code:
            if char in counts:
                counts[char] += 1
                
        # Add missing closing delimiters
        fixed_code = code
        for opener, closer in delimiters.items():
            missing = counts[opener] - counts[closer]
            if missing > 0:
                fixed_code += closer * missing
                
        return fixed_code
        
    def _fix_invalid_syntax(self, code: str) -> str:
        """Attempt to fix invalid syntax.
        
        Args:
            code: Original code
            
        Returns:
            Fixed code (or original if no fix found)
        """
        # This is a simplified implementation
        # In a real system, you'd want more sophisticated syntax fixing
        
        # Check for common issues
        
        # Missing colons after control statements
        fixed_code = re.sub(
            r'(if|for|while|def|class|with|try|except|finally)([^:]*?)(\n|\r\n?)',
            r'\1\2:\3',
            code
        )
        
        # Fix string quotes
        fixed_code = self._fix_string_quotes(fixed_code)
        
        return fixed_code
        
    def _fix_string_quotes(self, code: str) -> str:
        """Fix mismatched string quotes.
        
        Args:
            code: Original code
            
        Returns:
            Fixed code
        """
        # This is a simplified implementation
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count quotes
            single_quotes = line.count("'")
            double_quotes = line.count('"')
            
            # If odd number of quotes, try to fix
            if single_quotes % 2 == 1:
                # Add closing quote at end if it seems appropriate
                if not line.strip().endswith("'"):
                    line += "'"
            
            if double_quotes % 2 == 1:
                # Add closing quote at end if it seems appropriate
                if not line.strip().endswith('"'):
                    line += '"'
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def _fix_undefined_variable(self, code: str, match: re.Match) -> str:
        """Fix undefined variable by adding a declaration.
        
        Args:
            code: Original code
            match: Regex match containing the variable name
            
        Returns:
            Fixed code
        """
        var_name = match.group(1)
        
        # Simple heuristic: add variable initialization at the beginning
        # In a real system, you'd want to analyze the code to place this better
        if var_name.isupper():
            # Looks like a constant
            declaration = f"{var_name} = None  # TODO: Set appropriate constant value\n"
        else:
            # Regular variable
            declaration = f"{var_name} = None  # TODO: Initialize this variable properly\n"
            
        # Add after imports but before other code
        import_end = 0
        for match in re.finditer(r'^import|^from', code, re.MULTILINE):
            line_end = code.find('\n', match.start())
            if line_end > import_end:
                import_end = line_end
                
        if import_end > 0:
            return code[:import_end+1] + '\n' + declaration + code[import_end+1:]
        else:
            return declaration + code
            
    def _fix_missing_import(self, code: str, match: re.Match) -> str:
        """Fix missing import by adding import statement.
        
        Args:
            code: Original code
            match: Regex match containing the module name
            
        Returns:
            Fixed code
        """
        module_name = match.group(1)
        
        # Add import at the beginning
        import_stmt = f"import {module_name}  # TODO: Verify this import\n"
        
        # Add to beginning of file
        return import_stmt + code
        
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues.
        
        Args:
            code: Original code
            
        Returns:
            Fixed code
        """
        # This is a simplified implementation
        # In a real system, you'd want to analyze the code structure
        
        lines = code.split('\n')
        fixed_lines = []
        
        # Convert tabs to spaces
        for line in lines:
            fixed_line = line.replace('\t', '    ')
            fixed_lines.append(fixed_line)
            
        return '\n'.join(fixed_lines)
        
    def learn(self, code: str, language: str = "python") -> LearningResult:
        """Improve code through iterative execution and error fixing.
        
        Args:
            code: Original code to improve
            language: Programming language of the code
            
        Returns:
            LearningResult object with improvement details
        """
        current_code = code
        execution_results = []
        errors_fixed = []
        
        for attempt in range(self.max_attempts):
            logger.info(f"Learning attempt {attempt+1}/{self.max_attempts}")
            
            # Execute current code
            result = self.kernel.execute(current_code, language)
            execution_results.append(result)
            
            # If successful, we're done
            if result.success:
                logger.info("Execution successful")
                break
                
            # Try to fix errors
            original_code = current_code
            error_message = result.error
            
            for pattern in self.error_patterns:
                match = pattern.match(error_message)
                if match:
                    logger.info(f"Found error: {pattern.description}")
                    current_code = pattern.apply_fix(current_code, match)
                    errors_fixed.append(pattern.description)
                    
                    # If code was modified, log the change
                    if current_code != original_code:
                        logger.info(f"Applied fix for: {pattern.description}")
                        break
            
            # If no patterns matched or code wasn't modified, we can't improve further
            if current_code == original_code:
                logger.warning("No improvements found for current errors")
                break
                
        # Calculate improvement score
        if not execution_results:
            improvement_score = 0.0
        elif execution_results[-1].success:
            # Successfully fixed all errors
            improvement_score = 1.0
        else:
            # Partial improvement
            improvement_score = len(errors_fixed) / (len(errors_fixed) + 1)
            
        return LearningResult(
            original_code=code,
            improved_code=current_code,
            execution_results=execution_results,
            errors_fixed=errors_fixed,
            improvement_score=improvement_score,
        )
