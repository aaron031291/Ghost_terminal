"""
Grace Auto-Updater

This module provides Grace with the ability to safely evolve her own codebase.
It ensures that new logic—whether uploaded externally or generated internally—is
verified, compared, and optionally integrated into her live system without
corrupting her cognition or structure.

Created: May 2025
"""

import os
import sys
import json
import hashlib
import difflib
import importlib
import inspect
import datetime
import shutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Grace system imports
from Grace.core import system_paths
from Grace.memory import memory_manager
from Grace.diagnostics import system_monitor
from Grace.ingestion import code_validator
from Grace.logs import logger


class CodeFingerprint:
    """Generates and compares fingerprints of code modules."""
    
    def __init__(self, code_content: str, module_path: Optional[str] = None):
        self.content = code_content
        self.module_path = module_path
        self.hash = self._generate_hash()
        self.structure = self._analyze_structure()
        
    def _generate_hash(self) -> str:
        """Generate a hash of the code content."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
    
    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the code (functions, classes, imports)."""
        structure = {
            'imports': self._extract_imports(),
            'functions': self._extract_functions(),
            'classes': self._extract_classes(),
            'line_count': len(self.content.splitlines()),
            'complexity': self._calculate_complexity()
        }
        return structure
    
    def _extract_imports(self) -> List[str]:
        """Extract import statements from the code."""
        # Simple regex-based extraction for demonstration
        import re
        imports = []
        for line in self.content.splitlines():
            if re.match(r'^import\s+|^from\s+.*\s+import', line.strip()):
                imports.append(line.strip())
        return imports
    
    def _extract_functions(self) -> List[str]:
        """Extract function names from the code."""
        import re
        functions = []
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for match in re.finditer(pattern, self.content):
            functions.append(match.group(1))
        return functions
    
    def _extract_classes(self) -> List[str]:
        """Extract class names from the code."""
        import re
        classes = []
        pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'
        for match in re.finditer(pattern, self.content):
            classes.append(match.group(1))
        return classes
    
    def _calculate_complexity(self) -> int:
        """Calculate a simple complexity score for the code."""
        # This is a simplified metric - in production, use a more sophisticated approach
        complexity = 0
        complexity += len(self._extract_functions()) * 2
        complexity += len(self._extract_classes()) * 3
        complexity += len(self._extract_imports())
        complexity += self.content.count('if ') + self.content.count('else:') + self.content.count('elif ')
        complexity += self.content.count('for ') + self.content.count('while ')
        complexity += self.content.count('try:') + self.content.count('except')
        return complexity
    
    def compare_with(self, other: 'CodeFingerprint') -> Dict[str, Any]:
        """Compare this fingerprint with another one."""
        similarity = {
            'hash_match': self.hash == other.hash,
            'content_similarity': self._calculate_content_similarity(other.content),
            'structure_similarity': self._compare_structure(other.structure),
            'function_overlap': self._calculate_overlap(self.structure['functions'], other.structure['functions']),
            'class_overlap': self._calculate_overlap(self.structure['classes'], other.structure['classes']),
            'import_overlap': self._calculate_overlap(self.structure['imports'], other.structure['imports']),
        }
        return similarity
    
    def _calculate_content_similarity(self, other_content: str) -> float:
        """Calculate the similarity between two code contents."""
        matcher = difflib.SequenceMatcher(None, self.content, other_content)
        return matcher.ratio()
    
    def _compare_structure(self, other_structure: Dict[str, Any]) -> float:
        """Compare the structure similarity."""
        # Simple comparison for demonstration
        similarities = []
        
        # Compare line counts
        line_ratio = min(self.structure['line_count'], other_structure['line_count']) / max(self.structure['line_count'], other_structure['line_count']) if max(self.structure['line_count'], other_structure['line_count']) > 0 else 1.0
        similarities.append(line_ratio)
        
        # Compare complexity
        complexity_ratio = min(self.structure['complexity'], other_structure['complexity']) / max(self.structure['complexity'], other_structure['complexity']) if max(self.structure['complexity'], other_structure['complexity']) > 0 else 1.0
        similarities.append(complexity_ratio)
        
        # Average the similarities
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate the overlap between two lists."""
        if not list1 or not list2:
            return 0.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class ConflictDetector:
    """Detects conflicts between new code and existing modules."""
    
    def __init__(self, existing_module_path: str, new_code_content: str):
        self.existing_module_path = existing_module_path
        self.new_code_content = new_code_content
        self.existing_fingerprint = self._get_existing_fingerprint()
        self.new_fingerprint = CodeFingerprint(new_code_content)
        
    def _get_existing_fingerprint(self) -> CodeFingerprint:
        """Get the fingerprint of the existing module."""
        try:
            with open(self.existing_module_path, 'r') as f:
                content = f.read()
            return CodeFingerprint(content, self.existing_module_path)
        except FileNotFoundError:
            # If the file doesn't exist, return an empty fingerprint
            return CodeFingerprint("", self.existing_module_path)
    
    def detect_conflicts(self) -> Dict[str, Any]:
        """Detect conflicts between the new code and existing module."""
        comparison = self.existing_fingerprint.compare_with(self.new_fingerprint)
        
        conflicts = {
            'has_conflicts': False,
            'details': [],
            'severity': 'none',
            'comparison': comparison
        }
        
        # Check for function name conflicts
        function_conflicts = self._detect_function_conflicts()
        if function_conflicts:
            conflicts['has_conflicts'] = True
            conflicts['details'].extend(function_conflicts)
        
        # Check for class name conflicts
        class_conflicts = self._detect_class_conflicts()
        if class_conflicts:
            conflicts['has_conflicts'] = True
            conflicts['details'].extend(class_conflicts)
        
        # Check for dangerous imports
        dangerous_imports = self._detect_dangerous_imports()
        if dangerous_imports:
            conflicts['has_conflicts'] = True
            conflicts['details'].extend(dangerous_imports)
            conflicts['severity'] = 'high'
        
        # Set the overall severity
        if conflicts['has_conflicts'] and conflicts['severity'] == 'none':
            conflicts['severity'] = 'medium' if len(conflicts['details']) > 3 else 'low'
        
        return conflicts
    
    def _detect_function_conflicts(self) -> List[Dict[str, str]]:
        """Detect conflicts in function definitions."""
        conflicts = []
        existing_functions = set(self.existing_fingerprint.structure['functions'])
        new_functions = set(self.new_fingerprint.structure['functions'])
        
        # Find overlapping function names
        overlapping = existing_functions.intersection(new_functions)
        
        for func_name in overlapping:
            conflicts.append({
                'type': 'function_conflict',
                'name': func_name,
                'message': f"Function '{func_name}' already exists in the module"
            })
        
        return conflicts
    
    def _detect_class_conflicts(self) -> List[Dict[str, str]]:
        """Detect conflicts in class definitions."""
        conflicts = []
        existing_classes = set(self.existing_fingerprint.structure['classes'])
        new_classes = set(self.new_fingerprint.structure['classes'])
        
        # Find overlapping class names
        overlapping = existing_classes.intersection(new_classes)
        
        for class_name in overlapping:
            conflicts.append({
                'type': 'class_conflict',
                'name': class_name,
                'message': f"Class '{class_name}' already exists in the module"
            })
        
        return conflicts
    
    def _detect_dangerous_imports(self) -> List[Dict[str, str]]:
        """Detect potentially dangerous imports in the new code."""
        conflicts = []
        dangerous_modules = [
            'os.system', 'subprocess', 'eval', 'exec', 
            'pickle', 'marshal', 'socket', 'shutil.rmtree',
            'os.remove', 'os.unlink', 'sys.exit'
        ]
        
        for imp in self.new_fingerprint.structure['imports']:
            for dangerous in dangerous_modules:
                if dangerous in imp:
                    conflicts.append({
                        'type': 'dangerous_import',
                        'name': imp,
                        'message': f"Potentially dangerous import detected: '{imp}'"
                    })
                    break
        
        return conflicts


class VersionManager:
    """Manages versions of code modules."""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(system_paths.GRACE_ROOT, 'versions')
        os.makedirs(self.base_path, exist_ok=True)
    
    def backup_module(self, module_path: str) -> str:
        """Create a backup of a module before updating it."""
        if not os.path.exists(module_path):
            return None
        
        # Create a timestamped version of the file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        module_name = os.path.basename(module_path)
        module_dir = os.path.dirname(module_path)
        
        # Create a directory structure mirroring the original
        relative_dir = os.path.relpath(module_dir, system_paths.GRACE_ROOT)
        backup_dir = os.path.join(self.base_path, relative_dir)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create the backup file path
        backup_path = os.path.join(
            backup_dir, 
            f"{os.path.splitext(module_name)[0]}_{timestamp}{os.path.splitext(module_name)[1]}"
        )
        
        # Copy the file
        shutil.copy2(module_path, backup_path)
        
        return backup_path
    
    def get_version_history(self, module_path: str) -> List[str]:
        """Get the version history of a module."""
        module_name = os.path.basename(module_path)
        module_dir = os.path.dirname(module_path)
        relative_dir = os.path.relpath(module_dir, system_paths.GRACE_ROOT)
        backup_dir = os.path.join(self.base_path, relative_dir)
        
        if not os.path.exists(backup_dir):
            return []
        
        base_name = os.path.splitext(module_name)[0]
        ext = os.path.splitext(module_name)[1]
        
        # Find all versions of this module
        versions = []
        for file in os.listdir(backup_dir):
            if file.startswith(base_name + '_') and file.endswith(ext):
                versions.append(os.path.join(backup_dir, file))
        
        # Sort by timestamp (newest first)
        versions.sort(reverse=True)
        
        return versions
    
    def restore_version(self, backup_path: str, target_path: str) -> bool:
        """Restore a previous version of a module."""
        if not os.path.exists(backup_path):
            return False
        
        # Backup the current version first
        self.backup_module(target_path)
        
        # Copy the backup to the target
        shutil.copy2(backup_path, target_path)
        
        return True


class CodeScorer:
    """Scores code based on trust and utility metrics."""
    
    def __init__(self, code_content: str, module_path: str = None):
        self.code_content = code_content
        self.module_path = module_path
        
    def calculate_trust_score(self) -> Dict[str, Any]:
        """Calculate a trust score for the code."""
        # Initialize with maximum score
        score = 100
        issues = []
        
        # Check syntax validity
        syntax_valid, syntax_error = self._check_syntax()
        if not syntax_valid:
            score -= 50
            issues.append({
                'type': 'syntax_error',
                'message': str(syntax_error),
                'severity': 'high'
            })
        
        # Check for dangerous patterns
        dangerous_patterns = self._check_dangerous_patterns()
        for pattern in dangerous_patterns:
            score -= pattern['penalty']
            issues.append({
                'type': 'dangerous_pattern',
                'message': pattern['message'],
                'severity': pattern['severity']
            })
        
        # Check import safety
        import_issues = self._check_import_safety()
        for issue in import_issues:
            score -= issue['penalty']
            issues.append({
                'type': 'import_issue',
                'message': issue['message'],
                'severity': issue['severity']
            })
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'issues': issues,
            'verdict': 'trusted' if score >= 70 else 'suspicious' if score >= 40 else 'untrusted'
        }
    
    def calculate_utility_score(self) -> Dict[str, Any]:
        """Calculate a utility score for the code."""
        # Initialize with neutral score
        score = 50
        insights = []
        
        # Check code complexity
        complexity = self._analyze_complexity()
        if complexity['score'] > 0:
            score += complexity['score']
            insights.append({
                            'type': 'complexity',
                'message': complexity['message'],
                'impact': 'positive' if complexity['score'] > 0 else 'negative'
            })
        
        # Check documentation quality
        doc_quality = self._analyze_documentation()
        if doc_quality['score'] > 0:
            score += doc_quality['score']
            insights.append({
                'type': 'documentation',
                'message': doc_quality['message'],
                'impact': 'positive' if doc_quality['score'] > 0 else 'negative'
            })
        
        # Check test coverage
        test_coverage = self._analyze_test_coverage()
        if test_coverage['score'] > 0:
            score += test_coverage['score']
            insights.append({
                'type': 'test_coverage',
                'message': test_coverage['message'],
                'impact': 'positive' if test_coverage['score'] > 0 else 'negative'
            })
        
        # Check code style
        code_style = self._analyze_code_style()
        if code_style['score'] > 0:
            score += code_style['score']
            insights.append({
                'type': 'code_style',
                'message': code_style['message'],
                'impact': 'positive' if code_style['score'] > 0 else 'negative'
            })
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'insights': insights,
            'verdict': 'high_value' if score >= 75 else 'moderate_value' if score >= 50 else 'low_value'
        }
    
    def _check_syntax(self) -> Tuple[bool, Optional[str]]:
        """Check if the code has valid syntax."""
        try:
            compile(self.code_content, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def _check_dangerous_patterns(self) -> List[Dict[str, Any]]:
        """Check for dangerous code patterns."""
        dangerous_patterns = []
        
        # Check for eval/exec usage
        if 'eval(' in self.code_content or 'exec(' in self.code_content:
            dangerous_patterns.append({
                'message': 'Code contains eval() or exec() which can execute arbitrary code',
                'penalty': 30,
                'severity': 'high'
            })
        
        # Check for file operations
        if 'open(' in self.code_content and ('w' in self.code_content or 'a' in self.code_content):
            dangerous_patterns.append({
                'message': 'Code contains file write operations',
                'penalty': 15,
                'severity': 'medium'
            })
        
        # Check for system commands
        if 'os.system(' in self.code_content or 'subprocess' in self.code_content:
            dangerous_patterns.append({
                'message': 'Code contains system command execution',
                'penalty': 25,
                'severity': 'high'
            })
        
        # Check for network operations
        if 'socket' in self.code_content or 'urllib' in self.code_content or 'requests' in self.code_content:
            dangerous_patterns.append({
                'message': 'Code contains network operations',
                'penalty': 10,
                'severity': 'medium'
            })
        
        return dangerous_patterns
    
    def _check_import_safety(self) -> List[Dict[str, Any]]:
        """Check for potentially unsafe imports."""
        issues = []
        dangerous_imports = {
            'os': {'penalty': 5, 'severity': 'low'},
            'sys': {'penalty': 5, 'severity': 'low'},
            'subprocess': {'penalty': 20, 'severity': 'high'},
            'pickle': {'penalty': 15, 'severity': 'medium'},
            'marshal': {'penalty': 15, 'severity': 'medium'},
            'socket': {'penalty': 10, 'severity': 'medium'},
            'shutil': {'penalty': 10, 'severity': 'medium'},
        }
        
        import re
        for line in self.code_content.splitlines():
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                for module, info in dangerous_imports.items():
                    pattern = r'import\s+' + module + r'\b|from\s+' + module + r'\s+import'
                    if re.search(pattern, line):
                        issues.append({
                            'message': f"Code imports potentially dangerous module: {module}",
                            'penalty': info['penalty'],
                            'severity': info['severity']
                        })
        
        return issues
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity."""
        # Count lines, functions, classes
        lines = len(self.code_content.splitlines())
        
        import re
        functions = len(re.findall(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', self.code_content))
        classes = len(re.findall(r'class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]', self.code_content))
        
        # Count control structures
        control_structures = len(re.findall(r'\b(if|else|elif|for|while|try|except|with)\b', self.code_content))
        
        # Calculate complexity score
        if lines > 500:
            return {
                'score': -10,
                'message': 'Code is excessively long (over 500 lines)'
            }
        elif lines > 200 and functions < 3:
            return {
                'score': -5,
                'message': 'Long code with few functions suggests poor modularization'
            }
        elif functions > 10 and classes > 5:
            return {
                'score': 10,
                'message': 'Well-structured code with good modularization'
            }
        elif control_structures > 50:
            return {
                'score': -5,
                'message': 'Code has excessive control structures'
            }
        
        return {
            'score': 5,
            'message': 'Code has reasonable complexity'
        }
    
    def _analyze_documentation(self) -> Dict[str, Any]:
        """Analyze documentation quality."""
        import re
        
        # Check for module docstring
        module_docstring = re.search(r'^""".*?"""', self.code_content, re.DOTALL)
        
        # Check for function docstrings
        function_docstrings = re.findall(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\):\s*""".*?"""', self.code_content, re.DOTALL)
        
        # Check for class docstrings
        class_docstrings = re.findall(r'class\s+[a-zA-Z_][a-zA-Z0-9_]*.*?:\s*""".*?"""', self.code_content, re.DOTALL)
        
        # Count functions and classes
        functions = len(re.findall(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', self.code_content))
        classes = len(re.findall(r'class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]', self.code_content))
        
        # Calculate documentation coverage
        doc_coverage = 0
        if module_docstring:
            doc_coverage += 1
        
        if functions > 0:
            doc_coverage += len(function_docstrings) / functions
        
        if classes > 0:
            doc_coverage += len(class_docstrings) / classes
        
        # Normalize to a score between -10 and 15
        if functions + classes == 0:
            # Only module docstring matters
            score = 5 if module_docstring else -5
            message = 'Module has a docstring' if module_docstring else 'Module lacks a docstring'
        else:
            # Calculate average coverage
            avg_coverage = doc_coverage / (2 if classes > 0 else 1)
            
            if avg_coverage > 0.8:
                score = 15
                message = 'Excellent documentation coverage'
            elif avg_coverage > 0.5:
                score = 10
                message = 'Good documentation coverage'
            elif avg_coverage > 0.2:
                score = 5
                message = 'Moderate documentation coverage'
            else:
                score = -10
                message = 'Poor documentation coverage'
        
        return {
            'score': score,
            'message': message
        }
    
    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage."""
        # Check if this is a test file
        is_test_file = 'test' in (self.module_path or '').lower() or 'test' in self.code_content.lower()
        
        if is_test_file:
            # Count assertions
            import re
            assertions = len(re.findall(r'\bassert\w*\(', self.code_content))
            
            if assertions > 20:
                return {
                    'score': 15,
                    'message': 'Comprehensive test suite with many assertions'
                }
            elif assertions > 10:
                return {
                    'score': 10,
                    'message': 'Good test coverage with multiple assertions'
                }
            elif assertions > 0:
                return {
                    'score': 5,
                    'message': 'Basic test coverage present'
                }
        
        # Check if the code has accompanying tests
        # This would require more context than we have here
        
        return {
            'score': 0,
            'message': 'Unable to determine test coverage'
        }
    
    def _analyze_code_style(self) -> Dict[str, Any]:
        """Analyze code style."""
        issues = []
        
        # Check line length
        long_lines = 0
        for line in self.code_content.splitlines():
            if len(line) > 100:
                long_lines += 1
        
        if long_lines > 10:
            issues.append('Many lines exceed recommended length')
        
        # Check variable naming
        import re
        snake_case = len(re.findall(r'\b[a-z][a-z0-9_]*\b', self.code_content))
        camel_case = len(re.findall(r'\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b', self.code_content))
        
        if snake_case < camel_case:
            issues.append('Inconsistent variable naming style (prefers camelCase over snake_case)')
        
        # Check indentation
        inconsistent_indent = False
        spaces_indent = 0
        tabs_indent = 0
        
        for line in self.code_content.splitlines():
            if line.startswith(' '):
                spaces_indent += 1
            elif line.startswith('\t'):
                tabs_indent += 1
        
        if spaces_indent > 0 and tabs_indent > 0:
            inconsistent_indent = True
            issues.append('Mixed use of tabs and spaces for indentation')
        
        # Calculate style score
        if len(issues) == 0:
            return {
                'score': 10,
                'message': 'Code follows good style practices'
            }
        elif len(issues) == 1:
            return {
                'score': 5,
                'message': f'Generally good style with one issue: {issues[0]}'
            }
        else:
            return {
                'score': -5,
                'message': f'Multiple style issues: {", ".join(issues)}'
            }


class AutoUpdater:
    """Main class for Grace's auto-update functionality."""
    
    def __init__(self, grace_root: str = None):
        self.grace_root = grace_root or system_paths.GRACE_ROOT
        self.version_manager = VersionManager(os.path.join(self.grace_root, 'versions'))
        self.log_path = os.path.join(self.grace_root, 'logs', 'auto_updates.json')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)
    
    def process_update(self, code_content: str, target_path: str, auto_integrate: bool = False, 
                       trust_threshold: int = 70, utility_threshold: int = 50,
                       contributor: str = "unknown") -> Dict[str, Any]:
        """
        Process a code update request.
        
        Args:
            code_content: The new code content
            target_path: The path where the code should be installed
            auto_integrate: Whether to automatically integrate the code if it passes checks
            trust_threshold: Minimum trust score required for auto-integration
            utility_threshold: Minimum utility score required for auto-integration
            contributor: The name/id of the contributor
            
        Returns:
            A dictionary with the update results
        """
        # Normalize the target path
        if not os.path.isabs(target_path):
            target_path = os.path.join(self.grace_root, target_path)
        
        # Create result structure
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'target_path': target_path,
            'contributor': contributor,
            'auto_integrate': auto_integrate,
            'status': 'pending',
            'scores': {},
            'conflicts': {},
            'backup_path': None,
            'message': '',
        }
        
        try:
            # Score the code
            scorer = CodeScorer(code_content, target_path)
            trust_result = scorer.calculate_trust_score()
            utility_result = scorer.calculate_utility_score()
            
            result['scores'] = {
                'trust': trust_result,
                'utility': utility_result
            }
            
            # Check for conflicts
            if os.path.exists(target_path):
                conflict_detector = ConflictDetector(target_path, code_content)
                conflicts = conflict_detector.detect_conflicts()
                result['conflicts'] = conflicts
                
                if conflicts['has_conflicts'] and conflicts['severity'] == 'high':
                    result['status'] = 'rejected'
                    result['message'] = 'Update rejected due to high severity conflicts'
                    self._log_update(result)
                    return result
            
            # Determine if we should integrate
            should_integrate = (
                auto_integrate and 
                trust_result['score'] >= trust_threshold and
                utility_result['score'] >= utility_threshold and
                (not result.get('conflicts', {}).get('has_conflicts', False) or 
                 result.get('conflicts', {}).get('severity', 'none') == 'low')
            )
            
            if should_integrate:
                # Backup the existing file
                if os.path.exists(target_path):
                                       backup_path = self.version_manager.backup_module(target_path)
                    result['backup_path'] = backup_path
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Write the new code
                with open(target_path, 'w') as f:
                    f.write(code_content)
                
                result['status'] = 'integrated'
                result['message'] = 'Update successfully integrated'
                
                # Attempt to reload the module if it's already imported
                self._try_reload_module(target_path)
            else:
                # Store in staging area
                staging_path = self._store_in_staging(code_content, target_path, contributor)
                
                result['status'] = 'staged'
                result['staging_path'] = staging_path
                result['message'] = 'Update staged for manual review'
        
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'Error processing update: {str(e)}'
            result['error'] = str(e)
        
        # Log the update
        self._log_update(result)
        
        # Notify the memory system
        self._notify_memory_system(result)
        
        return result
    
    def _store_in_staging(self, code_content: str, target_path: str, contributor: str) -> str:
        """Store code in staging area for manual review."""
        # Create staging directory if it doesn't exist
        staging_dir = os.path.join(self.grace_root, 'staging')
        os.makedirs(staging_dir, exist_ok=True)
        
        # Create a unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        rel_path = os.path.relpath(target_path, self.grace_root)
        safe_path = rel_path.replace('/', '_').replace('\\', '_')
        staging_filename = f"{safe_path}_{contributor}_{timestamp}.py"
        staging_path = os.path.join(staging_dir, staging_filename)
        
        # Write the code to the staging file
        with open(staging_path, 'w') as f:
            f.write(f"# Staged update for: {rel_path}\n")
            f.write(f"# Contributor: {contributor}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Target path: {target_path}\n\n")
            f.write(code_content)
        
        return staging_path
    
    def _log_update(self, result: Dict[str, Any]) -> None:
        """Log the update to the auto_updates.json file."""
        try:
            # Read existing logs
            with open(self.log_path, 'r') as f:
                logs = json.load(f)
            
            # Add new log
            logs.append(result)
            
            # Write back to file
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log update: {str(e)}")
    
    def _notify_memory_system(self, result: Dict[str, Any]) -> None:
        """Notify Grace's memory system about the update."""
        try:
            # Create a memory entry
            memory_entry = {
                'type': 'code_update',
                'timestamp': result['timestamp'],
                'target_path': result['target_path'],
                'contributor': result['contributor'],
                'status': result['status'],
                'trust_score': result['scores'].get('trust', {}).get('score', 0),
                'utility_score': result['scores'].get('utility', {}).get('score', 0),
                'message': result['message']
            }
            
            # Add to memory
            memory_manager.add_system_memory(memory_entry)
        except Exception as e:
            logger.error(f"Failed to notify memory system: {str(e)}")
    
    def _try_reload_module(self, module_path: str) -> bool:
        """Try to reload a module if it's already imported."""
        try:
            # Convert file path to module name
            rel_path = os.path.relpath(module_path, self.grace_root)
            module_name = os.path.splitext(rel_path)[0].replace('/', '.').replace('\\', '.')
            
            # Check if the module is already imported
            if module_name in sys.modules:
                # Reload the module
                importlib.reload(sys.modules[module_name])
                logger.info(f"Successfully reloaded module: {module_name}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to reload module {module_path}: {str(e)}")
            return False
    
    def list_staged_updates(self) -> List[Dict[str, Any]]:
        """List all updates in the staging area."""
        staging_dir = os.path.join(self.grace_root, 'staging')
        if not os.path.exists(staging_dir):
            return []
        
        staged_updates = []
        for filename in os.listdir(staging_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(staging_dir, filename)
                
                # Extract metadata from the file
                metadata = self._extract_staging_metadata(file_path)
                
                staged_updates.append({
                    'filename': filename,
                    'path': file_path,
                    'metadata': metadata,
                    'created': datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                })
        
        # Sort by creation time (newest first)
        staged_updates.sort(key=lambda x: x['created'], reverse=True)
        
        return staged_updates
    
    def _extract_staging_metadata(self, file_path: str) -> Dict[str, str]:
        """Extract metadata from a staged update file."""
        metadata = {}
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i > 5:  # Only check the first few lines
                        break
                    if line.startswith('# '):
                        parts = line[2:].strip().split(': ', 1)
                        if len(parts) == 2:
                            metadata[parts[0].lower()] = parts[1]
        except Exception:
            pass
        
        return metadata
    
    def approve_staged_update(self, staging_path: str) -> Dict[str, Any]:
        """Approve and integrate a staged update."""
        if not os.path.exists(staging_path):
            return {
                'status': 'error',
                'message': f'Staged update not found: {staging_path}'
            }
        
        try:
            # Extract metadata
            metadata = self._extract_staging_metadata(staging_path)
            target_path = metadata.get('target path')
            contributor = metadata.get('contributor', 'unknown')
            
            if not target_path:
                return {
                    'status': 'error',
                    'message': 'Target path not found in staged update metadata'
                }
            
            # Read the code content (skip metadata lines)
            with open(staging_path, 'r') as f:
                lines = f.readlines()
            
            # Find where the actual code starts (after metadata)
            code_start = 0
            for i, line in enumerate(lines):
                if not line.startswith('# '):
                    code_start = i
                    break
            
            code_content = ''.join(lines[code_start:])
            
            # Process the update with auto_integrate=True
            result = self.process_update(
                code_content=code_content,
                target_path=target_path,
                auto_integrate=True,
                trust_threshold=0,  # Skip trust check since we're manually approving
                utility_threshold=0,  # Skip utility check since we're manually approving
                contributor=contributor
            )
            
            # Delete the staged file
            os.remove(staging_path)
            
            return result
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error approving staged update: {str(e)}'
            }
    
    def reject_staged_update(self, staging_path: str, reason: str = None) -> Dict[str, Any]:
        """Reject a staged update."""
        if not os.path.exists(staging_path):
            return {
                'status': 'error',
                'message': f'Staged update not found: {staging_path}'
            }
        
        try:
            # Extract metadata
            metadata = self._extract_staging_metadata(staging_path)
            target_path = metadata.get('target path')
            contributor = metadata.get('contributor', 'unknown')
            
            # Create rejection result
            result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'target_path': target_path,
                'contributor': contributor,
                'status': 'rejected',
                'message': f'Update rejected: {reason or "No reason provided"}',
                'staging_path': staging_path
            }
            
            # Log the rejection
            self._log_update(result)
            
            # Notify memory system
            self._notify_memory_system(result)
            
            # Delete the staged file
            os.remove(staging_path)
            
            return result
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error rejecting staged update: {str(e)}'
            }
    
    def get_update_history(self, module_path: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get update history for a specific module or all modules."""
        try:
            # Read logs
            with open(self.log_path, 'r') as f:
                logs = json.load(f)
            
            # Filter by module path if provided
            if module_path:
                if not os.path.isabs(module_path):
                    module_path = os.path.join(self.grace_root, module_path)
                
                logs = [log for log in logs if log.get('target_path') == module_path]
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Limit the number of results
            return logs[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get update history: {str(e)}")
            return []
    
    def rollback_update(self, module_path: str, version_path: str = None) -> Dict[str, Any]:
        """
        Rollback a module to a previous version.
        
        Args:
            module_path: The path of the module to rollback
            version_path: The specific version to rollback to. If None, rolls back to the most recent version.
            
        Returns:
            A dictionary with the rollback results
        """
        if not os.path.isabs(module_path):
            module_path = os.path.join(self.grace_root, module_path)
        
        # Get version history
        versions = self.version_manager.get_version_history(module_path)
        
        if not versions:
            return {
                'status': 'error',
                'message': f'No previous versions found for {module_path}'
            }
        
        try:
            # Determine which version to restore
            if version_path:
                if not os.path.exists(version_path):
                    return {
                        'status': 'error',
                        'message': f'Specified version not found: {version_path}'
                    }
                restore_path = version_path
            else:
                # Use the most recent version
                restore_path = versions[0]
            
            # Restore the version
            success = self.version_manager.restore_version(restore_path, module_path)
            
            if success:
                # Try to reload the module
                self._try_reload_module(module_path)
                
                result = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'target_path': module_path,
                    'contributor': 'system',
                    'status': 'rollback',
                    'message': f'Successfully rolled back to version: {os.path.basename(restore_path)}',
                    'version_path': restore_path
                }
                
                # Log the rollback
                self._log_update(result)
                
                # Notify memory system
                self._notify_memory_system(result)
                
                return result
            else:
                return {
                    'status': 'error',
                    'message': f'Failed to restore version: {restore_path}'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during rollback: {str(e)}'
            }


# Convenience functions for external use

def update_module(code_content: str, target_path: str, auto_integrate: bool = False, 
                 trust_threshold: int = 70, utility_threshold: int = 50,
                 contributor: str = "unknown") -> Dict[str, Any]:
    """
    Update a Grace module with new code.
    
    Args:
        code_content: The new code content
        target_path: The path where the code should be installed
        auto_integrate: Whether to automatically integrate the code if it passes checks
        trust_threshold: Minimum trust score required for auto-integration
        utility_threshold: Minimum utility score required for auto-integration
        contributor: The name/id of the contributor
        
    Returns:
        A dictionary with the update results
    """
    updater = AutoUpdater()
    return updater.process_update(
        code_content=code_content,
        target_path=target_path,
        auto_integrate=auto_integrate,
        trust_threshold=trust_threshold,
        utility_threshold=utility_threshold,
        contributor=contributor
    )


def list_staged_updates() -> List[Dict[str, Any]]:
    """List all updates in the staging area."""
    updater = AutoUpdater()
    return updater.list_staged_updates()


def approve_update(staging_path: str) -> Dict[str, Any]:
    """Approve and integrate a staged update."""
    updater = AutoUpdater()
    return updater.approve_staged_update(staging_path)


def reject_update(staging_path: str, reason: str = None) -> Dict[str, Any]:
    """Reject a staged update."""
    updater = AutoUpdater()
    return updater.reject_staged_update(staging_path, reason)


def get_update_history(module_path: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get update history for a specific module or all modules."""
    updater = AutoUpdater()
    return updater.get_update_history(module_path, limit)


def rollback_module(module_path: str, version_path: str = None) -> Dict[str, Any]:
    """Rollback a module to a previous version."""
    updater = AutoUpdater()
    return updater.rollback_update(module_path, version_path)


if __name__ == "__main__":
    # Command-line interface for the updater
    import argparse
    
        parser = argparse.ArgumentParser(description="Grace Auto-Updater")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update a module")
    update_parser.add_argument("--file", required=True, help="File containing the new code")
    update_parser.add_argument("--target", required=True, help="Target path for the update")
    update_parser.add_argument("--auto", action="store_true", help="Auto-integrate if checks pass")
    update_parser.add_argument("--trust", type=int, default=70, help="Trust threshold (0-100)")
    update_parser.add_argument("--utility", type=int, default=50, help="Utility threshold (0-100)")
    update_parser.add_argument("--contributor", default="cli_user", help="Contributor name/id")
    
    # List staged updates command
    subparsers.add_parser("list-staged", help="List all staged updates")
    
    # Approve update command
    approve_parser = subparsers.add_parser("approve", help="Approve a staged update")
    approve_parser.add_argument("--path", required=True, help="Path to the staged update file")
    
    # Reject update command
    reject_parser = subparsers.add_parser("reject", help="Reject a staged update")
    reject_parser.add_argument("--path", required=True, help="Path to the staged update file")
    reject_parser.add_argument("--reason", help="Reason for rejection")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Get update history")
    history_parser.add_argument("--module", help="Module path (optional)")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum number of entries")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a module")
    rollback_parser.add_argument("--module", required=True, help="Module path to rollback")
    rollback_parser.add_argument("--version", help="Specific version to rollback to (optional)")
    
    args = parser.parse_args()
    
    # Initialize the updater
    updater = AutoUpdater()
    
    # Execute the appropriate command
    if args.command == "update":
        try:
            with open(args.file, 'r') as f:
                code_content = f.read()
            
            result = updater.process_update(
                code_content=code_content,
                target_path=args.target,
                auto_integrate=args.auto,
                trust_threshold=args.trust,
                utility_threshold=args.utility,
                contributor=args.contributor
            )
            
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.command == "list-staged":
        staged = updater.list_staged_updates()
        print(json.dumps(staged, indent=2))
    
    elif args.command == "approve":
        result = updater.approve_staged_update(args.path)
        print(json.dumps(result, indent=2))
    
    elif args.command == "reject":
        result = updater.reject_staged_update(args.path, args.reason)
        print(json.dumps(result, indent=2))
    
    elif args.command == "history":
        history = updater.get_update_history(args.module, args.limit)
        print(json.dumps(history, indent=2))
    
    elif args.command == "rollback":
        result = updater.rollback_update(args.module, args.version)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

 
    
                
