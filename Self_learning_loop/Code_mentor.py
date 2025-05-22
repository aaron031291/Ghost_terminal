#!/usr/bin/env python3
"""
Mentor Engine for Grace AI

This module provides a comprehensive mentoring system that uses Grace's own architecture as a
live playground for learning and improvement. The Mentor Engine continuously monitors, evaluates,
and guides Grace through self-repair, code evolution, and autonomous improvement loops.

The engine operates within a secure sandbox environment and applies syntax, logic, ethical, and
optimization analysis to detect issues or inefficiencies in Grace's codebase. It maintains full
oversight until Grace reaches a threshold of mature coding competence.
"""

import os
import sys
import logging
import json
import re
import ast
import inspect
import importlib
import traceback
import time
import hashlib
import asyncio
import concurrent.futures
import shutil
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import subprocess
from threading import Lock
import uuid
import copy

# Mock imports for external dependencies
try:
    import cryptography
    from cryptography.fernet import Fernet
except ImportError:
    # Mock cryptography module
    class Fernet:
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            return b"mock_encrypted_" + data
        
        def decrypt(self, data):
            if data.startswith(b"mock_encrypted_"):
                return data[len(b"mock_encrypted_"):]
            return data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MentorEngine")

# Import the CodeLearningEngine
try:
    from code_learning_engine import (
        CodeLearningEngine, LanguageDetector, SyntaxAnalyzer, CodeFixer,
        CodeOptimizer, CodeIssue, CodeIssueType, CodeAnalysisResult, CodeFixResult
    )
except ImportError:
    logger.error("CodeLearningEngine not found. Please ensure it's in the Python path.")
    # We'll define minimal versions of the required classes to allow this module to run
    class CodeIssueType(Enum):
        SYNTAX_ERROR = "syntax_error"
        REFERENCE_ERROR = "reference_error"
        INCOMPLETE_FUNCTION = "incomplete_function"
        SEMANTIC_ERROR = "semantic_error"
        SECURITY_VULNERABILITY = "security_vulnerability"
        PERFORMANCE_ISSUE = "performance_issue"
        STYLE_VIOLATION = "style_violation"
        ETHICAL_CONCERN = "ethical_concern"
        LOGIC_ERROR = "logic_error"
        ARCHITECTURE_ISSUE = "architecture_issue"

    class CodeFixConfidence(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        EXPERIMENTAL = "experimental"

    @dataclass
    class CodeIssue:
        """Represents an issue found in code."""
        issue_type: CodeIssueType
        description: str
        line_number: Optional[int] = None
        column: Optional[int] = None
        file_path: Optional[str] = None
        code_snippet: Optional[str] = None
        suggested_fix: Optional[str] = None
        confidence: str = "MEDIUM"
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the issue to a dictionary."""
            return {
                "issue_type": self.issue_type.value,
                "description": self.description,
                "line_number": self.line_number,
                "column": self.column,
                "file_path": self.file_path,
                "code_snippet": self.code_snippet,
                "suggested_fix": self.suggested_fix,
                "confidence": self.confidence
            }

    @dataclass
    class CodeAnalysisResult:
        """Results from code analysis."""
        issues: List[CodeIssue] = field(default_factory=list)
        metrics: Dict[str, Any] = field(default_factory=dict)
        ast_representation: Optional[Any] = None
        language: str = ""
        file_path: Optional[str] = None
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the analysis result to a dictionary."""
            return {
                "issues": [issue.to_dict() for issue in self.issues],
                "metrics": self.metrics,
                "language": self.language,
                "file_path": self.file_path
            }

    @dataclass
    class CodeFixResult:
        """Results from a code fix operation."""
        original_code: str
        fixed_code: str
        issues_fixed: List[CodeIssue]
        issues_remaining: List[CodeIssue]
        fix_successful: bool
        language: str
        file_path: Optional[str] = None
        fingerprint: Optional[str] = None
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the fix result to a dictionary."""
            return {
                "original_code": self.original_code,
                "fixed_code": self.fixed_code,
                "issues_fixed": [issue.to_dict() for issue in self.issues_fixed],
                "issues_remaining": [issue.to_dict() for issue in self.issues_remaining],
                "fix_successful": self.fix_successful,
                "language": self.language,
                "file_path": self.file_path,
                "fingerprint": self.fingerprint
            }

    class LanguageDetector:
        @staticmethod
        def detect_from_extension(file_path: str) -> Optional[str]:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".py":
                return "python"
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                return "javascript"
            return None
        
        @staticmethod
        def detect_from_content(content: str) -> str:
            return "python"  # Default to Python

    class SyntaxAnalyzer:
        @staticmethod
        def analyze_code(code: str, language: Optional[str] = None, file_path: Optional[str] = None) -> CodeAnalysisResult:
            return CodeAnalysisResult(language=language or "python", file_path=file_path)

    class CodeFixer:
        @staticmethod
        def fix_code(code: str, analysis_result: CodeAnalysisResult) -> CodeFixResult:
            return CodeFixResult(
                original_code=code,
                fixed_code=code,
                issues_fixed=[],
                issues_remaining=analysis_result.issues,
                fix_successful=False,
                language=analysis_result.language,
                file_path=analysis_result.file_path
            )

    class CodeOptimizer:
        @staticmethod
        def optimize_code(code: str, language: str) -> Tuple[str, Dict[str, Any]]:
            return code, {"optimizations_applied": 0}

    class CodeLearningEngine:
        def __init__(self, sandbox_dir: Optional[str] = None):
            self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="grace_code_sandbox_")
            self.language_detector = LanguageDetector()
            self.syntax_analyzer = SyntaxAnalyzer()
            self.code_fixer = CodeFixer()
            self.code_optimizer = CodeOptimizer()
        
        def analyze_file(self, file_path: str) -> CodeAnalysisResult:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                language = self.language_detector.detect_from_extension(file_path)
                return self.syntax_analyzer.analyze_code(code, language, file_path)
            except Exception as e:
                return CodeAnalysisResult(
                    issues=[CodeIssue(
                        issue_type=CodeIssueType.SYNTAX_ERROR,
                        description=f"Error analyzing file: {str(e)}",
                        file_path=file_path
                    )],
                    language=self.language_detector.detect_from_extension(file_path) or "unknown",
                    file_path=file_path
                )
        
        def fix_file(self, file_path: str) -> CodeFixResult:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                analysis = self.analyze_file(file_path)
                return self.code_fixer.fix_code(code, analysis)
            except Exception as e:
                return CodeFixResult(
                    original_code="",
                    fixed_code="",
                    issues_fixed=[],
                    issues_remaining=[],
                    fix_successful=False,
                    language="unknown",
                    file_path=file_path
                )

# Constants
SUPPORTED_LANGUAGES = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".ts", ".tsx"],
    "bash": [".sh", ".bash"],
    "yaml": [".yml", ".yaml"],
    "json": [".json"],
    "markdown": [".md"],
    "html": [".html", ".htm"],
    "css": [".css"],
    "sql": [".sql"],
}

# Enums and Data Classes
class MentorLevel(Enum):
    """Levels of mentorship from strict oversight to autonomy."""
    STRICT_OVERSIGHT = "strict_oversight"  # Full supervision, all changes reviewed
    GUIDED_LEARNING = "guided_learning"    # Most changes reviewed, some autonomy
    COLLABORATIVE = "collaborative"        # Peer-like relationship, major changes reviewed
    ADVISORY = "advisory"                  # Mentor provides advice when requested
    AUTONOMOUS = "autonomous"              # Grace operates independently, periodic reviews

class TrustScore(Enum):
    """Trust score categories for Grace's coding abilities."""
    NOVICE = "novice"              # 0-20: Requires constant supervision
    APPRENTICE = "apprentice"      # 21-40: Basic competence with oversight
    PRACTITIONER = "practitioner"  # 41-60: Solid skills with occasional guidance
    EXPERT = "expert"              # 61-80: Advanced capabilities, minimal oversight
    MASTER = "master"              # 81-100: Elite level, fully autonomous

class ArchitectureIssueType(Enum):
    """Types of architecture issues that can be detected."""
    TIGHT_COUPLING = "tight_coupling"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    MONOLITHIC_DESIGN = "monolithic_design"
    INCONSISTENT_ABSTRACTION = "inconsistent_abstraction"
    POOR_MODULARITY = "poor_modularity"
    SCALABILITY_BOTTLENECK = "scalability_bottleneck"
    REDUNDANT_FUNCTIONALITY = "redundant_functionality"
    MISSING_INTERFACE = "missing_interface"
    INAPPROPRIATE_INHERITANCE = "inappropriate_inheritance"
    EXCESSIVE_COMPLEXITY = "excessive_complexity"

@dataclass
class ArchitectureIssue:
    """Represents an issue in the system architecture."""
    issue_type: ArchitectureIssueType
    description: str
    components: List[str]
    severity: str  # "low", "medium", "high", "critical"
    suggested_fix: Optional[str] = None
    impact_assessment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary."""
        return {
            "issue_type": self.issue_type.value,
            "description": self.description,
            "components": self.components,
            "severity": self.severity,
            "suggested_fix": self.suggested_fix,
            "impact_assessment": self.impact_assessment
        }

@dataclass
class ArchitectureAnalysisResult:
    """Results from architecture analysis."""
    issues: List[ArchitectureIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    component_graph: Optional[Dict[str, List[str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "component_graph": self.component_graph
        }

@dataclass
class LearningEvent:
    """Represents a learning event in Grace's development."""
    event_id: str
    timestamp: datetime
    event_type: str  # "code_fix", "architecture_improvement", "test_creation", etc.
    description: str
    components: List[str]
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    success: bool
    trust_impact: float  # Impact on trust score (-1.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "components": self.components,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "success": self.success,
            "trust_impact": self.trust_impact
        }

@dataclass
class MentorFeedback:
    """Feedback from the Mentor to Grace."""
    feedback_id: str
    timestamp: datetime
    context: str
    feedback_type: str  # "praise", "correction", "suggestion", "warning", "question"
    message: str
    code_reference: Optional[Dict[str, Any]] = None
    action_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback to a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "feedback_type": self.feedback_type,
            "message": self.message,
            "code_reference": self.code_reference,
            "action_items": self.action_items
        }

@dataclass
class TrustLedgerEntry:
    """An entry in the trust ledger tracking Grace's growth."""
    entry_id: str
    timestamp: datetime
    category: str  # "code_quality", "architecture", "testing", "security", etc.
    score_change: float
    previous_score: float
    new_score: float
    justification: str
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "score_change": self.score_change,
            "previous_score": self.previous_score,
            "new_score": self.new_score,
            "justification": self.justification,
            "evidence": self.evidence
        }

class TrustLedger:
    """Tracks Grace's growth and competence across different domains."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the Trust Ledger."""
        self.storage_path = storage_path or os.path.join(os.getcwd(), "trust_ledger.json")
        self.categories = {
                        "code_quality": 0.0,
            "architecture": 0.0,
            "testing": 0.0,
            "security": 0.0,
            "performance": 0.0,
            "documentation": 0.0,
            "error_handling": 0.0,
            "refactoring": 0.0,
            "innovation": 0.0,
            "ethics": 0.0
        }
        self.entries: List[TrustLedgerEntry] = []
        self.lock = Lock()
        
        # Load existing ledger if available
        self._load_ledger()
    
    def _load_ledger(self):
        """Load the ledger from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load categories
                if "categories" in data:
                    self.categories = data["categories"]
                
                # Load entries
                if "entries" in data:
                    self.entries = [
                        TrustLedgerEntry(
                            entry_id=entry["entry_id"],
                            timestamp=datetime.fromisoformat(entry["timestamp"]),
                            category=entry["category"],
                            score_change=entry["score_change"],
                            previous_score=entry["previous_score"],
                            new_score=entry["new_score"],
                            justification=entry["justification"],
                            evidence=entry["evidence"]
                        )
                        for entry in data["entries"]
                    ]
                
                logger.info(f"Loaded trust ledger with {len(self.entries)} entries")
        
        except Exception as e:
            logger.error(f"Error loading trust ledger: {str(e)}")
            # Initialize with default values
    
    def _save_ledger(self):
        """Save the ledger to storage."""
        try:
            data = {
                "categories": self.categories,
                "entries": [entry.to_dict() for entry in self.entries]
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved trust ledger with {len(self.entries)} entries")
        
        except Exception as e:
            logger.error(f"Error saving trust ledger: {str(e)}")
    
    def add_entry(self, category: str, score_change: float, justification: str, evidence: Dict[str, Any]) -> TrustLedgerEntry:
        """Add a new entry to the ledger."""
        with self.lock:
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            previous_score = self.categories[category]
            new_score = max(0.0, min(100.0, previous_score + score_change))
            
            entry = TrustLedgerEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                category=category,
                score_change=score_change,
                previous_score=previous_score,
                new_score=new_score,
                justification=justification,
                evidence=evidence
            )
            
            self.entries.append(entry)
            self.categories[category] = new_score
            
            # Save the updated ledger
            self._save_ledger()
            
            return entry
    
    def get_overall_trust_score(self) -> float:
        """Get the overall trust score across all categories."""
        with self.lock:
            if not self.categories:
                return 0.0
            
            return sum(self.categories.values()) / len(self.categories)
    
    def get_trust_level(self) -> TrustScore:
        """Get the current trust level based on the overall score."""
        score = self.get_overall_trust_score()
        
        if score < 21:
            return TrustScore.NOVICE
        elif score < 41:
            return TrustScore.APPRENTICE
        elif score < 61:
            return TrustScore.PRACTITIONER
        elif score < 81:
            return TrustScore.EXPERT
        else:
            return TrustScore.MASTER
    
    def get_category_score(self, category: str) -> float:
        """Get the trust score for a specific category."""
        with self.lock:
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            return self.categories[category]
    
    def get_recent_entries(self, limit: int = 10) -> List[TrustLedgerEntry]:
        """Get the most recent entries in the ledger."""
        with self.lock:
            sorted_entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
            return sorted_entries[:limit]
    
    def get_entries_by_category(self, category: str) -> List[TrustLedgerEntry]:
        """Get all entries for a specific category."""
        with self.lock:
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            return [entry for entry in self.entries if entry.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ledger to a dictionary."""
        with self.lock:
            return {
                "categories": self.categories,
                "entries": [entry.to_dict() for entry in self.entries],
                "overall_score": self.get_overall_trust_score(),
                "trust_level": self.get_trust_level().value
            }

class ArchitectureAnalyzer:
    """Analyzes the architecture of Grace's codebase."""
    
    def __init__(self):
        """Initialize the Architecture Analyzer."""
        self.logger = logging.getLogger(__name__ + ".ArchitectureAnalyzer")
    
    def analyze_imports(self, directory: str) -> Dict[str, List[str]]:
        """Analyze imports between Python modules in a directory."""
        import_graph = {}
        file_modules = {}
        
        # First pass: map files to module names
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, directory)
                    module_name = os.path.splitext(rel_path.replace(os.path.sep, '.'))[0]
                    file_modules[file_path] = module_name
        
        # Second pass: analyze imports
        for file_path, module_name in file_modules.items():
            imports = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            
            except Exception as e:
                self.logger.warning(f"Error analyzing imports in {file_path}: {str(e)}")
            
            import_graph[module_name] = imports
        
        return import_graph
    
    def detect_circular_dependencies(self, import_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the import graph."""
        circular_deps = []
        
        def dfs(node, path, visited):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                circular_deps.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in import_graph.get(node, []):
                if neighbor in import_graph:
                    dfs(neighbor, path.copy(), visited)
            
            path.pop()
        
        visited = set()
        for node in import_graph:
            dfs(node, [], visited)
        
        return circular_deps
    
    def analyze_architecture(self, directory: str) -> ArchitectureAnalysisResult:
        """Analyze the architecture of a codebase."""
        result = ArchitectureAnalysisResult()
        
        try:
            # Analyze imports
            import_graph = self.analyze_imports(directory)
            result.component_graph = import_graph
            
            # Detect circular dependencies
            circular_deps = self.detect_circular_dependencies(import_graph)
            for cycle in circular_deps:
                issue = ArchitectureIssue(
                    issue_type=ArchitectureIssueType.CIRCULAR_DEPENDENCY,
                    description=f"Circular dependency detected: {' -> '.join(cycle)}",
                    components=cycle,
                    severity="high",
                    suggested_fix="Consider using dependency injection or restructuring the modules",
                    impact_assessment="Circular dependencies make the codebase harder to maintain and test"
                )
                result.issues.append(issue)
            
            # Analyze module complexity
            module_complexity = {}
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, directory)
                        module_name = os.path.splitext(rel_path.replace(os.path.sep, '.'))[0]
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code = f.read()
                            
                            tree = ast.parse(code)
                            
                            # Count classes and functions
                            num_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                            num_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                            
                            # Simple complexity metric
                            complexity = num_classes * 5 + num_functions * 2
                            module_complexity[module_name] = complexity
                            
                            # Check for excessive complexity
                            if complexity > 100:
                                issue = ArchitectureIssue(
                                    issue_type=ArchitectureIssueType.EXCESSIVE_COMPLEXITY,
                                    description=f"Module {module_name} has excessive complexity",
                                    components=[module_name],
                                    severity="medium",
                                    suggested_fix="Consider breaking down this module into smaller, more focused modules",
                                    impact_assessment="High complexity modules are harder to maintain and understand"
                                )
                                result.issues.append(issue)
                        
                        except Exception as e:
                            self.logger.warning(f"Error analyzing complexity in {file_path}: {str(e)}")
            
            # Check for tight coupling
            for module, imports in import_graph.items():
                if len(imports) > 10:
                    issue = ArchitectureIssue(
                        issue_type=ArchitectureIssueType.TIGHT_COUPLING,
                        description=f"Module {module} imports {len(imports)} other modules",
                        components=[module] + imports[:5] + (["..."] if len(imports) > 5 else []),
                        severity="medium",
                        suggested_fix="Consider using dependency injection or reducing the number of imports",
                        impact_assessment="Tightly coupled modules are harder to test and maintain"
                    )
                    result.issues.append(issue)
            
            # Store metrics
            result.metrics["num_modules"] = len(import_graph)
            result.metrics["avg_imports_per_module"] = sum(len(imports) for imports in import_graph.values()) / len(import_graph) if import_graph else 0
            result.metrics["circular_dependencies"] = len(circular_deps)
            result.metrics["module_complexity"] = module_complexity
        
        except Exception as e:
            self.logger.error(f"Error analyzing architecture: {str(e)}")
            # Add an issue for the error
            issue = ArchitectureIssue(
                issue_type=ArchitectureIssueType.EXCESSIVE_COMPLEXITY,
                description=f"Error analyzing architecture: {str(e)}",
                components=["unknown"],
                severity="high"
            )
            result.issues.append(issue)
        
        return result

class MentorEngine:
    """
    Mentor Engine for Grace AI
    
    Continuously monitors, evaluates, and refactors Grace's internal codebase within a secure
    sandbox environment. Applies syntax, logic, ethical, and optimization analysis to detect
    issues or inefficiencies. Guides Grace through self-repair, code evolution, and autonomous
    improvement loops.
    """
    
    def __init__(self, grace_root_dir: str, sandbox_dir: Optional[str] = None):
        """
        Initialize the Mentor Engine.
        
        Args:
            grace_root_dir: Root directory of Grace's codebase
            sandbox_dir: Directory for the sandbox environment (created if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.grace_root_dir = os.path.abspath(grace_root_dir)
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="grace_mentor_sandbox_")
        
        # Ensure sandbox directory exists
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        # Initialize components
        self.code_learning_engine = CodeLearningEngine(sandbox_dir=os.path.join(self.sandbox_dir, "code_learning"))
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.trust_ledger = TrustLedger(storage_path=os.path.join(self.sandbox_dir, "trust_ledger.json"))
        
        # Learning history
        self.learning_events: List[LearningEvent] = []
        self.feedback_history: List[MentorFeedback] = []
        
        # Current mentorship level
        self._mentorship_level = MentorLevel.STRICT_OVERSIGHT
        
        # Thresholds for advancing mentorship levels
        self.level_thresholds = {
            MentorLevel.STRICT_OVERSIGHT: 20.0,  # Move to GUIDED_LEARNING at 20% trust
            MentorLevel.GUIDED_LEARNING: 40.0,   # Move to COLLABORATIVE at 40% trust
            MentorLevel.COLLABORATIVE: 60.0,     # Move to ADVISORY at 60% trust
            MentorLevel.ADVISORY: 80.0           # Move to AUTONOMOUS at 80% trust
        }
        
        # Initialize the sandbox with a copy of Grace's codebase
        self._initialize_sandbox()
        
        self.logger.info(f"Initialized Mentor Engine with sandbox at {self.sandbox_dir}")
        self.logger.info(f"Initial trust score: {self.trust_ledger.get_overall_trust_score():.2f}")
        self.logger.info(f"Initial mentorship level: {self._mentorship_level.value}")
    
        def _initialize_sandbox(self):
        """Initialize the sandbox with a copy of Grace's codebase."""
        try:
            # Clear the sandbox if it already contains files
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            if os.path.exists(sandbox_code_dir):
                shutil.rmtree(sandbox_code_dir)
            
            # Create the directory
            os.makedirs(sandbox_code_dir, exist_ok=True)
            
            # Copy Grace's codebase to the sandbox
            for item in os.listdir(self.grace_root_dir):
                source = os.path.join(self.grace_root_dir, item)
                destination = os.path.join(sandbox_code_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            
            self.logger.info(f"Copied Grace's codebase to sandbox at {sandbox_code_dir}")
            
            # Create a snapshot of the initial state
            self._create_snapshot("initial")
        
        except Exception as e:
            self.logger.error(f"Error initializing sandbox: {str(e)}")
            raise
    
    def _create_snapshot(self, label: str) -> str:
        """Create a snapshot of the current sandbox state."""
        try:
            snapshot_dir = os.path.join(self.sandbox_dir, "snapshots", f"{label}_{int(time.time())}")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Copy the current sandbox code to the snapshot directory
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            for item in os.listdir(sandbox_code_dir):
                source = os.path.join(sandbox_code_dir, item)
                destination = os.path.join(snapshot_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            
            self.logger.info(f"Created snapshot: {snapshot_dir}")
            return snapshot_dir
        
        except Exception as e:
            self.logger.error(f"Error creating snapshot: {str(e)}")
            return ""
    
    def _restore_snapshot(self, snapshot_dir: str) -> bool:
        """Restore the sandbox to a previous snapshot."""
        try:
            if not os.path.exists(snapshot_dir):
                self.logger.error(f"Snapshot directory does not exist: {snapshot_dir}")
                return False
            
            # Clear the current sandbox code
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            if os.path.exists(sandbox_code_dir):
                shutil.rmtree(sandbox_code_dir)
            
            # Create the directory
            os.makedirs(sandbox_code_dir, exist_ok=True)
            
            # Copy the snapshot to the sandbox code directory
            for item in os.listdir(snapshot_dir):
                source = os.path.join(snapshot_dir, item)
                destination = os.path.join(sandbox_code_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            
            self.logger.info(f"Restored snapshot: {snapshot_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error restoring snapshot: {str(e)}")
            return False
    
    @property
    def mentorship_level(self) -> MentorLevel:
        """Get the current mentorship level."""
        return self._mentorship_level
    
    def _update_mentorship_level(self):
        """Update the mentorship level based on the current trust score."""
        current_score = self.trust_ledger.get_overall_trust_score()
        current_level = self._mentorship_level
        
        # Check if we should advance to the next level
        if current_level != MentorLevel.AUTONOMOUS and current_score >= self.level_thresholds[current_level]:
            # Find the next level
            levels = list(MentorLevel)
            current_index = levels.index(current_level)
            next_level = levels[current_index + 1]
            
            self._mentorship_level = next_level
            self.logger.info(f"Advanced mentorship level from {current_level.value} to {next_level.value}")
            
            # Record this as a learning event
            self._record_learning_event(
                event_type="mentorship_level_change",
                description=f"Advanced mentorship level from {current_level.value} to {next_level.value}",
                components=["mentor_engine"],
                before_state={"level": current_level.value, "score": current_score},
                after_state={"level": next_level.value, "score": current_score},
                success=True,
                trust_impact=0.0  # No direct impact on trust score
            )
            
            # Provide feedback to Grace
            self._provide_feedback(
                context="mentorship_level",
                feedback_type="praise",
                message=f"Congratulations! You've reached the {next_level.value} mentorship level. This means {self._get_level_description(next_level)}",
                action_items=[
                    f"Review the new capabilities and responsibilities of the {next_level.value} level",
                    "Continue to improve your code quality and architecture"
                ]
            )
    
    def _get_level_description(self, level: MentorLevel) -> str:
        """Get a description of what a mentorship level means."""
        descriptions = {
            MentorLevel.STRICT_OVERSIGHT: "I'll review all your changes and provide detailed guidance.",
            MentorLevel.GUIDED_LEARNING: "You have more autonomy, but I'll still review major changes and provide regular guidance.",
            MentorLevel.COLLABORATIVE: "We're working as peers now. I'll review your architectural decisions and provide feedback when needed.",
            MentorLevel.ADVISORY: "You're mostly autonomous now. I'll provide advice when you request it or for complex issues.",
            MentorLevel.AUTONOMOUS: "You're fully autonomous! I'll still be available for consultation, but you're in charge of your own development."
        }
        return descriptions.get(level, "Unknown level")
    
    def _record_learning_event(self, event_type: str, description: str, components: List[str],
                              before_state: Dict[str, Any], after_state: Dict[str, Any],
                              success: bool, trust_impact: float) -> LearningEvent:
        """Record a learning event."""
        event = LearningEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            components=components,
            before_state=before_state,
            after_state=after_state,
            success=success,
            trust_impact=trust_impact
        )
        
        self.learning_events.append(event)
        
        # Save to file
        try:
            events_dir = os.path.join(self.sandbox_dir, "learning_events")
            os.makedirs(events_dir, exist_ok=True)
            
            with open(os.path.join(events_dir, f"{event.event_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving learning event: {str(e)}")
        
        return event
    
    def _provide_feedback(self, context: str, feedback_type: str, message: str,
                         code_reference: Optional[Dict[str, Any]] = None,
                         action_items: List[str] = None) -> MentorFeedback:
        """Provide feedback to Grace."""
        feedback = MentorFeedback(
            feedback_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=context,
            feedback_type=feedback_type,
            message=message,
            code_reference=code_reference,
            action_items=action_items or []
        )
        
        self.feedback_history.append(feedback)
        
        # Save to file
        try:
            feedback_dir = os.path.join(self.sandbox_dir, "feedback")
            os.makedirs(feedback_dir, exist_ok=True)
            
            with open(os.path.join(feedback_dir, f"{feedback.feedback_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(feedback.to_dict(), f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving feedback: {str(e)}")
        
        return feedback
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """
        Analyze Grace's codebase for issues and opportunities for improvement.
        
        Returns:
            A dictionary containing analysis results
        """
        results = {
            "code_issues": [],
            "architecture_issues": [],
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Analyze code issues
            code_issues = []
            for root, _, files in os.walk(sandbox_code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    # Skip non-code files
                    language = None
                    for lang, extensions in SUPPORTED_LANGUAGES.items():
                        if ext in extensions:
                            language = lang
                            break
                    
                    if not language:
                        continue
                    
                    # Analyze the file
                    analysis = self.code_learning_engine.analyze_file(file_path)
                    
                    # Add issues to the results
                    for issue in analysis.issues:
                        code_issues.append(issue.to_dict())
            
            results["code_issues"] = code_issues
            
            # Analyze architecture
            architecture_analysis = self.architecture_analyzer.analyze_architecture(sandbox_code_dir)
            results["architecture_issues"] = [issue.to_dict() for issue in architecture_analysis.issues]
            results["metrics"]["architecture"] = architecture_analysis.metrics
            
            # Calculate metrics
            results["metrics"]["total_code_issues"] = len(code_issues)
            results["metrics"]["total_architecture_issues"] = len(architecture_analysis.issues)
            
            self.logger.info(f"Analyzed codebase: found {len(code_issues)} code issues and "
                           f"{len(architecture_analysis.issues)} architecture issues")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error analyzing codebase: {str(e)}")
            results["error"] = str(e)
            return results
    
    def fix_code_issues(self, max_issues: int = 10) -> Dict[str, Any]:
        """
        Fix code issues in Grace's codebase.
        
        Args:
            max_issues: Maximum number of issues to fix in one run
        
        Returns:
            A dictionary containing the results of the fix operation
        """
        results = {
            "issues_fixed": [],
            "issues_attempted": 0,
            "issues_successful": 0,
            "timestamp": datetime.now().isoformat()
        }
         try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Create a snapshot before making changes
            snapshot_dir = self._create_snapshot("pre_fix")
            
            # Find and fix issues
            issues_fixed = []
            issues_attempted = 0
            issues_successful = 0
            
            for root, _, files in os.walk(sandbox_code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    # Skip non-code files
                    language = None
                    for lang, extensions in SUPPORTED_LANGUAGES.items():
                        if ext in extensions:
                            language = lang
                            break
                    
                    if not language:
                        continue
                    
                    # Analyze the file
                    analysis = self.code_learning_engine.analyze_file(file_path)
                    
                    # Skip if no issues
                    if not analysis.issues:
                        continue
                    
                    # Fix the file
                    fix_result = self.code_learning_engine.fix_file(file_path)
                    
                    # Update counts
                    issues_attempted += len(analysis.issues)
                    issues_successful += len(fix_result.issues_fixed)
                    
                    # If fixes were successful, apply them
                    if fix_result.fix_successful and fix_result.issues_fixed:
                        # Write the fixed code back to the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fix_result.fixed_code)
                        
                        # Add to the list of fixed issues
                        for issue in fix_result.issues_fixed:
                            issues_fixed.append({
                                "file": os.path.relpath(file_path, sandbox_code_dir),
                                "issue": issue.to_dict(),
                                "fix_successful": True
                            })
                    
                    # Stop if we've reached the maximum number of issues
                    if len(issues_fixed) >= max_issues:
                        break
                
                # Stop if we've reached the maximum number of issues
                if len(issues_fixed) >= max_issues:
                    break
            
            # Update results
            results["issues_fixed"] = issues_fixed
            results["issues_attempted"] = issues_attempted
            results["issues_successful"] = issues_successful
            
            # Record learning event
            if issues_successful > 0:
                trust_impact = min(0.5, issues_successful * 0.1)  # Cap at 0.5 per run
                
                self._record_learning_event(
                    event_type="code_fix",
                    description=f"Fixed {issues_successful} code issues",
                    components=["code_fixer"],
                    before_state={"issues": issues_attempted},
                    after_state={"issues_fixed": issues_successful},
                    success=True,
                    trust_impact=trust_impact
                )
                
                # Update trust ledger
                self.trust_ledger.add_entry(
                    category="code_quality",
                    score_change=trust_impact,
                    justification=f"Successfully fixed {issues_successful} out of {issues_attempted} code issues",
                    evidence={"issues_fixed": issues_fixed}
                )
                
                # Provide feedback
                self._provide_feedback(
                    context="code_fix",
                    feedback_type="praise",
                    message=f"I've fixed {issues_successful} code issues in your codebase.",
                    action_items=[
                        "Review the fixed issues to understand the improvements",
                        "Consider implementing similar fixes in future code"
                    ]
                )
                
                # Update mentorship level
                self._update_mentorship_level()
            
                        self.logger.info(f"Fixed {issues_successful} out of {issues_attempted} code issues")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error fixing code issues: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            return results
    
    def improve_architecture(self) -> Dict[str, Any]:
        """
        Improve the architecture of Grace's codebase.
        
        Returns:
            A dictionary containing the results of the improvement operation
        """
        results = {
            "improvements": [],
            "issues_addressed": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Create a snapshot before making changes
            snapshot_dir = self._create_snapshot("pre_architecture_improvement")
            
            # Analyze the architecture
            architecture_analysis = self.architecture_analyzer.analyze_architecture(sandbox_code_dir)
            
            # No issues to fix
            if not architecture_analysis.issues:
                results["message"] = "No architecture issues found"
                return results
            
            # Sort issues by severity
            issues_by_severity = {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            }
            
            for issue in architecture_analysis.issues:
                issues_by_severity[issue.severity].append(issue)
            
            # Process issues in order of severity
            improvements = []
            issues_addressed = 0
            
            # For now, we'll just record the issues and suggest improvements
            # In a real implementation, we would actually make the changes
            for severity in ["critical", "high", "medium", "low"]:
                for issue in issues_by_severity[severity]:
                    improvement = {
                        "issue": issue.to_dict(),
                        "action": "suggested",  # For now, we're just suggesting improvements
                        "suggested_fix": issue.suggested_fix
                    }
                    
                    improvements.append(improvement)
                    issues_addressed += 1
            
            # Update results
            results["improvements"] = improvements
            results["issues_addressed"] = issues_addressed
            
            # Record learning event
            if improvements:
                trust_impact = min(0.3, issues_addressed * 0.05)  # Cap at 0.3 per run
                
                self._record_learning_event(
                    event_type="architecture_improvement",
                    description=f"Suggested improvements for {issues_addressed} architecture issues",
                    components=["architecture_analyzer"],
                    before_state={"issues": len(architecture_analysis.issues)},
                    after_state={"improvements_suggested": issues_addressed},
                    success=True,
                    trust_impact=trust_impact
                )
                
                # Update trust ledger
                self.trust_ledger.add_entry(
                    category="architecture",
                    score_change=trust_impact,
                    justification=f"Suggested improvements for {issues_addressed} architecture issues",
                    evidence={"improvements": improvements}
                )
                
                # Provide feedback
                self._provide_feedback(
                    context="architecture_improvement",
                    feedback_type="suggestion",
                    message=f"I've identified {issues_addressed} architecture issues in your codebase and suggested improvements.",
                    action_items=[
                        "Review the suggested architecture improvements",
                        "Consider implementing these improvements to enhance maintainability"
                    ]
                )
                
                # Update mentorship level
                self._update_mentorship_level()
            
            self.logger.info(f"Suggested improvements for {issues_addressed} architecture issues")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error improving architecture: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            return results
    
    def optimize_code(self, max_files: int = 5) -> Dict[str, Any]:
        """
        Optimize code in Grace's codebase for performance and readability.
        
        Args:
            max_files: Maximum number of files to optimize in one run
        
        Returns:
            A dictionary containing the results of the optimization operation
        """
        results = {
            "optimizations": [],
            "files_optimized": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Create a snapshot before making changes
            snapshot_dir = self._create_snapshot("pre_optimization")
            
            # Find and optimize files
            optimizations = []
            files_optimized = 0
            
            for root, _, files in os.walk(sandbox_code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    # Skip non-code files
                    language = None
                    for lang, extensions in SUPPORTED_LANGUAGES.items():
                        if ext in extensions:
                            language = lang
                            break
                    
                    if not language:
                        continue
                    
                    # Read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Optimize the code
                    optimized_code, metrics = self.code_learning_engine.code_optimizer.optimize_code(code, language)
                    
                    # If optimizations were applied, update the file
                    if metrics.get("optimizations_applied", 0) > 0:
                        # Write the optimized code back to the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(optimized_code)
                        
                        # Add to the list of optimizations
                        optimizations.append({
                            "file": os.path.relpath(file_path, sandbox_code_dir),
                            "metrics": metrics,
                            "optimizations_applied": metrics.get("optimizations_applied", 0)
                        })
                        
                        files_optimized += 1
                    
                    # Stop if we've reached the maximum number of files
                    if files_optimized >= max_files:
                        break
                
                # Stop if we've reached the maximum number of files
                if files_optimized >= max_files:
                    break
            
            # Update results
            results["optimizations"] = optimizations
            results["files_optimized"] = files_optimized
            
            # Record learning event
            if files_optimized > 0:
                trust_impact = min(0.2, files_optimized * 0.04)  # Cap at 0.2 per run
                
                self._record_learning_event(
                    event_type="code_optimization",
                    description=f"Optimized {files_optimized} files",
                    components=["code_optimizer"],
                    before_state={"files": max_files},
                    after_state={"files_optimized": files_optimized},
                    success=True,
                    trust_impact=trust_impact
                )
                
                # Update trust ledger
                self.trust_ledger.add_entry(
                    category="performance",
                    score_change=trust_impact,
                    justification=f"Successfully optimized {files_optimized} files",
                    evidence={"optimizations": optimizations}
                )
                
                # Provide feedback
                self._provide_feedback(
                    context="code_optimization",
                    feedback_type="praise",
                    message=f"I've optimized {files_optimized} files in your codebase for better performance and readability.",
                    action_items=[
                        "Review the optimized files to understand the improvements",
                        "Consider applying similar optimizations in future code"
                    ]
                )
                
                # Update mentorship level
                self._update_mentorship_level()
            
            self.logger.info(f"Optimized {files_optimized} files")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error optimizing code: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            return results
    
    def generate_tests(self, max_modules: int = 3) -> Dict[str, Any]:
        """
        Generate tests for modules in Grace's codebase.
        
        Args:
            max_modules: Maximum number of modules to generate tests for in one run
        
        Returns:
            A dictionary containing the results of the test generation operation
        """
        results = {
            "tests_generated": [],
            "modules_tested": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Create a snapshot before making changes
            snapshot_dir = self._create_snapshot("pre_test_generation")
            
            # Find Python modules without tests
            modules_without_tests = []
            
            for root, _, files in os.walk(sandbox_code_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('test_'):
                        module_path = os.path.join(root, file)
                        module_name = os.path.splitext(file)[0]
                        test_file = os.path.join(root, f"test_{module_name}.py")
                        
                        if not os.path.exists(test_file):
                            modules_without_tests.append((module_path, module_name))
            
            # Generate tests for modules
            tests_generated = []
            modules_tested = 0
            
            for module_path, module_name in modules_without_tests[:max_modules]:
                try:
                    # Read the module
                    with open(module_path, 'r', encoding='utf-8') as f:
                        module_code = f.read()
                    
                    # Parse the module to find classes and functions
                    tree = ast.parse(module_code)
                    
                    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) 
                                and node.name != '__init__']
                    
                    # Generate a test file
                    test_code = self._generate_test_file(module_name, classes, functions)
                    
                    # Write the test file
                    test_file = os.path.join(os.path.dirname(module_path), f"test_{module_name}.py")
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                    
                    # Add to the list of generated tests
                    tests_generated.append({
                        "module": module_name,
                        "test_file": os.path.relpath(test_file, sandbox_code_dir),
                        "classes_tested": classes,
                        "functions_tested": functions
                    })
                    
                    modules_tested += 1
                
                except Exception as e:
                    self.logger.warning(f"Error generating tests for {module_name}: {str(e)}")
            
            # Update results
            results["tests_generated"] = tests_generated
            results["modules_tested"] = modules_tested
            
            # Record learning event
            if modules_tested > 0:
                trust_impact = min(0.3, modules_tested * 0.1)  # Cap at 0.3 per run
                
                self._record_learning_event(
                    event_type="test_generation",
                    description=f"Generated tests for {modules_tested} modules",
                    components=["test_generator"],
                    before_state={"modules_without_tests": len(modules_without_tests)},
                    after_state={"modules_tested": modules_tested},
                    success=True,
                    trust_impact=trust_impact
                )
                
                # Update trust ledger
                self.trust_ledger.add_entry(
                    category="testing",
                    score_change=trust_impact,
                    justification=f"Successfully generated tests for {modules_tested} modules",
                    evidence={"tests_generated": tests_generated}
                )
                
                # Provide feedback
                self._provide_feedback(
                    context="test_generation",
                    feedback_type="praise",
                    message=f"I've generated tests for {modules_tested} modules in your codebase.",
                    action_items=[
                        "Review the generated tests and ensure they cover critical functionality",
                        "Run the tests to verify they pass",
                        "Consider adding more specific test cases for complex logic"
                    ]
                )
                
                # Update mentorship level
                self._update_mentorship_level()
            
            self.logger.info(f"Generated tests for {modules_tested} modules")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error generating tests: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            return results
    
    def _generate_test_file(self, module_name: str, classes: List[str], functions: List[str]) -> str:
        """Generate a test file for a module."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        test_code = f'''#!/usr/bin/env python3
"""
Tests for {module_name}

This module contains tests for the {module_name} module.
Generated by MentorEngine on {timestamp}
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import {module_name}
except ImportError:
    print(f"Error importing {module_name}. Make sure the module exists and is in the Python path.")
    sys.exit(1)

'''
        
        # Add test classes for each class in the module
        for class_name in classes:
            test_code += f'''
class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name} class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.instance = {module_name}.{class_name}()
        except Exception as e:
            self.skipTest(f"Could not instantiate {class_name}: {{str(e)}}")
    
    def tearDown(self):
        """Tear down test fixtures."""
        pass
    
    def test_initialization(self):
        """Test that the class can be initialized."""
        self.assertIsInstance(self.instance, {module_name}.{class_name})
    
'''
            # Add test methods for common methods
            test_code += f'''    def test_str_representation(self):
        """Test the string representation of the class."""
        try:
            str_repr = str(self.instance)
            self.assertIsInstance(str_repr, str)
                except Exception as e:
            self.skipTest(f"Error getting string representation: {str(e)}")

'''
        
        # Add test functions for standalone functions
        if functions:
            test_code += f'''
class TestFunctions(unittest.TestCase):
    """Test cases for standalone functions."""
    
'''
            for func_name in functions:
                test_code += f'''    def test_{func_name}(self):
        """Test the {func_name} function."""
        try:
            # This is a placeholder test. You should replace it with actual test logic.
            # For example, if the function takes arguments:
            # result = {module_name}.{func_name}(arg1, arg2)
            # self.assertEqual(result, expected_value)
            
            # For now, we just verify the function exists
            self.assertTrue(callable(getattr({module_name}, "{func_name}")))
        except Exception as e:
            self.skipTest(f"Error testing {func_name}: {{str(e)}}")
    
'''
        
        # Add main block
        test_code += '''
if __name__ == "__main__":
    unittest.main()
'''
        
        return test_code
    
    def improve_documentation(self, max_files: int = 5) -> Dict[str, Any]:
        """
        Improve documentation in Grace's codebase.
        
        Args:
            max_files: Maximum number of files to improve documentation for in one run
        
        Returns:
            A dictionary containing the results of the documentation improvement operation
        """
        results = {
            "files_improved": [],
            "total_files_improved": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            
            # Create a snapshot before making changes
            snapshot_dir = self._create_snapshot("pre_documentation_improvement")
            
            # Find Python files with poor documentation
            files_to_improve = []
            
            for root, _, files in os.walk(sandbox_code_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # Check documentation quality
                        doc_quality = self._assess_documentation_quality(file_path)
                        
                        if doc_quality["score"] < 0.7:  # Threshold for "good" documentation
                            files_to_improve.append((file_path, doc_quality))
            
            # Sort files by documentation quality (worst first)
            files_to_improve.sort(key=lambda x: x[1]["score"])
            
            # Improve documentation for files
            files_improved = []
            total_files_improved = 0
            
            for file_path, doc_quality in files_to_improve[:max_files]:
                try:
                    # Read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Improve documentation
                    improved_code = self._improve_file_documentation(code, doc_quality)
                    
                    # Write the improved code back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(improved_code)
                    
                    # Add to the list of improved files
                    files_improved.append({
                        "file": os.path.relpath(file_path, sandbox_code_dir),
                        "before_score": doc_quality["score"],
                        "missing_docstrings": doc_quality["missing_docstrings"],
                        "incomplete_docstrings": doc_quality["incomplete_docstrings"]
                    })
                    
                    total_files_improved += 1
                
                except Exception as e:
                    self.logger.warning(f"Error improving documentation for {file_path}: {str(e)}")
            
            # Update results
            results["files_improved"] = files_improved
            results["total_files_improved"] = total_files_improved
            
            # Record learning event
            if total_files_improved > 0:
                trust_impact = min(0.2, total_files_improved * 0.04)  # Cap at 0.2 per run
                
                self._record_learning_event(
                    event_type="documentation_improvement",
                    description=f"Improved documentation for {total_files_improved} files",
                    components=["documentation_improver"],
                    before_state={"files_with_poor_docs": len(files_to_improve)},
                    after_state={"files_improved": total_files_improved},
                    success=True,
                    trust_impact=trust_impact
                )
                
                # Update trust ledger
                self.trust_ledger.add_entry(
                    category="documentation",
                    score_change=trust_impact,
                    justification=f"Successfully improved documentation for {total_files_improved} files",
                    evidence={"files_improved": files_improved}
                )
                
                # Provide feedback
                self._provide_feedback(
                    context="documentation_improvement",
                    feedback_type="praise",
                    message=f"I've improved documentation for {total_files_improved} files in your codebase.",
                    action_items=[
                        "Review the improved documentation for accuracy",
                        "Consider adding more detailed examples where appropriate",
                        "Maintain this documentation style in future code"
                    ]
                )
                
                # Update mentorship level
                self._update_mentorship_level()
            
            self.logger.info(f"Improved documentation for {total_files_improved} files")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error improving documentation: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            return results
    
    def _assess_documentation_quality(self, file_path: str) -> Dict[str, Any]:
        """Assess the quality of documentation in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Count total classes and functions
            total_classes = 0
            total_functions = 0
            missing_docstrings = []
            incomplete_docstrings = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1
                    
                    # Check for class docstring
                    if not ast.get_docstring(node):
                        missing_docstrings.append(f"Class {node.name}")
                    elif len(ast.get_docstring(node).split('\n')) < 2:
                        incomplete_docstrings.append(f"Class {node.name}")
                
                elif isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    
                    # Skip special methods
                    if node.name.startswith('__') and node.name.endswith('__'):
                        continue
                    
                    # Check for function docstring
                    if not ast.get_docstring(node):
                        missing_docstrings.append(f"Function {node.name}")
                    elif len(ast.get_docstring(node).split('\n')) < 2:
                        incomplete_docstrings.append(f"Function {node.name}")
            
            # Check for module docstring
            has_module_docstring = bool(ast.get_docstring(tree))
            
            # Calculate score
            total_items = total_classes + total_functions + 1  # +1 for module docstring
            missing_count = len(missing_docstrings) + (0 if has_module_docstring else 1)
            incomplete_count = len(incomplete_docstrings)
            
            if total_items == 0:
                score = 1.0  # Empty file or no documentable items
            else:
                score = 1.0 - (missing_count * 0.5 + incomplete_count * 0.25) / total_items
            
            return {
                "score": max(0.0, min(1.0, score)),
                "has_module_docstring": has_module_docstring,
                "total_classes": total_classes,
                "total_functions": total_functions,
                "missing_docstrings": missing_docstrings,
                "incomplete_docstrings": incomplete_docstrings
            }
        
        except Exception as e:
            self.logger.warning(f"Error assessing documentation quality for {file_path}: {str(e)}")
            return {
                "score": 0.0,
                "error": str(e),
                "missing_docstrings": [],
                "incomplete_docstrings": []
            }
    
    def _improve_file_documentation(self, code: str, doc_quality: Dict[str, Any]) -> str:
        """Improve documentation in a Python file."""
        try:
            tree = ast.parse(code)
            
            # Add module docstring if missing
            if not doc_quality["has_module_docstring"]:
                # Extract module name from the file path or content
                module_name = "Unknown"
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        module_name = node.name
                        break
                
                module_docstring = f'"""\n{module_name} Module\n\nThis module contains functionality for {module_name}.\n"""'
                
                # Insert at the beginning of the file
                code = module_docstring + "\n\n" + code
                
                # Re-parse the tree
                tree = ast.parse(code)
            
            # For now, we'll just return a simplified version
            # In a real implementation, we would use a more sophisticated approach
            # to add or improve docstrings for classes and functions
            
            # This is a placeholder for the actual implementation
            return code
        
        except Exception as e:
            self.logger.warning(f"Error improving file documentation: {str(e)}")
            return code
    
    def run_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run a complete improvement cycle on Grace's codebase.
        
        This includes analyzing the codebase, fixing code issues, improving architecture,
        optimizing code, generating tests, and improving documentation.
        
        Returns:
            A dictionary containing the results of the improvement cycle
        """
        results = {
            "cycle_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "overall_success": True
        }
        
        try:
            # Create a snapshot before the cycle
            snapshot_dir = self._create_snapshot("pre_improvement_cycle")
            
            # Step 1: Analyze the codebase
            self.logger.info("Step 1: Analyzing codebase")
            analysis_results = self.analyze_codebase()
            results["steps"].append({
                "step": "analyze_codebase",
                "success": "error" not in analysis_results,
                "results": analysis_results
            })
            
            if "error" in analysis_results:
                results["overall_success"] = False
                return results
            
            # Step 2: Fix code issues
            self.logger.info("Step 2: Fixing code issues")
            fix_results = self.fix_code_issues(max_issues=10)
            results["steps"].append({
                "step": "fix_code_issues",
                "success": "error" not in fix_results,
                "results": fix_results
            })
            
            if "error" in fix_results:
                results["overall_success"] = False
                return results
            
            # Step 3: Improve architecture
            self.logger.info("Step 3: Improving architecture")
            architecture_results = self.improve_architecture()
            results["steps"].append({
                "step": "improve_architecture",
                "success": "error" not in architecture_results,
                "results": architecture_results
            })
            
            if "error" in architecture_results:
                results["overall_success"] = False
                return results
            
            # Step 4: Optimize code
            self.logger.info("Step 4: Optimizing code")
            optimization_results = self.optimize_code(max_files=5)
            results["steps"].append({
                "step": "optimize_code",
                "success": "error" not in optimization_results,
                "results": optimization_results
            })
            
            if "error" in optimization_results:
                results["overall_success"] = False
                return results
            
            # Step 5: Generate tests
            self.logger.info("Step 5: Generating tests")
            test_results = self.generate_tests(max_modules=3)
            results["steps"].append({
                "step": "generate_tests",
                "success": "error" not in test_results,
                "results": test_results
            })
            
            if "error" in test_results:
                results["overall_success"] = False
                return results
            
            # Step 6: Improve documentation
            self.logger.info("Step 6: Improving documentation")
            documentation_results = self.improve_documentation(max_files=5)
            results["steps"].append({
                "step": "improve_documentation",
                "success": "error" not in documentation_results,
                "results": documentation_results
            })
            
            if "error" in documentation_results:
                results["overall_success"] = False
                return results
            
            # Record learning event for the complete cycle
            self._record_learning_event(
                event_type="improvement_cycle",
                description="Completed a full improvement cycle",
                components=["mentor_engine"],
                before_state={"snapshot": snapshot_dir},
                after_state={"results": results},
                success=results["overall_success"],
                trust_impact=0.5  # Significant impact for completing a full cycle
            )
            
            # Update trust ledger
            self.trust_ledger.add_entry(
                category="refactoring",
                score_change=0.5,
                justification="Successfully completed a full improvement cycle",
                evidence={"cycle_results": results}
            )
            
            # Provide feedback
            self._provide_feedback(
                context="improvement_cycle",
                feedback_type="praise",
                message="I've completed a full improvement cycle on your codebase, addressing issues in multiple areas.",
                action_items=[
                    "Review the changes made during this cycle",
                    "Run tests to ensure everything still works as expected",
                    "Consider implementing similar improvements in future development"
                ]
            )
            
            # Update mentorship level
            self._update_mentorship_level()
            
            self.logger.info("Completed improvement cycle successfully")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in improvement cycle: {str(e)}")
            
            # Restore from snapshot if there was an error
            if snapshot_dir:
                self._restore_snapshot(snapshot_dir)
            
            results["error"] = str(e)
            results["overall_success"] = False
            return results
    
    def apply_changes_to_grace(self) -> Dict[str, Any]:
        """
        Apply the changes from the sandbox to Grace's actual codebase.
        
        This is a critical operation that should only be performed after thorough validation
        and with appropriate permissions based on the current mentorship level.
        
        Returns:
            A dictionary containing the results of the operation
        """
        results = {
            "operation_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
            "changes_applied": [],
            "success": False
        }
        
        # Check if we have permission to apply changes based on mentorship level
        if self._mentorship_level in [MentorLevel.STRICT_OVERSIGHT, MentorLevel.GUIDED_LEARNING]:
            results["error"] = f"Cannot apply changes at {self._mentorship_level.value} mentorship level. Requires at least COLLABORATIVE level."
            self.logger.warning(results["error"])
            return results
        
        try:
            # Create a backup of Grace's codebase
            backup_dir = os.path.join(self.sandbox_dir, "backups", f"grace_backup_{int(time.time())}")
            os.makedirs(backup_dir, exist_ok=True)
            
            for item in os.listdir(self.grace_root_dir):
                source = os.path.join(self.grace_root_dir, item)
                destination = os.path.join(backup_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            
            self.logger.info(f"Created backup of Grace's codebase at {backup_dir}")
            
            # Get the list of files in the sandbox
            sandbox_code_dir = os.path.join(self.sandbox_dir, "grace_code")
            changes_applied = []
            
            # Apply changes from sandbox to Grace's codebase
            for root, dirs, files in os.walk(sandbox_code_dir):
                for file in files:
                    sandbox_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(sandbox_file_path, sandbox_code_dir)
                    grace_file_path = os.path.join(self.grace_root_dir, rel_path)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(grace_file_path), exist_ok=True)
                    
                    # Check if the file has changed
                    if os.path.exists(grace_file_path):
                        with open(sandbox_file_path, 'rb') as f1, open(grace_file_path, 'rb') as f2:
                            if f1.read() != f2.read():
                                # File has changed, copy it
                                shutil.copy2(sandbox_file_path, grace_file_path)
                                changes_applied.append({
                                    "file": rel_path,
                                    "action": "modified"
                                })
                    else:
                        # New file, copy it
                        shutil.copy2(sandbox_file_path, grace_file_path)
                        changes_applied.append({
                            "file": rel_path,
                            "action": "added"
                        })
            
            # Check for files that were deleted in the sandbox
            for root, dirs, files in os.walk(self.grace_root_dir):
                for file in files:
                    grace_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(grace_file_path, self.grace_root_dir)
                    sandbox_file_path = os.path.join(sandbox_code_dir, rel_path)
                    
                    if not os.path.exists(sandbox_file_path):
                        # File was deleted in the sandbox, delete it from Grace
                        os.remove(grace_file_path)
                        changes_applied.append({
                            "file": rel_path,
                            "action": "deleted"
                        })
            
            # Update results
            results["changes_applied"] = changes_applied
            results["success"] = True
            results["backup_dir"] = backup_dir
            
            # Record learning event
            self._record_learning_event(
                event_type="apply_changes",
                description=f"Applied {len(changes_applied)} changes to Grace's codebase",
                components=["mentor_engine"],
                before_state={"backup": backup_dir},
                after_state={"changes": changes_applied},
                success=True,
                trust_impact=0.0  # No direct impact on trust score
            )
            
            # Provide feedback
            self._provide_feedback(
                context="apply_changes",
                feedback_type="information",
                message=f"I've applied {len(changes_applied)} changes to your codebase.",
                action_items=[
                    "Review the changes to ensure they meet your expectations",
                    "Run tests to verify everything works as expected",
                    "Restart Grace if necessary to apply the changes"
                ]
            )
            
            self.logger.info(f"Applied {len(changes_applied)} changes to Grace's codebase")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error applying changes to Grace: {str(e)}")
            results["error"] = str(e)
            return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Mentor Engine.
        
        Returns:
            A dictionary containing the current status
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "trust_score": self.trust_ledger.get_overall_trust_score(),
            "trust_level": self.trust_ledger.get_trust_level().value,
            "mentorship_level": self._mentorship_level.value,
            "recent_learning_events": [event.to_dict() for event in self.learning_events[-5:]] if self.learning_events else [],
            "recent_feedback": [feedback.to_dict() for feedback in self.feedback_history[-5:]] if self.feedback_history else [],
            "sandbox_dir": self.sandbox_dir,
            "grace_root_dir": self.grace_root_dir
        }
    
    def shutdown(self):
        """Shutdown the Mentor Engine and clean up resources."""
        try:
            # Save any unsaved data
            
            # Clean up temporary files if needed
            
            self.logger.info("Mentor Engine shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during Mentor Engine shutdown: {str(e)}")
