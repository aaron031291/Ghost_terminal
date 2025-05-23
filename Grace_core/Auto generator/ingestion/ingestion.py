import os
import sys
import ast
import uuid
import shutil
import logging
import tempfile
import zipfile
import hashlib
import importlib
import importlib.util
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import fastapi
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field, validator
from starlette.responses import JSONResponse

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("grace.ingestion")


class IngestionStatus(Enum):
    """Status of an ingestion operation"""
    PENDING = auto()
    VALIDATING = auto()
    SANDBOXING = auto()
    ANALYZING = auto()
    SCORING = auto()
    STAGING = auto()
    INTEGRATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REJECTED = auto()


class IngestionType(Enum):
    """Type of ingested content"""
    SINGLE_FILE = auto()
    MODULE = auto()
    PACKAGE = auto()


class TrustLevel(Enum):
    """Trust level for ingested code"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


class IngestionRequest(BaseModel):
    """Request model for ingestion operations"""
    name: str = Field(..., description="Name for the ingested component")
    description: Optional[str] = Field(None, description="Description of functionality")
    replace_existing: bool = Field(False, description="Whether to replace existing modules")
    auto_integrate: bool = Field(False, description="Whether to automatically integrate if safe")
    trust_threshold: float = Field(0.7, description="Minimum trust score to auto-integrate (0.0-1.0)")
    
    @validator('trust_threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Trust threshold must be between 0.0 and 1.0")
        return v


class IngestionResponse(BaseModel):
    """Response model for ingestion operations"""
    ingestion_id: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class IngestionResult(BaseModel):
    """Detailed result of an ingestion operation"""
    ingestion_id: str
    status: str
    name: str
    type: str
    files: List[str]
    trust_score: float
    utility_score: float
    issues: List[Dict[str, Any]] = []
    staged: bool = False
    integrated: bool = False
    fingerprint: str
    timestamp: datetime = Field(default_factory=datetime.now)


@dataclass
class SandboxResult:
    """Results from sandboxing an ingested component"""
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    memory_usage: float = 0.0
    execution_time: float = 0.0


@dataclass
class CodeFingerprint:
    """Fingerprint of a code component for comparison"""
    hash: str
    imports: Set[str]
    functions: Set[str]
    classes: Set[str]
    loc: int
    complexity: float


class IngestionPipeline:
    """Main pipeline for handling code ingestion and integration"""
    
    def __init__(self, grace_root: Path = None):
        """Initialize the ingestion pipeline
        
        Args:
            grace_root: Root directory of the Grace system
        """
        self.grace_root = grace_root or Path(__file__).parent.parent
        self.sandbox_dir = self.grace_root / "sandbox"
        self.staging_dir = self.grace_root / "staging"
        self.modules_dir = self.grace_root / "modules"
        
        # Ensure directories exist
        self.sandbox_dir.mkdir(exist_ok=True)
        self.staging_dir.mkdir(exist_ok=True)
        self.modules_dir.mkdir(exist_ok=True)
        
        # Track active ingestions
        self.active_ingestions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized ingestion pipeline at {self.grace_root}")
    
    async def process_upload(
        self, 
        file: UploadFile, 
        request: IngestionRequest
    ) -> IngestionResponse:
        """Process an uploaded file or archive
        
        Args:
            file: The uploaded file
            request: Ingestion request parameters
            
        Returns:
            Response with ingestion ID and initial status
        """
        ingestion_id = str(uuid.uuid4())
        
        # Create a unique sandbox directory for this ingestion
        sandbox_path = self.sandbox_dir / ingestion_id
        sandbox_path.mkdir(exist_ok=True)
        
        try:
            # Save the uploaded file
            file_path = sandbox_path / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Initialize ingestion tracking
            self.active_ingestions[ingestion_id] = {
                "id": ingestion_id,
                "name": request.name,
                "description": request.description,
                "status": IngestionStatus.PENDING,
                "file_path": str(file_path),
                "sandbox_path": str(sandbox_path),
                "replace_existing": request.replace_existing,
                "auto_integrate": request.auto_integrate,
                "trust_threshold": request.trust_threshold,
                "timestamp": datetime.now(),
                "result": None
            }
            
            logger.info(f"Received ingestion {ingestion_id}: {request.name} ({file.filename})")
            
            return IngestionResponse(
                ingestion_id=ingestion_id,
                status=IngestionStatus.PENDING.name,
                message=f"Ingestion {ingestion_id} started for {request.name}"
            )
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}", exc_info=True)
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process upload: {str(e)}"
            )
    
    async def run_ingestion_pipeline(self, ingestion_id: str) -> IngestionResult:
        """Run the full ingestion pipeline for a given ingestion ID
        
        Args:
            ingestion_id: The ID of the ingestion to process
            
        Returns:
            Result of the ingestion process
        """
        if ingestion_id not in self.active_ingestions:
            raise ValueError(f"Unknown ingestion ID: {ingestion_id}")
        
        ingestion = self.active_ingestions[ingestion_id]
        file_path = Path(ingestion["file_path"])
        sandbox_path = Path(ingestion["sandbox_path"])
        
        try:
            # Update status
            self._update_status(ingestion_id, IngestionStatus.VALIDATING)
            
            # Extract if it's a zip file
            ingestion_type = await self._extract_if_needed(file_path, sandbox_path)
            ingestion["type"] = ingestion_type
            
            # Validate syntax and structure
            validation_result = await self._validate_code(sandbox_path, ingestion_type)
            if not validation_result["success"]:
                return self._fail_ingestion(
                    ingestion_id, 
                    "Validation failed", 
                    validation_result["errors"]
                )
            
            # Update status
            self._update_status(ingestion_id, IngestionStatus.SANDBOXING)
            
            # Sandbox test
            sandbox_result = await self._sandbox_test(sandbox_path, ingestion_type)
            if not sandbox_result.success:
                return self._fail_ingestion(
                    ingestion_id, 
                    "Sandbox testing failed", 
                    sandbox_result.errors
                )
            
            # Update status
            self._update_status(ingestion_id, IngestionStatus.ANALYZING)
            
            # Generate fingerprint
            fingerprint = await self._generate_fingerprint(sandbox_path, ingestion_type)
            ingestion["fingerprint"] = fingerprint
            
            # Check for duplicates or conflicts
            conflict_check = await self._check_conflicts(fingerprint, ingestion["name"])
            if conflict_check["has_conflict"] and not ingestion["replace_existing"]:
                return self._fail_ingestion(
                    ingestion_id, 
                    "Conflicting module exists", 
                    [conflict_check["message"]]
                )
            
            # Update status
            self._update_status(ingestion_id, IngestionStatus.SCORING)
            
            # Score the ingestion
            scores = await self._score_ingestion(
                sandbox_path, 
                ingestion_type, 
                fingerprint, 
                sandbox_result
            )
            
            # Determine if it meets the threshold
            meets_threshold = scores["trust_score"] >= ingestion["trust_threshold"]
            
            # Stage the ingestion
            self._update_status(ingestion_id, IngestionStatus.STAGING)
            staging_path = await self._stage_ingestion(
                ingestion_id, 
                sandbox_path, 
                ingestion["name"], 
                ingestion_type
            )
            
            # Auto-integrate if requested and meets threshold
            integrated = False
            if ingestion["auto_integrate"] and meets_threshold:
                self._update_status(ingestion_id, IngestionStatus.INTEGRATING)
                integrated = await self._integrate_ingestion(
                    staging_path, 
                    ingestion["name"], 
                    ingestion_type,
                    ingestion["replace_existing"]
                )
            
            # Create result
            result = IngestionResult(
                ingestion_id=ingestion_id,
                status=IngestionStatus.COMPLETED.name,
                name=ingestion["name"],
                type=ingestion_type.name,
                files=self._list_files(sandbox_path),
                trust_score=scores["trust_score"],
                utility_score=scores["utility_score"],
                issues=scores["issues"],
                staged=True,
                integrated=integrated,
                fingerprint=fingerprint.hash
            )
            
            # Update ingestion record
            self._update_status(ingestion_id, IngestionStatus.COMPLETED, result)
            
            logger.info(f"Completed ingestion {ingestion_id} with trust score {scores['trust_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {str(e)}", exc_info=True)
            return self._fail_ingestion(ingestion_id, f"Pipeline error: {str(e)}")
    
    async def _extract_if_needed(self, file_path: Path, sandbox_path: Path) -> IngestionType:
        """Extract zip files if needed and determine ingestion type
        
        Args:
            file_path: Path to the uploaded file
            sandbox_path: Path to the sandbox directory
            
        Returns:
            Type of the ingestion
        """
        if file_path.suffix.lower() == '.zip':
            # Extract zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(sandbox_path)
            
            # Determine if it's a package or module
            if (sandbox_path / '__init__.py').exists():
                return IngestionType.PACKAGE
            else:
                return IngestionType.MODULE
        else:
            # Single Python file
            return IngestionType.SINGLE_FILE
    
    async def _validate_code(self, path: Path, ingestion_type: IngestionType) -> Dict[str, Any]:
        """Validate Python code syntax and structure
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Get list of Python files to check
        if ingestion_type == IngestionType.SINGLE_FILE:
            py_files = [path] if path.suffix.lower() == '.py' else [path / f for f in os.listdir(path) if f.endswith('.py')]
        else:
            py_files = list(path.glob('**/*.py'))
        
        if not py_files:
            return {
                "success": False,
                "errors": ["No Python files found in the upload"]
            }
        
        # Check each file for syntax errors
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Parse the AST to check for syntax errors
                try:
                    tree = ast.parse(code)
                    
                    # Check for potentially dangerous operations
                    visitor = SecurityVisitor()
                    visitor.visit(tree)
                    
                    if visitor.issues:
                        for issue in visitor.issues:
                            warnings.append(f"{py_file.name}: {issue}")
                    
                except SyntaxError as e:
                    errors.append(f"Syntax error in {py_file.name}: {str(e)}")
            
            except Exception as e:
                errors.append(f"Error reading {py_file.name}: {str(e)}")
        
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _sandbox_test(self, path: Path, ingestion_type: IngestionType) -> SandboxResult:
        """Test the code in a sandbox environment
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
            Results of sandbox testing
        """
        import resource
        import time
        
        # Create a temporary directory for isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy the files to the temp directory
            if ingestion_type == IngestionType.SINGLE_FILE:
                if path.is_file():
                    shutil.copy(path, temp_path / path.name)
                else:
                    for py_file in path.glob('*.py'):
                        shutil.copy(py_file, temp_path / py_file.name)
            else:
                shutil.copytree(path, temp_path / path.name, dirs_exist_ok=True)
            
                        # Prepare for import analysis
            errors = []
            warnings = []
            imports = set()
            functions = set()
            classes = set()
            
            # Track memory and execution time
            start_time = time.time()
            start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            
            try:
                # Find the main module to import
                if ingestion_type == IngestionType.SINGLE_FILE:
                    py_files = list(temp_path.glob('*.py'))
                    if not py_files:
                        return SandboxResult(success=False, errors=["No Python files found"])
                    
                    # Import the module
                    module_path = py_files[0]
                    module_name = module_path.stem
                    
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None:
                        return SandboxResult(success=False, errors=[f"Failed to create module spec for {module_name}"])
                    
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add to sys.modules temporarily
                    sys.modules[module_name] = module
                    
                    try:
                        # Execute the module in a controlled environment
                        spec.loader.exec_module(module)
                        
                        # Analyze the module
                        for name, obj in module.__dict__.items():
                            if name.startswith('__'):
                                continue
                                
                            if callable(obj):
                                if isinstance(obj, type):
                                    classes.add(f"{module_name}.{name}")
                                else:
                                    functions.add(f"{module_name}.{name}")
                        
                        # Extract imports
                        with open(module_path, 'r') as f:
                            tree = ast.parse(f.read())
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for name in node.names:
                                        imports.add(name.name)
                                elif isinstance(node, ast.ImportFrom):
                                    if node.module:
                                        imports.add(node.module)
                    
                    finally:
                        # Clean up
                        if module_name in sys.modules:
                            del sys.modules[module_name]
                
                else:  # MODULE or PACKAGE
                    package_path = temp_path / path.name
                    if not package_path.exists():
                        return SandboxResult(success=False, errors=[f"Package directory not found: {package_path}"])
                    
                    # Add to path temporarily
                    sys.path.insert(0, str(temp_path))
                    
                    try:
                        # Import the package
                        package_name = path.name
                        package = importlib.import_module(package_name)
                        
                        # Recursively analyze all modules in the package
                        for py_file in package_path.glob('**/*.py'):
                            rel_path = py_file.relative_to(package_path)
                            module_parts = [package_name]
                            
                            # Build the module name
                            if rel_path.parent != Path('.'):
                                module_parts.extend(rel_path.parent.parts)
                            
                            module_parts.append(rel_path.stem)
                            if module_parts[-1] == '__init__':
                                module_parts.pop()
                            
                            module_name = '.'.join(module_parts)
                            
                            try:
                                module = importlib.import_module(module_name)
                                
                                # Analyze the module
                                for name, obj in module.__dict__.items():
                                    if name.startswith('__'):
                                        continue
                                        
                                    if callable(obj):
                                        if isinstance(obj, type):
                                            classes.add(f"{module_name}.{name}")
                                        else:
                                            functions.add(f"{module_name}.{name}")
                                
                                # Extract imports
                                with open(py_file, 'r') as f:
                                    tree = ast.parse(f.read())
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.Import):
                                            for name in node.names:
                                                imports.add(name.name)
                                        elif isinstance(node, ast.ImportFrom):
                                            if node.module:
                                                imports.add(node.module)
                            
                            except Exception as e:
                                warnings.append(f"Error analyzing module {module_name}: {str(e)}")
                    
                    finally:
                        # Clean up
                        if sys.path and sys.path[0] == str(temp_path):
                            sys.path.pop(0)
                        
                        # Remove imported modules
                        for key in list(sys.modules.keys()):
                            if key == path.name or key.startswith(f"{path.name}."):
                                del sys.modules[key]
            
            except Exception as e:
                errors.append(f"Error in sandbox execution: {str(e)}")
            
            # Calculate resource usage
            end_time = time.time()
            end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            
            execution_time = end_time - start_time
            memory_usage = (end_mem - start_mem) / 1024  # Convert to MB
            
            return SandboxResult(
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                imports=imports,
                functions=functions,
                classes=classes,
                memory_usage=memory_usage,
                execution_time=execution_time
            )
    
    async def _generate_fingerprint(self, path: Path, ingestion_type: IngestionType) -> CodeFingerprint:
        """Generate a fingerprint for the code
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
            Code fingerprint
        """
        # Get all Python files
        if ingestion_type == IngestionType.SINGLE_FILE:
            if path.is_file():
                py_files = [path]
            else:
                py_files = list(path.glob('*.py'))
        else:
            py_files = list(path.glob('**/*.py'))
        
        # Initialize fingerprint data
        imports = set()
        functions = set()
        classes = set()
        loc = 0
        complexity = 0.0
        
        # Concatenate all file contents for hashing
        all_content = ""
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_content += content
                    
                    # Count lines of code
                    loc += len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
                    
                    # Parse AST
                    tree = ast.parse(content)
                    
                    # Extract imports
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                imports.add(name.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module)
                        elif isinstance(node, ast.FunctionDef):
                            functions.add(node.name)
                            
                            # Simple complexity metric: count branches
                            visitor = ComplexityVisitor()
                            visitor.visit(node)
                            complexity += visitor.complexity
                        elif isinstance(node, ast.ClassDef):
                            classes.add(node.name)
            
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {str(e)}")
        
        # Generate hash
        hash_value = hashlib.sha256(all_content.encode()).hexdigest()
        
        # Normalize complexity
        if len(functions) > 0:
            complexity /= len(functions)
        
        return CodeFingerprint(
            hash=hash_value,
            imports=imports,
            functions=functions,
            classes=classes,
            loc=loc,
            complexity=complexity
        )
    
    async def _check_conflicts(self, fingerprint: CodeFingerprint, name: str) -> Dict[str, Any]:
        """Check for conflicts with existing modules
        
        Args:
            fingerprint: Code fingerprint
            name: Module name
            
        Returns:
            Conflict check results
        """
        # Check if a module with the same name exists
        module_path = self.modules_dir / name
        if module_path.exists():
            # Generate fingerprint for the existing module
            existing_type = IngestionType.PACKAGE if (module_path / '__init__.py').exists() else IngestionType.MODULE
            if module_path.is_file():
                existing_type = IngestionType.SINGLE_FILE
            
            existing_fingerprint = await self._generate_fingerprint(module_path, existing_type)
            
            # Check if it's the same code
            if existing_fingerprint.hash == fingerprint.hash:
                return {
                    "has_conflict": True,
                    "duplicate": True,
                    "message": f"Module '{name}' already exists with identical code"
                }
            
            # Check for function/class name conflicts
            function_conflicts = fingerprint.functions.intersection(existing_fingerprint.functions)
            class_conflicts = fingerprint.classes.intersection(existing_fingerprint.classes)
            
            if function_conflicts or class_conflicts:
                return {
                    "has_conflict": True,
                    "duplicate": False,
                    "message": f"Module '{name}' has conflicting functions/classes: " +
                              f"{', '.join(function_conflicts.union(class_conflicts))}"
                }
            
            return {
                "has_conflict": True,
                "duplicate": False,
                "message": f"Module '{name}' already exists with different code"
            }
        
        return {
            "has_conflict": False
        }
    
    async def _score_ingestion(
        self, 
        path: Path, 
        ingestion_type: IngestionType,
        fingerprint: CodeFingerprint,
        sandbox_result: SandboxResult
    ) -> Dict[str, Any]:
        """Score the ingestion for trust and utility
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            fingerprint: Code fingerprint
            sandbox_result: Results from sandbox testing
            
        Returns:
            Scoring results
        """
        issues = []
        
        # Base trust score starts at 0.5
        trust_score = 0.5
        
        # Penalize for security warnings
        if sandbox_result.warnings:
            trust_score -= min(0.3, len(sandbox_result.warnings) * 0.05)
            issues.extend([{"type": "security", "message": warning} for warning in sandbox_result.warnings])
        
        # Check for suspicious imports
        suspicious_imports = {
            'os', 'subprocess', 'sys', 'shutil', 'importlib', 
            'pickle', 'marshal', 'base64', 'tempfile'
        }
        found_suspicious = sandbox_result.imports.intersection(suspicious_imports)
        if found_suspicious:
            trust_score -= min(0.3, len(found_suspicious) * 0.05)
            issues.append({
                "type": "suspicious_imports",
                "message": f"Uses potentially risky modules: {', '.join(found_suspicious)}"
            })
        
        # Reward for documentation
        doc_score = await self._evaluate_documentation(path, ingestion_type)
        trust_score += doc_score * 0.1
        
        # Reward for test coverage
        test_score = await self._evaluate_test_coverage(path, ingestion_type)
        trust_score += test_score * 0.1
        
        # Reward for code quality
        quality_score = await self._evaluate_code_quality(path, ingestion_type)
        trust_score += quality_score * 0.1
        
        # Utility score based on functionality and complexity
        utility_score = min(1.0, (
            len(sandbox_result.functions) * 0.02 +
            len(sandbox_result.classes) * 0.03 +
            fingerprint.complexity * 0.1
        ))
        
        # Penalize excessive resource usage
        if sandbox_result.memory_usage > 100:  # More than 100MB
            utility_score -= min(0.3, (sandbox_result.memory_usage - 100) / 1000)
            issues.append({
                "type": "resource_usage",
                "message": f"High memory usage: {sandbox_result.memory_usage:.2f}MB"
            })
        
        if sandbox_result.execution_time > 5:  # More than 5 seconds
            utility_score -= min(0.3, (sandbox_result.execution_time - 5) / 10)
            issues.append({
                "type": "performance",
                "message": f"Slow execution: {sandbox_result.execution_time:.2f}s"
            })
        
        # Ensure scores are in range [0, 1]
        trust_score = max(0.0, min(1.0, trust_score))
        utility_score = max(0.0, min(1.0, utility_score))
        
        return {
            "trust_score": trust_score,
            "utility_score": utility_score,
            "issues": issues
        }
    
    async def _evaluate_documentation(self, path: Path, ingestion_type: IngestionType) -> float:
        """Evaluate the quality of documentation
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
            Documentation score (0.0-1.0)
        """
        # Get Python files
        if ingestion_type == IngestionType.SINGLE_FILE:
            if path.is_file():
                py_files = [path]
            else:
                py_files = list(path.glob('*.py'))
        else:
            py_files = list(path.glob('**/*.py'))
        
        if not py_files:
            return 0.0
        
        total_score = 0.0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Count docstrings
                docstring_visitor = DocstringVisitor()
                docstring_visitor.visit(tree)
                
                # Calculate file score
                if docstring_visitor.total_nodes > 0:
                    file_score = docstring_visitor.docstring_nodes / docstring_visitor.total_nodes
                    total_score += file_score
            
            except Exception as e:
                logger.warning(f"Error evaluating documentation in {py_file}: {str(e)}")
        
        # Average across files
        return total_score / len(py_files) if py_files else 0.0
    
    async def _evaluate_test_coverage(self, path: Path, ingestion_type: IngestionType) -> float:
        """Evaluate test coverage
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
                        Test coverage score (0.0-1.0)
        """
        # Look for test files
        if ingestion_type == IngestionType.SINGLE_FILE:
            if path.is_file():
                # Check for a corresponding test file
                test_file = path.parent / f"test_{path.name}"
                if test_file.exists():
                    return 0.8  # Good score for having a test file
                return 0.0
            else:
                test_files = list(path.glob('test_*.py'))
        else:
            test_files = list(path.glob('**/test_*.py')) + list(path.glob('**/tests/*.py'))
        
        if not test_files:
            return 0.0
        
        # Simple heuristic: ratio of test files to implementation files
        py_files = list(path.glob('**/*.py')) if ingestion_type != IngestionType.SINGLE_FILE else list(path.glob('*.py'))
        impl_files = [f for f in py_files if not f.name.startswith('test_') and 'tests' not in f.parts]
        
        if not impl_files:
            return 0.0
        
        ratio = len(test_files) / len(impl_files)
        return min(1.0, ratio)
    
    async def _evaluate_code_quality(self, path: Path, ingestion_type: IngestionType) -> float:
        """Evaluate code quality
        
        Args:
            path: Path to the code
            ingestion_type: Type of ingestion
            
        Returns:
            Code quality score (0.0-1.0)
        """
        # Get Python files
        if ingestion_type == IngestionType.SINGLE_FILE:
            if path.is_file():
                py_files = [path]
            else:
                py_files = list(path.glob('*.py'))
        else:
            py_files = list(path.glob('**/*.py'))
        
        if not py_files:
            return 0.0
        
        total_score = 0.0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check line length
                lines = content.split('\n')
                long_lines = sum(1 for line in lines if len(line) > 100)
                line_score = 1.0 - (min(1.0, long_lines / len(lines) * 2) if lines else 0)
                
                # Check function length
                tree = ast.parse(content)
                function_visitor = FunctionLengthVisitor()
                function_visitor.visit(tree)
                
                func_score = 1.0
                if function_visitor.functions:
                    avg_length = sum(function_visitor.functions.values()) / len(function_visitor.functions)
                    func_score = 1.0 - min(1.0, (avg_length - 15) / 50) if avg_length > 15 else 1.0
                
                # Check for type hints
                type_hint_visitor = TypeHintVisitor()
                type_hint_visitor.visit(tree)
                
                type_score = 0.0
                if type_hint_visitor.total_annotations > 0:
                    type_score = type_hint_visitor.type_annotations / type_hint_visitor.total_annotations
                
                # Combine scores
                file_score = (line_score * 0.3) + (func_score * 0.4) + (type_score * 0.3)
                total_score += file_score
            
            except Exception as e:
                logger.warning(f"Error evaluating code quality in {py_file}: {str(e)}")
        
        # Average across files
        return total_score / len(py_files) if py_files else 0.0
    
    async def _stage_ingestion(
        self, 
        ingestion_id: str, 
        sandbox_path: Path, 
        name: str, 
        ingestion_type: IngestionType
    ) -> Path:
        """Stage the ingestion for potential integration
        
        Args:
            ingestion_id: Ingestion ID
            sandbox_path: Path to the sandbox directory
            name: Module name
            ingestion_type: Type of ingestion
            
        Returns:
            Path to the staged files
        """
        # Create staging directory
        staging_path = self.staging_dir / ingestion_id
        staging_path.mkdir(exist_ok=True)
        
        # Create module directory
        module_path = staging_path / name
        module_path.mkdir(exist_ok=True)
        
        # Copy files
        if ingestion_type == IngestionType.SINGLE_FILE:
            if sandbox_path.is_file():
                shutil.copy(sandbox_path, module_path / sandbox_path.name)
            else:
                for py_file in sandbox_path.glob('*.py'):
                    shutil.copy(py_file, module_path / py_file.name)
        else:
            # Copy all files from sandbox to staging
            source_path = sandbox_path
            if not (sandbox_path / '__init__.py').exists() and list(sandbox_path.glob('**/*.py')):
                # Find the directory containing the Python files
                for root, dirs, files in os.walk(sandbox_path):
                    if any(f.endswith('.py') for f in files):
                        source_path = Path(root)
                        break
            
            # Copy files
            for item in source_path.glob('**/*'):
                if item.is_file():
                    rel_path = item.relative_to(source_path)
                    dest_path = module_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(item, dest_path)
        
        # Create metadata file
        metadata = {
            "ingestion_id": ingestion_id,
            "name": name,
            "type": ingestion_type.name,
            "timestamp": datetime.now().isoformat(),
            "files": self._list_files(module_path)
        }
        
        with open(staging_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Staged ingestion {ingestion_id} at {staging_path}")
        return staging_path
    
    async def _integrate_ingestion(
        self, 
        staging_path: Path, 
        name: str, 
        ingestion_type: IngestionType,
        replace_existing: bool
    ) -> bool:
        """Integrate the staged ingestion into Grace
        
        Args:
            staging_path: Path to the staged files
            name: Module name
            ingestion_type: Type of ingestion
            replace_existing: Whether to replace existing modules
            
        Returns:
            Whether integration was successful
        """
        module_path = self.modules_dir / name
        staged_module_path = staging_path / name
        
        # Check if module already exists
        if module_path.exists():
            if not replace_existing:
                logger.warning(f"Module {name} already exists and replace_existing is False")
                return False
            
            # Backup existing module
            backup_path = self.modules_dir / f"{name}_backup_{int(datetime.now().timestamp())}"
            shutil.move(module_path, backup_path)
            logger.info(f"Backed up existing module {name} to {backup_path}")
        
        try:
            # Copy from staging to modules
            if staged_module_path.is_file():
                module_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(staged_module_path, module_path)
            else:
                shutil.copytree(staged_module_path, module_path)
            
            logger.info(f"Integrated module {name} into Grace")
            return True
        
        except Exception as e:
            logger.error(f"Failed to integrate module {name}: {str(e)}", exc_info=True)
            
            # Restore backup if available
            backup_path = next(self.modules_dir.glob(f"{name}_backup_*"), None)
            if backup_path:
                if module_path.exists():
                    shutil.rmtree(module_path)
                shutil.move(backup_path, module_path)
                logger.info(f"Restored backup of module {name}")
            
            return False
    
    def _update_status(
        self, 
        ingestion_id: str, 
        status: IngestionStatus, 
        result: IngestionResult = None
    ) -> None:
        """Update the status of an ingestion
        
        Args:
            ingestion_id: Ingestion ID
            status: New status
            result: Optional result object
        """
        if ingestion_id in self.active_ingestions:
            self.active_ingestions[ingestion_id]["status"] = status
            if result:
                self.active_ingestions[ingestion_id]["result"] = result
            
            logger.info(f"Ingestion {ingestion_id} status updated to {status.name}")
    
    def _fail_ingestion(
        self, 
        ingestion_id: str, 
        message: str, 
        errors: List[str] = None
    ) -> IngestionResult:
        """Mark an ingestion as failed
        
        Args:
            ingestion_id: Ingestion ID
            message: Failure message
            errors: List of error messages
            
        Returns:
            Failed ingestion result
        """
        ingestion = self.active_ingestions.get(ingestion_id, {})
        name = ingestion.get("name", "unknown")
        
        result = IngestionResult(
            ingestion_id=ingestion_id,
            status=IngestionStatus.FAILED.name,
            name=name,
            type=getattr(ingestion.get("type", IngestionType.SINGLE_FILE), "name", "UNKNOWN"),
            files=[],
            trust_score=0.0,
            utility_score=0.0,
            issues=[{"type": "error", "message": message}],
            staged=False,
            integrated=False,
            fingerprint=""
        )
        
        if errors:
            for error in errors:
                result.issues.append({"type": "error", "message": error})
        
        self._update_status(ingestion_id, IngestionStatus.FAILED, result)
        logger.error(f"Ingestion {ingestion_id} failed: {message}")
        
        return result
    
    def _list_files(self, path: Path) -> List[str]:
        """List all files in a directory recursively
        
        Args:
            path: Directory path
            
        Returns:
            List of file paths relative to the directory
        """
        if path.is_file():
            return [path.name]
        
        files = []
        for item in path.glob('**/*'):
            if item.is_file():
                files.append(str(item.relative_to(path)))
        
        return files
    
    def get_ingestion_status(self, ingestion_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an ingestion
        
        Args:
            ingestion_id: Ingestion ID
            
        Returns:
            Ingestion status or None if not found
        """
        return self.active_ingestions.get(ingestion_id)
    
        def list_ingestions(self) -> List[Dict[str, Any]]:
        """List all active ingestions
        
        Returns:
            List of ingestion status dictionaries
        """
        return list(self.active_ingestions.values())
    
    async def get_staged_ingestion(self, ingestion_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a staged ingestion
        
        Args:
            ingestion_id: Ingestion ID
            
        Returns:
            Staged ingestion details or None if not found
        """
        staging_path = self.staging_dir / ingestion_id
        metadata_path = staging_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error reading staged ingestion {ingestion_id}: {str(e)}")
            return None
    
    async def list_staged_ingestions(self) -> List[Dict[str, Any]]:
        """List all staged ingestions
        
        Returns:
            List of staged ingestion metadata
        """
        result = []
        
        for staging_dir in self.staging_dir.glob('*'):
            if not staging_dir.is_dir():
                continue
            
            metadata_path = staging_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                result.append(metadata)
            
            except Exception as e:
                logger.error(f"Error reading staged ingestion {staging_dir.name}: {str(e)}")
        
        return result
    
    async def manually_integrate(self, ingestion_id: str) -> Dict[str, Any]:
        """Manually integrate a staged ingestion
        
        Args:
            ingestion_id: Ingestion ID
            
        Returns:
            Integration result
        """
        # Get staged ingestion
        metadata = await self.get_staged_ingestion(ingestion_id)
        if not metadata:
            return {
                "success": False,
                "message": f"Staged ingestion {ingestion_id} not found"
            }
        
        # Get ingestion type
        try:
            ingestion_type = IngestionType[metadata["type"]]
        except (KeyError, ValueError):
            return {
                "success": False,
                "message": f"Invalid ingestion type: {metadata.get('type')}"
            }
        
        # Integrate
        staging_path = self.staging_dir / ingestion_id
        success = await self._integrate_ingestion(
            staging_path,
            metadata["name"],
            ingestion_type,
            True  # Force replace
        )
        
        if success:
            return {
                "success": True,
                "message": f"Successfully integrated {metadata['name']}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to integrate {metadata['name']}"
            }
    
    async def reject_ingestion(self, ingestion_id: str) -> Dict[str, Any]:
        """Reject a staged ingestion
        
        Args:
            ingestion_id: Ingestion ID
            
        Returns:
            Rejection result
        """
        # Get staged ingestion
        staging_path = self.staging_dir / ingestion_id
        if not staging_path.exists():
            return {
                "success": False,
                "message": f"Staged ingestion {ingestion_id} not found"
            }
        
        # Remove staged files
        try:
            shutil.rmtree(staging_path)
            
            # Update status if still active
            if ingestion_id in self.active_ingestions:
                self._update_status(ingestion_id, IngestionStatus.REJECTED)
            
            return {
                "success": True,
                "message": f"Rejected ingestion {ingestion_id}"
            }
        
        except Exception as e:
            logger.error(f"Error rejecting ingestion {ingestion_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to reject ingestion: {str(e)}"
            }


# AST Visitors for code analysis
class SecurityVisitor(ast.NodeVisitor):
    """AST visitor to detect potentially unsafe code patterns"""
    
    def __init__(self):
        self.issues = []
    
    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            dangerous_funcs = {
                'eval': 'Uses eval() which can execute arbitrary code',
                'exec': 'Uses exec() which can execute arbitrary code',
                'os.system': 'Uses os.system() which executes shell commands',
                'subprocess.call': 'Uses subprocess which executes external commands',
                'subprocess.Popen': 'Uses subprocess which executes external commands',
                '__import__': 'Uses dynamic imports which can load arbitrary modules'
            }
            
            func_name = node.func.id
            if func_name in dangerous_funcs:
                self.issues.append(dangerous_funcs[func_name])
        
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Check for module.function patterns
                if node.func.value.id == 'os' and node.func.attr in ['system', 'popen', 'spawn']:
                    self.issues.append(f"Uses os.{node.func.attr}() which executes shell commands")
                
                elif node.func.value.id == 'subprocess' and node.func.attr in ['call', 'Popen', 'run']:
                    self.issues.append(f"Uses subprocess.{node.func.attr}() which executes external commands")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        # Check for potentially dangerous imports
        dangerous_imports = {
            'pickle': 'Uses pickle module which can execute arbitrary code during deserialization',
            'marshal': 'Uses marshal module which is not secure for untrusted data',
            'shelve': 'Uses shelve module which depends on pickle'
        }
        
        for name in node.names:
            if name.name in dangerous_imports:
                self.issues.append(dangerous_imports[name.name])
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        # Check for potentially dangerous imports
        if node.module in ['pickle', 'marshal', 'shelve']:
            self.issues.append(f"Imports from {node.module} which may not be secure")
        
        self.generic_visit(node)


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate code complexity"""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        self.complexity += 1 + len(node.handlers)
        self.generic_visit(node)


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to check for docstrings"""
    
    def __init__(self):
        self.docstring_nodes = 0
        self.total_nodes = 0
    
    def visit_Module(self, node):
        self.total_nodes += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.total_nodes += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.total_nodes += 1
        if ast.get_docstring(node):
            self.docstring_nodes += 1
        self.generic_visit(node)


class FunctionLengthVisitor(ast.NodeVisitor):
    """AST visitor to check function lengths"""
    
    def __init__(self):
        self.functions = {}
    
    def visit_FunctionDef(self, node):
        # Count the number of lines in the function
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            length = node.end_lineno - node.lineno
            self.functions[node.name] = length
        self.generic_visit(node)


class TypeHintVisitor(ast.NodeVisitor):
    """AST visitor to check for type hints"""
    
    def __init__(self):
        self.type_annotations = 0
        self.total_annotations = 0
    
    def visit_FunctionDef(self, node):
        # Check return type annotation
        if node.returns:
            self.total_annotations += 1
            self.type_annotations += 1
        
        # Check argument type annotations
        for arg in node.args.args:
            self.total_annotations += 1
            if arg.annotation:
                self.type_annotations += 1
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        # Check variable type annotations
        self.total_annotations += 1
        self.type_annotations += 1
        self.generic_visit(node)


# FastAPI router for the ingestion API
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize the ingestion pipeline
pipeline = IngestionPipeline()


@router.post("/upload", response_model=IngestionResponse)
async def upload_code(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    replace_existing: bool = Form(False),
    auto_integrate: bool = Form(False),
    trust_threshold: float = Form(0.7)
):
    """Upload code for ingestion
    
    Args:
        file: Python file or zip archive
        name: Name for the module
        description: Description of functionality
        replace_existing: Whether to replace existing modules
        auto_integrate: Whether to automatically integrate if safe
        trust_threshold: Minimum trust score to auto-integrate (0.0-1.0)
        
    Returns:
        Ingestion response with ID and status
    """
    # Validate file type
    if not file.filename.endswith(('.py', '.zip')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Python files (.py) or zip archives (.zip) are supported"
        )
    
    # Create ingestion request
    request = IngestionRequest(
        name=name,
        description=description,
        replace_existing=replace_existing,
        auto_integrate=auto_integrate,
        trust_threshold=trust_threshold
    )
    
    # Process upload
    response = await pipeline.process_upload(file, request)
    
    # Start ingestion pipeline in background
    background_tasks.add_task(pipeline.run_ingestion_pipeline, response.ingestion_id)
    
    return response


@router.get("/status/{ingestion_id}", response_model=IngestionResult)
async def get_ingestion_status(ingestion_id: str):
    """Get the status of an ingestion
    
    Args:
        ingestion_id: Ingestion ID
        
    Returns:
        Ingestion result
    """
    ingestion = pipeline.get_ingestion_status(ingestion_id)
    if not ingestion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion {ingestion_id} not found"
        )
    
    if ingestion["result"]:
        return ingestion["result"]
    
    # Return in-progress status
    return IngestionResult(
        ingestion_id=ingestion_id,
        status=ingestion["status"].name,
        name=ingestion["name"],
        type=getattr(ingestion.get("type", IngestionType.SINGLE_FILE), "name", "UNKNOWN"),
        files=[],
        trust_score=0.0,
        utility_score=0.0,
        staged=False,
        integrated=False,
        fingerprint=""
    )


@router.get("/list", response_model=List[IngestionResult])
async def list_ingestions():
    """List all active ingestions
    
    Returns:
        List of ingestion results
    """
    ingestions = pipeline.list_ingestions()
    results = []
    
    for ingestion in ingestions:
        if ingestion.get("result"):
            results.append(ingestion["result"])
        else:
            results.append(IngestionResult(
                ingestion_id=ingestion["id"],
                status=ingestion["status"].name,
                name=ingestion["name"],
                type=getattr(ingestion.get("type", IngestionType.SINGLE_FILE), "name", "UNKNOWN"),
                files=[],
                trust_score=0.0,
                utility_score=0.0,
                staged=False,
                integrated=False,
                fingerprint=""
            ))
    
    return results


@router.get("/staged", response_model=List[Dict[str, Any]])
async def list_staged_ingestions():
    """List all staged ingestions
    
    Returns:
        List of staged ingestion metadata
    """
    return await pipeline.list_staged_ingestions()


@router.post("/integrate/{ingestion_id}", response_model=Dict[str, Any])
async def manually_integrate(ingestion_id: str):
    """Manually integrate a staged ingestion
    
    Args:
        ingestion_id: Ingestion ID
        
    Returns:
        Integration result
    """
    result = await pipeline.manually_integrate(ingestion_id)
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result


@router.post("/reject/{ingestion_id}", response_model=Dict[str, Any])
async def reject_ingestion(ingestion_id: str):
    """Reject a staged ingestion
    
    Args:
        ingestion_id: Ingestion ID
        
    Returns:
        Rejection result
    """
    result = await pipeline.reject_ingestion(ingestion_id)
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result



