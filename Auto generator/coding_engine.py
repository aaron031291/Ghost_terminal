"""Grace: Generative Reasoning and Coding Engine.

This module implements the main Grace class that integrates all components.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

from grace.config import Config
from grace.execution import CodeExecutionKernel, ExecutionResult
from grace.learning import SelfLearningLoop, LearningResult
from grace.export import UniversalExportLayer, ExportResult
from grace.training import TeachAndTrainInterface

logger = logging.getLogger("grace")


class Grace:
    """Main Grace class integrating all components."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize Grace.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        # Initialize configuration
        self.config = Config(config_path)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.execution_kernel = CodeExecutionKernel(self.config)
        self.learning_loop = SelfLearningLoop(self.config)
        self.export_layer = UniversalExportLayer(self.config)
        self.training_interface = TeachAndTrainInterface(self.config)
        
        logger.info("Grace initialized")
        
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config.get("logging", "level")
        log_file = self.config.get("logging", "file")
        
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            ],
        )
        
    def execute_code(self, code: str, language: str = "python", mode: Optional[str] = None) -> ExecutionResult:
        """Execute code.
        
        Args:
            code: Code to execute
            language: Programming language of the code
            mode: Execution mode (subprocess or inline). If None, uses config setting.
            
        Returns:
            ExecutionResult with execution details
        """
        logger.info(f"Executing {language} code")
        return self.execution_kernel.execute(code, language, mode)
        
    def learn_from_code(self, code: str, language: str = "python") -> LearningResult:
        """Improve code through self-learning.
        
        Args:
            code: Code to improve
            language: Programming language of the code
            
        Returns:
            LearningResult with improvement details
        """
        logger.info(f"Learning from {language} code")
        return self.learning_loop.learn(code, language)
        
    def export_project(
        self,
        project_path: str,
        destination: str,
        **kwargs,
    ) -> ExportResult:
        """Export project.
        
        Args:
            project_path: Path to project directory
            destination: Destination type (github, zip, docker, vercel)
            **kwargs: Additional arguments for specific destination
            
        Returns:
            ExportResult with export details
        """
        logger.info(f"Exporting project to {destination}")
        return self.export_layer.export(project_path, destination, **kwargs)
        
    def teach(self, file_path: str) -> Dict[str, Any]:
        """Teach Grace from a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with teaching results
        """
        logger.info(f"Teaching from file: {file_path}")
        return self.training_interface.teach(file_path)
        
    def train(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Train Grace on a directory of files.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to recursively process subdirectories
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training on directory: {directory_path}")
        return self.training_interface.train_on_directory(directory_path, recursive)
        
    def save_knowledge(self, output_path: str) -> bool:
        """Save knowledge base.
        
        Args:
            output_path: Path to save knowledge base
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Saving knowledge base to {output_path}")
        return self.training_interface.save_knowledge_base(output_path)
        
    def load_knowledge(self, input_path: str) -> bool:
        """Load knowledge base.
        
        Args:
            input_path: Path to load knowledge base from
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading knowledge base from {input_path}")
        return self.training_interface.load_knowledge_base(input_path)
        
    def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of matching knowledge items
        """
        logger.info(f"Querying knowledge base: {query}")
        return self.training_interface.query_knowledge(query)
        
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base.
        
        Returns:
            Dictionary with knowledge base summary
        """
        logger.info("Getting knowledge base summary")
        return self.training_interface.get_knowledge_summary()
        
    def get_version(self) -> str:
        """Get Grace version.
        
        Returns:
            Version string
        """
        return "0.1.0"  # Initial version
