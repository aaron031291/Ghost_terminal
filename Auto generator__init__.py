"""Grace: Generative Reasoning and Coding Engine.

Grace is a framework for generative reasoning and coding that provides:
1. Code Execution Kernel: Safe execution of code in various languages
2. Self-Learning Loop: Continuous improvement through feedback
3. Universal Export Layer: Export projects to various destinations
4. Teach-and-Train Interface: Learn from various document types
"""

from grace.grace import Grace
from grace.execution import CodeExecutionKernel, ExecutionResult
from grace.learning import SelfLearningLoop, LearningResult
from grace.export import UniversalExportLayer, ExportResult
from grace.training import TeachAndTrainInterface
from grace.config import Config

__version__ = "0.1.0"
