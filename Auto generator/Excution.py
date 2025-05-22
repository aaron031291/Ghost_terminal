        return ExecutionResult(
            success=success,
            output=stdout_capture.getvalue(),
            error=stderr_capture.getvalue(),
            execution_time=execution_time,
            memory_usage=memory_usage,
            return_value=return_value,
        )
        
    def execute(self, code: str, language: str = "python", mode: Optional[str] = None) -> ExecutionResult:
        """Execute code in the specified language.
        
        Args:
            code: Code to execute
            language: Programming language of the code
            mode: Execution mode (subprocess or inline). If None, uses config setting.
            
        Returns:
            ExecutionResult object with execution details
        """
        if language.lower() != "python":
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}",
                execution_time=0.0,
                memory_usage=0.0,
            )
            
        # Determine execution mode
        execution_mode = mode or self.sandbox_mode
        
        if execution_mode == "strict":
            # Use subprocess for maximum isolation
            return self.execute_python_subprocess(code)
        elif execution_mode == "inline":
            # Use inline execution for speed
            return self.execute_python_inline(code)
        else:
            logger.warning(f"Unknown execution mode: {execution_mode}, falling back to strict")
            return self.execute_python_subprocess(code)
