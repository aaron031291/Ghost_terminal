#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraceResponse module for handling standardized response objects.

This module provides a consistent response structure for API calls,
function returns, and system operations with status tracking and metadata.
"""

import time
from typing import Any, Dict, Optional


class GraceResponse:
    """
    A standardized response object for system operations.
    
    This class encapsulates the result of an operation with status information,
    processing metadata, and a consistent interface for success, error, and
    other response types.
    
    Attributes:
        status (str): Response status (e.g., "success", "error", "recall", "unknown")
        signal (str): The original input signal that triggered the response
        result (Any): The output data or message
        fingerprint (str): Context identifier for tracking and debugging
        processing_time (float): Time taken to process the request in seconds
        metadata (Dict[str, Any]): Additional contextual information
    """

    def __init__(
        self,
        status: str,
        signal: str,
        result: Any,
        fingerprint: str,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new GraceResponse instance.
        
        Args:
            status: Response status (e.g., "success", "error", "recall", "unknown")
            signal: The original input signal
            result: Output data or message
            fingerprint: Context identifier for tracking
            processing_time: Time taken to process the request in seconds
            metadata: Additional contextual information
        """
        self.status = status                # e.g. "success", "error", "recall", "unknown"
        self.signal = signal                # the original input signal
        self.result = result                # output or message
        self.fingerprint = fingerprint      # context fingerprint
        self.processing_time = processing_time
        self.metadata = metadata or {}
        
        # Add timestamp to metadata if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the response
        """
        return {
            "status": self.status,
            "signal": self.signal,
            "result": self.result,
            "fingerprint": self.fingerprint,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def success(cls, signal: str, result: Any, **kwargs) -> "GraceResponse":
        """
        Create a success response.
        
        Args:
            signal: The original input signal
            result: The successful output
            **kwargs: Additional parameters (fingerprint, processing_time, metadata)
            
        Returns:
            GraceResponse: A response object with "success" status
        """
        return cls(
            "success", 
            signal, 
            result, 
            kwargs.get("fingerprint", ""),
            kwargs.get("processing_time", 0.0), 
            kwargs.get("metadata")
        )

    @classmethod
    def recall(cls, signal: str, result: Any, **kwargs) -> "GraceResponse":
        """
        Create a recall response for operations requiring additional context.
        
        Args:
            signal: The original input signal
            result: Information about the recall
            **kwargs: Additional parameters (fingerprint, processing_time, metadata)
            
        Returns:
            GraceResponse: A response object with "recall" status
        """
        return cls(
            "recall", 
            signal, 
            result, 
            kwargs.get("fingerprint", ""),
            kwargs.get("processing_time", 0.0), 
            kwargs.get("metadata")
        )

    @classmethod
    def error(cls, signal: str, error_msg: str, **kwargs) -> "GraceResponse":
        """
        Create an error response.
        
        Args:
            signal: The original input signal
            error_msg: Description of the error
            **kwargs: Additional parameters (fingerprint, processing_time, metadata)
            
        Returns:
            GraceResponse: A response object with "error" status
        """
        return cls(
            "error", 
            signal, 
            error_msg, 
            kwargs.get("fingerprint", ""),
            kwargs.get("processing_time", 0.0), 
            kwargs.get("metadata")
        )

    @classmethod
    def unknown(cls, signal: str, **kwargs) -> "GraceResponse":
        """
        Create an unknown response for unrecognized signals.
        
        Args:
            signal: The unrecognized input signal
            **kwargs: Additional parameters (fingerprint, processing_time, metadata)
            
        Returns:
            GraceResponse: A response object with "unknown" status
        """
        return cls(
            "unknown", 
            signal, 
            f"No handler or memory for '{signal}'",
            kwargs.get("fingerprint", ""), 
            kwargs.get("processing_time", 0.0),
            kwargs.get("metadata")
        )


if __name__ == "__main__":
    # Example usage
    def process_request(request_data: str) -> GraceResponse:
        """Example function demonstrating GraceResponse usage."""
        start_time = time.time()
        
        try:
            # Simulate processing
            if not request_data:
                return GraceResponse.error(
                    request_data, 
                    "Empty request data",
                    processing_time=time.time() - start_time
                )
                
            if request_data == "need_more_info":
                return GraceResponse.recall(
                    request_data,
                    "Additional information required",
                    processing_time=time.time() - start_time
                )
                
            # Process successful request
            result = f"Processed: {request_data}"
            return GraceResponse.success(
                request_data,
                result,
                fingerprint=f"req-{hash(request_data)}",
                processing_time=time.time() - start_time,
                metadata={"request_length": len(request_data)}
            )
        except Exception as e:
            return GraceResponse.error(
                request_data,
                f"Processing error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    # Test the example function
    examples = ["test_request", "", "need_more_info", "another_request"]
    for example in examples:
        response = process_request(example)
        print(f"Request: '{example}'")
        print(f"Response: {response.to_dict()}")
        print("-" * 50)
