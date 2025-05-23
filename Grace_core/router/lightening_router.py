#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Router Module

This module provides high-performance routing capabilities for the Grace system,
directing signals to appropriate handlers and managing memory recall.

Copyright (c) 2023 Grace AI Systems
License: MIT
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime

from grace_response import GraceResponse
from memory.lightning_fingerprint import generate_fingerprint
from memory.memory_index import LightningMemoryIndex
from memory.fusion_memory import FusionMemory
from grace_modules import (
    training_module,
    logging_module,
    status_module,
    execution_module
)
from utils.timestamp import current_timestamp
from utils.exceptions import HandlerExecutionError, MemoryRecallError
from utils.grace_status import GraceStatus
from utils.webhook_manager import WebhookManager


# Configure module logger
logger = logging.getLogger(__name__)


class LightningRouter:
    """
    High-performance router for Grace system signals.
    
    This class handles the routing of commands and queries to appropriate
    handlers, while maintaining a memory index for rapid recall of previous
    interactions.
    
    Attributes:
        command_routes: Mapping of command signals to their handler functions
        memory_index: Index for storing and retrieving previous interactions
        webhook_enabled: Flag indicating if webhook notifications are enabled
        webhook_manager: Manager for webhook connections and notifications
    """
    
    def __init__(
        self, 
        custom_routes: Optional[Dict[str, Callable]] = None,
        webhook_enabled: bool = False,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize the Lightning Router with command routes and memory index.
        
        Args:
            custom_routes: Optional dictionary of additional command routes to register
            webhook_enabled: Whether to enable webhook notifications
            webhook_url: URL for webhook notifications (required if webhooks enabled)
        """
        logger.info("Initializing LightningRouter")
        
        # Core command routing table
        self.command_routes: Dict[str, Callable] = {
            "train": training_module.run,
            "log": logging_module.capture,
            "status": status_module.report,
            "execute": execution_module.run_logic
        }
        
        # Add any custom routes
        if custom_routes:
            self.command_routes.update(custom_routes)
            
        # Core memory index for rapid recall
        self.memory_index = LightningMemoryIndex()
        
        # Webhook configuration
        self.webhook_enabled = webhook_enabled
        if webhook_enabled:
            if not webhook_url:
                raise ValueError("webhook_url must be provided when webhook_enabled is True")
            self.webhook_manager = WebhookManager(webhook_url)
            logger.info(f"Webhook notifications enabled: {webhook_url}")
        
        logger.debug(f"Registered {len(self.command_routes)} command routes")

    def register_command(self, signal: str, handler: Callable) -> None:
        """
        Register a new command handler.
        
        Args:
            signal: The command signal to register
            handler: The function to handle this command
            
        Raises:
            ValueError: If the signal is already registered
        """
        if signal in self.command_routes:
            raise ValueError(f"Command '{signal}' is already registered")
            
        self.command_routes[signal] = handler
        logger.info(f"Registered new command handler for '{signal}'")

    def push_to_socket(self, fingerprint: str, result: Any, status: str) -> bool:
        """
        Push results to configured webhook endpoint.
        
        Args:
            fingerprint: The unique fingerprint of the request
            result: The result data to push
            status: The status of the operation
            
        Returns:
            bool: True if push was successful, False otherwise
        """
        try:
            return self.webhook_manager.send_notification({
                "fingerprint": fingerprint,
                "result": result,
                "status": status,
                "timestamp": current_timestamp()
            })
        except Exception as e:
            logger.error(f"Failed to push to webhook: {str(e)}", exc_info=True)
            return False

    def route_input(self, signal: str, payload: Dict[str, Any], source: str = "system") -> GraceResponse:
        """
        Main interface: accepts a signal from GUI or system.
        Determines if it's a command or a memory trigger.
        
        Args:
            signal: The command or trigger string
            payload: Dictionary containing the data for the command
            source: Origin of the signal, defaults to "system"
            
        Returns:
            GraceResponse: Response object with results or error information
            
        Raises:
            HandlerExecutionError: If the command handler raises an exception
            MemoryRecallError: If memory recall fails unexpectedly
        """
        start_time = time.time()
        fingerprint = generate_fingerprint(signal, payload, source)
        timestamp = current_timestamp()
        
        # Create system metadata for context
        system_metadata = {
            "source": source,
            "timestamp": timestamp,
            "uptime": GraceStatus.get_uptime_status()
        }
        
        logger.debug(f"Routing input: signal='{signal}', source='{source}', fingerprint='{fingerprint}'")
        
        try:
            # Command Routing Path
            if signal in self.command_routes:
                logger.info(f"Executing command handler for '{signal}'")
                handler = self.command_routes[signal]
                
                try:
                    result = handler(payload)
                except Exception as handler_error:
                    logger.error(f"Handler execution failed: {str(handler_error)}", exc_info=True)
                    
                    # Log failure to FusionMemory
                    FusionMemory.log_result(fingerprint, str(handler_error), status="error")
                    
                    # Send webhook notification if enabled
                    if self.webhook_enabled:
                        self.push_to_socket(fingerprint, str(handler_error), "error")
                        
                    raise HandlerExecutionError(f"Failed to execute '{signal}': {str(handler_error)}") from handler_error
                
                # Store the interaction in memory
                self.memory_index.store_event(fingerprint, signal, payload, result, timestamp)
                
                # Log successful result to FusionMemory
                FusionMemory.log_result(fingerprint, result, status="success")
                
                # Send webhook notification if enabled
                if self.webhook_enabled:
                    self.push_to_socket(fingerprint, result, "success")
                
                processing_time = time.time() - start_time
                logger.info(f"Command '{signal}' processed in {processing_time:.3f}s")
                
                return GraceResponse.success(
                    signal, 
                    result, 
                    fingerprint=fingerprint,
                    processing_time=processing_time,
                    metadata=system_metadata
                )
                
            # Memory Recall Path (non-command)
            logger.info(f"Attempting memory recall for '{signal}'")
            try:
                recall = self.memory_index.recall(fingerprint)
            except Exception as recall_error:
                logger.error(f"Memory recall failed: {str(recall_error)}", exc_info=True)
                
                # Log failure to FusionMemory
                FusionMemory.log_result(fingerprint, str(recall_error), status="error")
                
                # Send webhook notification if enabled
                if self.webhook_enabled:
                    self.push_to_socket(fingerprint, str(recall_error), "error")
                    
                raise MemoryRecallError(f"Failed to recall memory: {str(recall_error)}") from recall_error
                
            if recall:
                processing_time = time.time() - start_time
                logger.info(f"Memory recalled for '{signal}' in {processing_time:.3f}s")
                
                # Log recall to FusionMemory
                FusionMemory.log_result(fingerprint, recall, status="recall")
                
                # Send webhook notification if enabled
                if self.webhook_enabled:
                    self.push_to_socket(fingerprint, recall, "recall")
                
                return GraceResponse.recall(
                    signal, 
                    recall, 
                    fingerprint=fingerprint,
                    processing_time=processing_time,
                    metadata=system_metadata
                )
            
            # No handler or memory match found
            processing_time = time.time() - start_time
            logger.warning(f"Unknown signal '{signal}' (processed in {processing_time:.3f}s)")
            
            # Log unknown signal to FusionMemory
            FusionMemory.log_result(fingerprint, f"Unknown signal: {signal}", status="unknown")
            
            # Send webhook notification if enabled
            if self.webhook_enabled:
                self.push_to_socket(fingerprint, f"Unknown signal: {signal}", "unknown")
            
            return GraceResponse.unknown(
                signal, 
                fingerprint=fingerprint,
                processing_time=processing_time,
                metadata=system_metadata
            )
            
        except (HandlerExecutionError, MemoryRecallError) as known_error:
            # Re-raise known errors for specific handling
            raise
            
        except Exception as e:
            # Catch-all for unexpected errors
            processing_time = time.time() - start_time
            logger.error(f"Unexpected error processing '{signal}': {str(e)}", exc_info=True)
            
            # Log error to FusionMemory
            FusionMemory.log_result(fingerprint, str(e), status="error")
            
            # Send webhook notification if enabled
            if self.webhook_enabled:
                self.push_to_socket(fingerprint, str(e), "error")
            
            return GraceResponse.error(
                signal, 
                str(e), 
                fingerprint=fingerprint,
                processing_time=processing_time,
                metadata=system_metadata
            )

    def clear_memory(self) -> int:
        """
        Clear the memory index.
        
        Returns:
            int: Number of entries cleared
        """
        count = self.memory_index.clear()
        logger.info(f"Cleared {count} entries from memory index")
        return count


# For direct script execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    router = LightningRouter(webhook_enabled=False)
    response = router.route_input("status", {"detail_level": "full"})
    print(f"Response: {response}")
