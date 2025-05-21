                    f"Healing attempt {current_result['healing_attempts']} failed for "
                    f"{current_result['module_name']}"
                )
        except Exception as e:
            current_result["errors"].append(f"Healing error: {str(e)}")
            current_result["status"] = "failed_healing"
            logger.error(f"Error during healing: {str(e)}")
        
        return current_result
    
    def _calculate_trust_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate a trust score for the module based on validation results.
        
        Args:
            result: Validation result dictionary
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        # Base score
        score = 1.0
        
        # Deduct for errors
        score -= len(result["errors"]) * 0.1
        
        # Deduct for warnings
        score -= len(result["warnings"]) * 0.05
        
        # Deduct for healing attempts
        score -= result["healing_attempts"] * 0.1
        
        # Check contributor trust from ledger
        contributor_trust = self.trust_ledger.get_contributor_trust(result["contributor_id"])
        if contributor_trust is not None:
            # Weight contributor trust as 30% of the score
            score = 0.7 * score + 0.3 * contributor_trust
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _generate_fingerprint(self, module_path: str) -> str:
        """
        Generate a unique fingerprint for the module.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Fingerprint string
        """
        try:
            with open(module_path, 'rb') as f:
                content = f.read()
            
            # Create SHA-256 hash of the content
            fingerprint = hashlib.sha256(content).hexdigest()
            return fingerprint
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            return "error-generating-fingerprint"
    
    def _log_to_trust_ledger(self, result: Dict[str, Any]) -> None:
        """
        Log validation result to the trust ledger.
        
        Args:
            result: Validation result dictionary
        """
        try:
            entry = {
                "validation_id": result["validation_id"],
                "module_name": result["module_name"],
                "contributor_id": result["contributor_id"],
                "fingerprint": result["fingerprint"],
                "timestamp": result["timestamp"],
                "status": result["status"],
                "error_count": len(result["errors"]),
                "warning_count": len(result["warnings"]),
                "healing_attempts": result["healing_attempts"],
                "trust_score": result["trust_score"]
            }
            
            self.trust_ledger.add_entry(entry)
            logger.info(f"Logged validation result to trust ledger: {result['validation_id']}")
        except Exception as e:
            logger.error(f"Error logging to trust ledger: {str(e)}")
    
    def _move_to_staging(self, module_path: str, validation_id: str) -> str:
        """
        Move a validated module to the staging vault.
        
        Args:
            module_path: Path to the module file
            validation_id: Unique validation ID
            
        Returns:
            Path to the staged module
        """
        if not self.staging_vault_path:
            raise ValueError("Staging vault path not configured")
        
        module_name = os.path.basename(module_path)
        staged_name = f"{validation_id}_{module_name}"
        staged_path = os.path.join(self.staging_vault_path, staged_name)
        
        # Copy to staging vault
        with open(module_path, 'rb') as src, open(staged_path, 'wb') as dst:
            dst.write(src.read())
        
        logger.info(f"Moved module {module_name} to staging at {staged_path}")
        return staged_path
    
    def _register_module(self, module_path: str, validation_id: str) -> str:
        """
        Register a validated module with the module registry.
        
        Args:
            module_path: Path to the module file
            validation_id: Unique validation ID
            
        Returns:
            Path to the registered module
        """
        module_name = os.path.basename(module_path)
        
        # Register with the module registry
        registry_result = self.module_registry.register_module(
            module_path,
            metadata={
                "validation_id": validation_id,
                "pre_entry_timestamp": datetime.now().isoformat()
            }
        )
        
        # Notify central intelligence
        self.central_intelligence.notify_new_module(registry_result["module_id"])
        
        logger.info(f"Registered module {module_name} with ID {registry_result['module_id']}")
        return registry_result["registry_path"]


class PreEntryAPI:
    """
    API interface for the PreEntryValidator.
    
    Provides HTTP endpoints for module validation and registration.
    """
    
    def __init__(self, validator: PreEntryValidator = None, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize the PreEntryAPI.
        
        Args:
            validator: PreEntryValidator instance
            host: Host to bind the API server
            port: Port to bind the API server
        """
        self.validator = validator or PreEntryValidator()
        self.host = host
        self.port = port
        self.app = Flask("pre_entry_api")
        
        # Register routes
        self._register_routes()
        
        logger.info(f"PreEntryAPI initialized on {host}:{port}")
    
    def _register_routes(self) -> None:
        """Register API routes."""
        
        @self.app.route('/validate', methods=['POST'])
        def validate_module():
            """Endpoint to validate a module."""
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Get contributor ID from request
            data = request.form.to_dict()
            contributor_id = data.get('contributor_id', 'anonymous')
            
            # Save file to temporary location
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            try:
                # Validate the module
                result = self.validator.validate_module(temp_path, contributor_id)
                
                # Clean up temporary file if it was registered or staged
                if result["status"] in ["accepted", "staged"]:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in validate_module endpoint: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
    
    def start(self) -> None:
        """Start the API server."""
        self.app.run(host=self.host, port=self.port)


class BatchValidator:
    """
    Batch validation utility for processing multiple modules.
    
    Useful for initial system setup or bulk imports.
    """
    
    def __init__(self, validator: PreEntryValidator = None):
        """
        Initialize the BatchValidator.
        
        Args:
            validator: PreEntryValidator instance
        """
        self.validator = validator or PreEntryValidator()
        logger.info("BatchValidator initialized")
    
    def validate_directory(self, 
                          directory_path: str, 
                          contributor_id: str = "system",
                          recursive: bool = False) -> Dict[str, Any]:
        """
        Validate all Python modules in a directory.
        
        Args:
            directory_path: Path to the directory containing modules
            contributor_id: ID of the contributor
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary with validation results
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        logger.info(f"Starting batch validation of directory: {directory_path}")
        
        results = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "healed": 0,
            "failed": 0,
            "modules": []
        }
        
        # Find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        for module_path in Path(directory_path).glob(pattern):
            results["total"] += 1
            
            # Validate the module
            try:
                result = self.validator.validate_module(str(module_path), contributor_id)
                results["modules"].append(result)
                
                # Update counters
                if result["status"] == "accepted":
                    results["accepted"] += 1
                elif result["status"] == "healed":
                    results["healed"] += 1
                elif result["status"].startswith("failed_"):
                    results["failed"] += 1
                else:
                    results["rejected"] += 1
                    
            except Exception as e:
                logger.error(f"Error validating {module_path}: {str(e)}")
                results["failed"] += 1
                results["modules"].append({
                    "module_name": os.path.basename(str(module_path)),
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"Batch validation completed: {results['accepted']} accepted, "
                   f"{results['healed']} healed, {results['rejected']} rejected, "
                   f"{results['failed']} failed")
        
        return results


def main():
    """Main entry point for running the pre-entry validator as a standalone service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grace Pre-Entry Validator")
    parser.add_argument("--registry", help="Path to module registry", default=None)
    parser.add_argument("--staging", help="Path to staging vault", default=None)
    parser.add_argument("--host", help="API host", default="0.0.0.0")
    parser.add_argument("--port", help="API port", type=int, default=5000)
    parser.add_argument("--batch", help="Batch validate a directory", default=None)
    parser.add_argument("--recursive", help="Recursive directory search", action="store_true")
    
    args = parser.parse_args()
    
    # Create validator
    validator = PreEntryValidator(
        registry_path=args.registry,
        staging_vault_path=args.staging
    )
    
    # Run in batch mode if specified
    if args.batch:
        batch_validator = BatchValidator(validator)
        results = batch_validator.validate_directory(
            args.batch, 
            recursive=args.recursive
        )
        print(json.dumps(results, indent=2))
        return
    
    # Otherwise, start API server
    api = PreEntryAPI(validator, host=args.host, port=args.port)
    api.start()


if __name__ == "__main__":
    main()
