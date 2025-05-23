def generate_training_report(results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate training report.
    
    Args:
        results: Training results
        output_path: Path to save report. If None, report is not saved.
        
    Returns:
        Report dictionary
    """
    # Create report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_files": results["processed_files"] + results["failed_files"],
            "processed_files": results["processed_files"],
            "failed_files": results["failed_files"],
            "knowledge_items": results["knowledge_items"],
            "success_rate": results["processed_files"] / (results["processed_files"] + results["failed_files"]) * 100 if (results["processed_files"] + results["failed_files"]) > 0 else 0,
        },
        "errors": results["errors"],
        "file_types": {},
    }
    
    # Count file types
    for file_path in results.get("processed_file_paths", []):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in report["file_types"]:
            report["file_types"][ext] = 0
        report["file_types"][ext] += 1
    
    # Save report if output path is provided
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Training report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving training report: {str(e)}")
    
    return report
