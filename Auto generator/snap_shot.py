import datetime
import shutil

def create_snapshot(knowledge_base_path: str, config: Config) -> str:
    """Create a versioned snapshot of the knowledge base.
    
    Args:
        knowledge_base_path: Path to knowledge base file
        config: Configuration object
        
    Returns:
        Path to created snapshot
    """
    # Create snapshots directory if it doesn't exist
    snapshots_dir = config.get("training", "snapshots_dir", "training_history")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create snapshot filename
    snapshot_path = os.path.join(snapshots_dir, f"knowledge_base_{timestamp}.json")
    
    # Copy current knowledge base to snapshot
    shutil.copy2(knowledge_base_path, snapshot_path)
    
    logger.info(f"Created knowledge base snapshot: {snapshot_path}")
    return snapshot_path
