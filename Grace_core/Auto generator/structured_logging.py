import structlog
from grace.config import Config

# Initialize structured logger
def setup_logger(config: Config):
    """Set up structured logger."""
    log_level = config.get("logging", "level")
    log_file = config.get("logging", "file")
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    
    logger = structlog.get_logger("grace.training")
    return logger
