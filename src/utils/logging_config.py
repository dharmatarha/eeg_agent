import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Initializes the root logger with a standardized configuration:
    - Log Level is read from environment variable LOG_LEVEL (default: INFO)
    - Console Output is written to stdout
    - Optional File Output is written to logs/eeg_agent.log (default: True)
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Standard format for all application logs
    log_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = []
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # 2. Rotating File Handler
    log_to_file = os.environ.get("LOG_TO_FILE", "True").lower() in ("true", "1", "yes")
    if log_to_file:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, "eeg_agent.log")
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB max size per log file
            backupCount=5,              # Keep up to 5 rotated log files
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
        
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Suppress verbose third-party logging unless in DEBUG mode
    if log_level > logging.DEBUG:
        noisy_loggers = [
            "urllib3",
            "websocket",
            "chromadb",
            "httpcore",
            "httpx",
            "openai",
            "langchain",
            "langchain_core",
            "sentence_transformers"
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger("eeg_agent").info("Logging system initialized (Level: %s).", log_level_str)
