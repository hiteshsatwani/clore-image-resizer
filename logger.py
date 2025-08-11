"""Logging configuration for the image processing pipeline."""

import logging
import logging.handlers
import structlog
import json
import os
from datetime import datetime
from pathlib import Path
from config import config

# Global log storage for shutdown saving
log_entries = []

class LogCapture:
    """Captures log entries for saving on shutdown."""
    
    def __init__(self):
        self.entries = []
    
    def add_entry(self, entry):
        """Add a log entry."""
        self.entries.append({
            'timestamp': datetime.now().isoformat(),
            'entry': entry
        })
        # Keep only last 1000 entries to prevent memory issues
        if len(self.entries) > 1000:
            self.entries = self.entries[-1000:]
    
    def save_to_file(self, filepath=None):
        """Save captured logs to file."""
        if not filepath:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = logs_dir / f"pipeline_shutdown_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'shutdown_time': datetime.now().isoformat(),
                    'total_entries': len(self.entries),
                    'logs': self.entries
                }, f, indent=2)
            print(f"📝 Logs saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")
            return None

# Global log capture instance
log_capture = LogCapture()

class CustomJSONRenderer:
    """Custom JSON renderer that captures logs."""
    
    def __call__(self, _, __, event_dict):
        # Capture the log entry
        log_capture.add_entry(event_dict)
        # Return JSON formatted string
        return json.dumps(event_dict, default=str)

def setup_logging():
    """Configure structured logging with file output."""
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            CustomJSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging with file handler
    log_formatter = logging.Formatter('%(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "pipeline.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return structlog.get_logger()

def save_shutdown_logs(stats=None):
    """Save logs on shutdown with optional stats."""
    if stats:
        log_capture.add_entry({
            'event': 'pipeline_shutdown',
            'stats': stats,
            'level': 'info'
        })
    
    return log_capture.save_to_file()

# Global logger instance
logger = setup_logging()