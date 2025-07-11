"""Configuration management for the image processing pipeline."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the image processing pipeline."""
    
    # AWS Configuration
    aws_region: str = os.getenv('AWS_REGION', 'us-east-1')
    sqs_queue_url: str = os.getenv('QUEUE_URL', '')
    s3_bucket: str = os.getenv('S3_BUCKET', '')
    
    # Database Configuration
    db_connection_string: str = os.getenv('DB_CONNECTION_STRING', '')
    db_host: str = os.getenv('DB_HOST', 'localhost')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', 'products')
    db_user: str = os.getenv('DB_USER', 'postgres')
    db_password: str = os.getenv('DB_PASSWORD', '')
    
    # Processing Configuration
    batch_size: int = int(os.getenv('BATCH_SIZE', '10'))
    max_retries: int = int(os.getenv('MAX_RETRIES', '3'))
    visibility_timeout: int = int(os.getenv('VISIBILITY_TIMEOUT', '900'))
    
    # GPU Configuration
    device: str = os.getenv('DEVICE', 'cuda')
    max_gpu_memory: int = int(os.getenv('MAX_GPU_MEMORY', '20'))  # GB
    
    # S3 Configuration
    s3_key_prefix: str = os.getenv('S3_KEY_PREFIX', 'processed')
    s3_max_retries: int = int(os.getenv('S3_MAX_RETRIES', '3'))
    
    # Logging Configuration
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.sqs_queue_url:
            raise ValueError("QUEUE_URL environment variable is required")
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET environment variable is required")
        if not self.db_connection_string:
            if not all([self.db_host, self.db_name, self.db_user, self.db_password]):
                raise ValueError("Database configuration is incomplete")
            self.db_connection_string = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

# Global configuration instance
config = Config()