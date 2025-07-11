# Clore Image Processing Pipeline

Simple image processing system that converts product images to 9:16 aspect ratio using AI.

## Files
- `pipeline_orchestrator.py` - Main application
- `image_resizer.py` - AI image processing
- `database_service.py` - Database operations
- `sqs_service.py` - SQS message handling
- `s3_service.py` - S3 upload/download
- `config.py` - Configuration management
- `logger.py` - Logging
- `requirements-production.txt` - Dependencies
- `SIMPLE_DEPLOYMENT.md` - Setup instructions
- `SIMPLE_OPERATIONS.md` - Weekly workflow

## Quick Start

1. Follow `SIMPLE_DEPLOYMENT.md` to set up AWS infrastructure
2. Use `SIMPLE_OPERATIONS.md` for weekly operations
3. Start instance → SSH → Run script → Stop instance

## Cost
- **g5.2xlarge**: $1.21/hour
- **Processing time**: 7-9 hours for 2000-3000 products
- **Cost per session**: ~$8.50-11