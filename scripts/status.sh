#!/bin/bash

# Status script for image processing service

set -e

SERVICE_NAME="image-processor"

echo "=== Service Status ==="
sudo systemctl status $SERVICE_NAME

echo ""
echo "=== Queue Status ==="
python3 -c "
from sqs_service import SQSService
sqs = SQSService()
print(f'Queue Depth: {sqs.get_queue_depth()}')
print(f'In Flight Messages: {sqs.get_in_flight_messages()}')
"

echo ""
echo "=== Recent Logs ==="
sudo journalctl -u $SERVICE_NAME --no-pager -n 10