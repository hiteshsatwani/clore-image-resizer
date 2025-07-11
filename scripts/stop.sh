#!/bin/bash

# Stop script for image processing service

set -e

SERVICE_NAME="image-processor"

echo "Stopping image processing service..."

# Check if service is running
if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo "Service is not running"
    exit 0
fi

# Stop the service
sudo systemctl stop $SERVICE_NAME

# Wait for shutdown
sleep 2

# Check status
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Service failed to stop"
    exit 1
else
    echo "Service stopped successfully"
fi