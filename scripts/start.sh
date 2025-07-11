#!/bin/bash

# Start script for image processing service

set -e

SERVICE_NAME="image-processor"

echo "Starting image processing service..."

# Check if service is already running
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Service is already running"
    systemctl status $SERVICE_NAME
    exit 0
fi

# Start the service
sudo systemctl start $SERVICE_NAME

# Wait a moment for startup
sleep 2

# Check status
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Service started successfully"
    systemctl status $SERVICE_NAME
else
    echo "Service failed to start"
    sudo journalctl -u $SERVICE_NAME --no-pager -n 20
    exit 1
fi