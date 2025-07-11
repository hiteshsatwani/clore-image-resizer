#!/bin/bash

# Deployment script for image processing service

set -e

# Configuration
APP_DIR="/home/ec2-user/image-processor"
SERVICE_NAME="image-processor"

echo "Starting deployment of image processing service..."

# Create application directory
sudo mkdir -p $APP_DIR
sudo chown ec2-user:ec2-user $APP_DIR

# Copy application files
echo "Copying application files..."
cp -r . $APP_DIR/

# Install Python dependencies
echo "Installing Python dependencies..."
cd $APP_DIR
pip3 install -r requirements-production.txt

# Create .env file from example if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit $APP_DIR/.env with your configuration"
fi

# Install systemd service
echo "Installing systemd service..."
sudo cp systemd/image-processor.service /etc/systemd/system/
sudo systemctl daemon-reload

# Set permissions
echo "Setting file permissions..."
chmod +x $APP_DIR/pipeline_orchestrator.py
chmod +x $APP_DIR/scripts/*.sh

# Create log directory
sudo mkdir -p /var/log/image-processor
sudo chown ec2-user:ec2-user /var/log/image-processor

echo "Deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit $APP_DIR/.env with your configuration"
echo "2. Test the service: sudo systemctl start $SERVICE_NAME"
echo "3. Check status: sudo systemctl status $SERVICE_NAME"
echo "4. View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "5. Enable on boot: sudo systemctl enable $SERVICE_NAME"