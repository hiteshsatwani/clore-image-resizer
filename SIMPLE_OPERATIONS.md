# Simple Weekly Operations

## Before Processing

### 1. Start Instance
```bash
# Start the instance
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Wait for it to be ready (2-3 minutes)
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get IP
export PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
```

### 2. Start Processing
```bash
# SSH to instance
ssh -i image-processor-key.pem ec2-user@$PUBLIC_IP

# Go to app directory
cd image-processor

# Start processing
python3 pipeline_orchestrator.py
```

## During Processing

### Monitor Progress
```bash
# Check queue depth
aws sqs get-queue-attributes --queue-url $QUEUE_URL --attribute-names ApproximateNumberOfMessages

# Check GPU usage
nvidia-smi

# View logs
# (logs appear in terminal where you ran the script)
```

## After Processing

### Stop Everything
```bash
# Stop the script (Ctrl+C)
# Exit SSH (type 'exit')

# Stop instance to save money
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

## That's It

- **Before**: Start instance, SSH, run script
- **During**: Monitor via terminal output
- **After**: Stop script, stop instance

**Total time**: 5 minutes setup + 5 minutes cleanup = 10 minutes per week