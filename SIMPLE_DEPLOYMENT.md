# Simple Image Processing Setup

## Quick Setup

### 1. Create Infrastructure
```bash
# Set variables
export AWS_REGION="us-east-2"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create SQS queue
aws sqs create-queue --queue-name "image-processing-queue" --region $AWS_REGION
export QUEUE_URL=$(aws sqs get-queue-url --queue-name "image-processing-queue" --region $AWS_REGION --query 'QueueUrl' --output text)

# Create S3 bucket
export S3_BUCKET="clore-processed-images-$(date +%s)"
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION

# Create security group
export SG_ID=$(aws ec2 create-security-group --group-name image-processor-sg --description "Security group for image processor" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0

# Create key pair
aws ec2 create-key-pair --key-name image-processor-key --query 'KeyMaterial' --output text > image-processor-key.pem
chmod 400 image-processor-key.pem
```

### 2. Launch Instance
```bash
# Get AMI ID
export AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" --query 'Images[0].ImageId' --output text)

# Launch g5.2xlarge instance
export INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type g5.2xlarge \
    --key-name image-processor-key \
    --security-group-ids $SG_ID \
    --query 'Instances[0].InstanceId' \
    --output text)

# Wait for running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get IP
export PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
```

### 3. Setup Instance
```bash
# SSH and install everything
ssh -i image-processor-key.pem ec2-user@$PUBLIC_IP

# Run these commands on the instance:
sudo yum update -y
sudo yum install -y python3 python3-pip git htop

# Install NVIDIA drivers
sudo yum install -y gcc kernel-devel-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-12.2.0-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-12.2.0-1.x86_64.rpm
sudo yum clean all
sudo yum install -y cuda

# Install Python packages
pip3 install torch torchvision diffusers transformers accelerate safetensors pillow boto3 psycopg2-binary python-dotenv structlog tenacity

# Download code
git clone https://github.com/your-repo/image-processor.git
cd image-processor
```

### 4. Configure
```bash
# Create .env file
cat > .env << EOF
AWS_REGION=us-east-2
QUEUE_URL=$QUEUE_URL
S3_BUCKET=$S3_BUCKET
DB_CONNECTION_STRING="postgresql://postgres:rozne0-bevrar-rendoD@cloredevnew.cr06ku8y6c6t.us-east-2.rds.amazonaws.com:5432/cloreapp"
BATCH_SIZE=10
DEVICE=cuda
LOG_LEVEL=INFO
EOF
```

### 5. Run
```bash
# Test GPU
nvidia-smi

# Run processor
python3 pipeline_orchestrator.py
```

## Weekly Usage

### Start Processing
```bash
# Start instance
aws ec2 start-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# SSH and run
ssh -i image-processor-key.pem ec2-user@$PUBLIC_IP
cd image-processor
python3 pipeline_orchestrator.py
```

### Stop When Done
```bash
# Stop instance (saves money)
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

## Cost
- **g5.2xlarge**: $1.21/hour
- **Processing time**: 7-9 hours for 2000-3000 products
- **Cost per session**: ~$8.50-11
- **Monthly**: ~$17-22 (2 sessions/week)

That's it. No complex scripts, no auto-recovery, just simple start/stop when needed.