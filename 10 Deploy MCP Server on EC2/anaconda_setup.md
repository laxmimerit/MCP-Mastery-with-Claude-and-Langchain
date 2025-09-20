# Anaconda Setup on Amazon Linux EC2

Quick setup guide for installing Anaconda and Python on an Amazon Linux EC2 instance.

## Prerequisites
- Amazon Linux EC2 instance
- SSH access with key pair

## Connecting to EC2 Instance

Connect to your EC2 instance using SSH:

**From Windows (Command Prompt/PowerShell):**
```bash
# If pem file is in current directory
ssh -i "mcp.pem" ec2-user@ec2-3-6-90-231.ap-south-1.compute.amazonaws.com

# If pem file is in .ssh folder
ssh -i "~/.ssh/mcp.pem" ec2-user@ec2-3-6-90-231.ap-south-1.compute.amazonaws.com
```

**Note:** Always use `ec2-user` (not `root`) for Amazon Linux instances. The system will display a Deep Learning AMI welcome message showing available tools and GPU configurations.

## Installation Steps

1. **Update the system**
   ```bash
   sudo yum update -y
   ```

2. **Download Anaconda installer**
   ```bash
   curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
   ```

3. **Run the installer**
   ```bash
   bash Anaconda3-2025.06-0-Linux-x86_64.sh
   ```
   - Follow the prompts
   - Accept the license agreement
   - Choose installation location: `/home/ec2-user/anaconda3` (when prompted)
   - When asked about shell initialization, choose "no" initially

4. **Add Anaconda to PATH**
   ```bash
   export PATH="/home/ec2-user/anaconda3/bin:$PATH"
   echo 'export PATH="/home/ec2-user/anaconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

5. **Initialize conda**
   ```bash
   conda init bash
   source ~/.bashrc
   ```

## Verification

Check installations:
```bash
conda --version
python --version
pip --version
```

## Optional: Install uv (fast Python package manager)
```bash
pip install uv
```

You should now have a fully functional Anaconda environment with Python 3.13 and conda package manager.