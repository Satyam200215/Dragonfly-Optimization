DADO for Efficient Data Migration in Cloud Instances

Overview

The DADO project is a robust framework designed to optimize data migration in containerized heterogeneous cloud environments. It leverages the Adaptive Dragonfly Optimization (ADrO) algorithm with a Levy flight strategy for efficient resource allocation and an Actor-Critic Neural Network (ACNN) for dynamic workload prediction. A multi-objective function evaluates migration based on energy consumption, migration time, transmission cost, and resource capacity. The system is implemented using Python and Docker, ensuring scalability, portability, and energy efficiency. Performance evaluations demonstrate significant improvements over traditional methods like Genetic Algorithm (GA) and Particle Swarm Optimization (PSO).

This project was developed as a B.Tech Major Project by Ch Sricharan, Sadanand Goud, and Satyam Das at Geethanjali College of Engineering and Technology, under the guidance of S Durga Prasad.

#Features

Optimized Data Migration: Uses ADrO with Levy flight to avoid local optima and ensure efficient container-to-VM and VM-to-PM assignments.
Dynamic Workload Prediction: Employs ACNN to forecast workload patterns, enabling proactive migration decisions.
Multi-Objective Evaluation: Balances energy, cost, time, and resource capacity for cost-effective and sustainable migrations.
Containerized Deployment: Utilizes Docker for portable and scalable cloud environments.
Performance Visualization: Generates graphs using Matplotlib to analyze energy efficiency, migration time, and resource utilization.
Scalability: Handles a 50% workload increase without performance degradation.

#Repository Structure

├── ec2_scripts/                # Scripts for AWS EC2 instance management and optimization
│   ├── create_ec2.py          # Script to launch EC2 instances
│   ├── deploy_containers.py   # Script to deploy Docker containers
│   ├── optimize.py            # Main script for ADrO and ACNN optimization
├── docker/                    # Docker-related files
│   ├── Dockerfile             # Dockerfile for building container images
│   ├── mycontainer.tar        # Exported Docker container image
├── keys/                      # Directory for SSH key storage (e.g., SSH1.pem)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

#Prerequisites

AWS Account: Configured with AWS CLI and necessary permissions.
Python 3.9+: For running optimization scripts and neural network models.
Docker: For building and deploying containerized environments.
SSH Key: AWS EC2 key pair for instance access.
Hardware: Minimum 8GB RAM, multi-core CPU for simulations.
OS: Compatible with Linux, macOS, or Windows (with WSL for Docker).

#Setup Instructions
Clone the Repository:

git clone https://github.com/Satyam200215/Dragonfly-Optimization.git
cd Dragonfly-Optimization

Install Dependencies: Install required Python packages:

pip install -r requirements.txt

Required packages: boto3, paramiko, numpy, tensorflow (or pytorch), scipy, pandas, matplotlib.

Configure AWS CLI: Set up AWS credentials:

aws configure

Provide your AWS Access Key, Secret Key, region, and output format.

Set Up SSH Key:

Create an AWS EC2 key pair named my-key in the AWS Console.
Save the .pem file to the keys/ directory (e.g., keys/SSH1.pem).

Update optimize.py with the correct path to your .pem file.

Update EC2 Configuration:
Open ec2_scripts/create_ec2.py.
Update the AMI ID and security group ID to match your AWS region and account.

Usage

Launch EC2 Instances: Create AWS EC2 instances to host containers:
python ec2_scripts/create_ec2.py
Build Docker Container: Build and export the Docker container image:

cd docker
docker build -t mycontainer:latest .
docker save -o mycontainer.tar mycontainer:latest

Deploy Containers: Deploy the Docker container to EC2 instances:
python ec2_scripts/deploy_containers.py

Run Optimization: Execute the optimization script to predict workloads, allocate resources, and evaluate performance:
python ec2_scripts/optimize.py

Expected Output
Console Output:
Details of the least loaded VM and container (CPU%, Memory%, Network IO).
Best container metrics (load, cost, energy, CPU, memory, fitness score).
Summary of energy and cost conservation across VMs.


#Performance Metrics:

Energy consumption reduced by at least 20% compared to GA and PSO.
Migration optimization and workload prediction completed within 3 seconds for up to 10,000 tasks.

#Results

Migration Efficiency: ADrO optimized resource allocation, minimizing energy consumption and migration time.
Workload Prediction: ACNN achieved high accuracy with errors below 5%.
Scalability: Handled 50% workload increase without performance degradation.
Visualization: Clear Matplotlib graphs for performance metrics, outperforming GA and PSO benchmarks.


#Future Enhancements

Develop a mobile app for on-the-go management.
Integrate advanced AI models (e.g., deep Q-networks) for improved predictions.
Add multi-level user roles for enhanced access control.
Enable real-time visualizations using Plotly or Grafana.
Implement automated alerts for resource overuse or migration failures.

#References

Tej.C and K.S. Rekha, 2023. Energy Efficient Data Migration Concerning Interoperability Using Optimized Deep Learning in Container-Based Heterogeneous Cloud Computing.
Mohammed Shehab et al., 2021. Dragonfly Algorithm: A Comprehensive Survey of Its Results, Variants, and Applications.
Full list available in the project report (Chapter 9).


Project Repository: https://github.com/Satyam200215/Dragonfly-Optimization
