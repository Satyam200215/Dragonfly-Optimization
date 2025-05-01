# Dragonfly-Optimization
Cloud computing has transformed how computing resources are managed and utilized, delivering benefits such as optimized infrastructure use, increased flexibility, and faster deployment times. Despite these advantages, achieving seamless interoperability across diverse cloud environments remains a significant challenge, especially in terms of ensuring smooth resource sharing and access in heterogeneous systems. Containers have emerged as a lightweight virtualization solution that enhances scalability, portability, and flexibility in cloud services. However, issues such as energy efficiency and effective data migration in these diverse environments persist. The proposed solution employs the Adaptive Dragonfly Optimization (ADrO) algorithm for efficient data migration. While the traditional Dragonfly optimization algorithm is prone to local optima trapping due to loss of population diversity, this limitation is addressed by integrating a Levy flight strategy, enhancing population diversity and accelerating convergence toward optimal solutions. In the migration process, user tasks are collected, organized, and assigned to containers, ensuring efficient resource allocation. Additionally, load prediction is performed using an Actor-Critic Neural Network (ACNN) to further optimize migration decisions, accounting for predicted load and system capacity. A multi-objective function is developed to evaluate migration based on parameters such as predicted load transmission cost, demand, resource capacity, agility, reputation, migration time, and energy consumption. The effectiveness of this approach is demonstrated through comprehensive evaluations and comparisons with other methods. The proposed energy-efficient data migration framework is implemented using Python, showcasing its potential to enhance performance in container-based heterogeneous cloud environments.
AWS VM and Container Optimization Project

## Overview
Creates EC2 instances on AWS, deploys Docker containers, and uses ACNN to find the least loaded VM and Dragonfly Algorithm with Levy Flight to select the best container.

## Setup
1. Install AWS CLI and configure with `aws configure`.
2. Create a key pair (`my-key`) in AWS Console and save to `keys/`.
3. Update `create_ec2.py` with your AMI ID and security group.
4. Install Python dependencies: `pip install boto3 paramiko numpy tensorflow scipy`.
5. Run `python ec2_scripts/create_ec2.py` to launch instances.
6. Build container: `cd docker && docker build -t mycontainer:latest . && docker save -o mycontainer.tar mycontainer:latest`.
7. Deploy containers: `python ec2_scripts/deploy_containers.py`.
8. Optimize: `python ec2_scripts/optimize.py`.

## Expected Output
- Least loaded VM with CPU%, Memory%, and Network IO.
- Best container with load, cost, energy, CPU, memory, and fitness score.
