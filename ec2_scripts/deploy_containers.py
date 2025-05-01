import paramiko
import os
from paramiko import RSAKey

def deploy_container(ip, username='bitnami', key_path='C:\\Users\\satya\\OneDrive\\Documents\\Desktop\\DADo\\keys\\SSH1.pem'):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Connecting to {ip} with key {key_path} and username {username}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = RSAKey.from_private_key_file(key_path)
    ssh.connect(ip, username=username, pkey=key)

    print(f"Connected to {ip}, transferring mycontainer.tar")
    tar_path = 'C:\\Users\\satya\\OneDrive\\Documents\\Desktop\\DADo\\docker\\mycontainer.tar'
    sftp = ssh.open_sftp()
    sftp.put(tar_path, "/home/bitnami/mycontainer.tar")  # Updated path to bitnami home directory
    sftp.close()

    print("Executing Docker setup and container deployment")
    commands = [
        "sudo apt-get update",  # Debian uses apt-get instead of yum
        "sudo apt-get install -y docker.io",  # Install Docker on Debian
        "sudo systemctl start docker",
        "sudo docker load -i /home/bitnami/mycontainer.tar",
        "sudo docker run -d --name container1_vm$(hostname) mycontainer:latest",
        "sudo docker run -d --name container2_vm$(hostname) mycontainer:latest"
    ]
    for cmd in commands:
        print(f"Executing: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode().strip() or stderr.read().decode().strip())

    ssh.close()

if __name__ == "__main__":
    print(f"Reading instance_info.txt from {os.getcwd()}")
    with open('instance_info.txt', 'r') as f:
        for line in f:
            instance_id, ip = line.strip().split(':')
            print(f"Deploying to {ip} ({instance_id})")
            deploy_container(ip)