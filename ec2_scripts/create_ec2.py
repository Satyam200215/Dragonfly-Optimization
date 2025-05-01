# ec2_scripts/create_ec2.py
import boto3
import time

ec2 = boto3.client('ec2', region_name='us-east-1')

def create_instances(instance_count=3):
    # Launch EC2 instances without initial tags (to avoid duplicate key error)
    response = ec2.run_instances(
        ImageId='ami-000e875cc81ac2df0',  # Amazon Linux 2 AMI (update for your region)
        InstanceType='t2.micro',
        MinCount=instance_count,
        MaxCount=instance_count,
        KeyName='SSH1',  # Create this key pair in AWS Console first
        SecurityGroupIds=['sg-0d3cbcd5ff6a47241']  # Replace with your security group ID
    )
    instance_ids = [i['InstanceId'] for i in response['Instances']]
    print(f"Launched instances: {instance_ids}")

    # Wait for instances to be running
    ec2.get_waiter('instance_running').wait(InstanceIds=instance_ids)

    # Apply tags to each instance individually
    for i, instance_id in enumerate(instance_ids):
        ec2.create_tags(
            Resources=[instance_id],
            Tags=[{'Key': 'Name', 'Value': f'VM{i+1}'}]
        )
    print(f"Tagged instances with names VM1, VM2, VM3")

    # Get public IPs
    instances = ec2.describe_instances(InstanceIds=instance_ids)['Reservations'][0]['Instances']
    public_ips = [i['PublicIpAddress'] for i in instances]
    print(f"Public IPs: {public_ips}")

    return instance_ids, public_ips

if __name__ == "__main__":
    instance_ids, public_ips = create_instances()
    with open('instance_info.txt', 'w') as f:
        for id, ip in zip(instance_ids, public_ips):
            f.write(f"{id}:{ip}\n")
    print("Instance details saved to instance_info.txt")