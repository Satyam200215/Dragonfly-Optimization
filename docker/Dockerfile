# Use Debian 12 as the base image to match the EC2 AMI
FROM debian:12-slim

# Install bash and clean up
RUN apt-get update && apt-get install -y \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Command to keep the container running (from your earlier context)
CMD ["sh", "-c", "while true; do echo 'Running'; sleep 1; done"]