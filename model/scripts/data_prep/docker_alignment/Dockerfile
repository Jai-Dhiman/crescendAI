# Dockerfile for running Nakamura's MIDI-to-MIDI alignment tool
# The alignment tool is a Linux binary that aligns performance MIDI with score MIDI

# Use x86_64 platform since AlignmentTool binaries are compiled for x86_64
FROM --platform=linux/amd64 ubuntu:22.04

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && pip3 install tqdm \
    && rm -rf /var/lib/apt/lists/*

# Copy the AlignmentTool binaries
WORKDIR /alignment
COPY AlignmentTool/ /alignment/

# Make the programs executable
RUN chmod +x /alignment/Programs/* /alignment/*.sh

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Default working directory
WORKDIR /alignment

# Entry point script will be mounted
CMD ["bash"]
