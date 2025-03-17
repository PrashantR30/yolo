#!/bin/bash

# Build script for YOLO detection system
# This script builds the Docker images for both CPU and GPU configurations

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting build process..."

# Build web-server image
echo "Building web-server image..."
docker build -t web-server:latest ./web-server

# Build yolo-detector CPU image
echo "Building yolo-detector CPU image..."
docker build -t yolo-detector:latest -f ./yolo-detector/Dockerfile.cpu ./yolo-detector

# Build yolo-detector GPU image (optional)
echo "Building yolo-detector GPU image..."
docker build -t yolo-detector:gpu -f ./yolo-detector/Dockerfile.gpu ./yolo-detector

echo "Build completed successfully!"
echo ""
echo "The following images are now available:"
echo "  - web-server:latest    (Web server image)"
echo "  - yolo-detector:latest (CPU-based YOLO detector image)"
echo "  - yolo-detector:gpu    (GPU-based YOLO detector image)"
echo ""
echo "To run with CPU:"
echo "  docker-compose up"
echo ""
echo "To run with GPU:"
echo "  docker-compose -f docker-compose.gpu.yml up"
echo ""
