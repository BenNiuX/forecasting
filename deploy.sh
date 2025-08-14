#!/bin/bash
set -e

# Check if .env exists
if [ ! -f docker-compose.env ]; then
    echo "Creating docker-compose.env from template..."
    cp docker-compose.env.template docker-compose.env
    echo "Please edit docker-compose.env with your actual values"
    exit 1
fi

# Build and deploy
echo "Building containers..."
docker compose build

echo "Starting services..."
docker compose --env-file docker-compose.env up -d

echo "Deployment complete!"
echo "Backend: http://localhost:8089"
echo "Frontend: http://localhost:3000"

echo "To stop services, run: docker compose stop"
echo "To restart services, run: docker compose restart"
echo "To remove all images, run: docker compose down --rmi all"
