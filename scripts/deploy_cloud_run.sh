#!/bin/bash

# Ensure we are running from the script's directory
cd "$(dirname "$0")/.."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "gcloud CLI could not be found. Please install it first."
    exit 1
fi

echo "Enter your Google Cloud Project ID (or press Enter if already configured globally):"
read PROJECT_ID

if [ -n "$PROJECT_ID" ]; then
    gcloud config set project $PROJECT_ID
fi

echo "=================================================="
echo "Deploying Investa Backend to Google Cloud Run..."
echo "=================================================="
# Deploy backend from root directory
gcloud run deploy investa-api \
    --source . \
    --port 8000 \
    --memory 2048Mi \
    --allow-unauthenticated \
    --region us-central1

# Get the deployed URL for the backend
BACKEND_URL=$(gcloud run services describe investa-api --region us-central1 --format 'value(status.url)')
echo "Backend deployed at: $BACKEND_URL"

echo "=================================================="
echo "Deploying Investa Web App to Google Cloud Run..."
echo "=================================================="
# Deploy frontend from web_app directory
cd web_app
gcloud run deploy investa-web \
    --source . \
    --port 3000 \
    --set-env-vars API_URL=$BACKEND_URL \
    --set-build-env-vars NEXT_PUBLIC_API_URL=$BACKEND_URL \
    --allow-unauthenticated \
    --region us-central1

echo "=================================================="
echo "Deployment Complete!"
echo "Make sure to update macos_app/Investa/Networking/APIConfig.swift"
echo "with the new backend URL: $BACKEND_URL/api"
echo "=================================================="
