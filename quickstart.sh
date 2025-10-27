#!/bin/bash

echo "=================================="
echo "🚀 Smart Filter Selector - Quick Start"
echo "=================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "   Please install from: https://ollama.com"
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
else
    echo "✅ Ollama is running"
fi

# Pull required models
echo ""
echo "📥 Pulling required models..."
echo "   This may take a few minutes..."

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "   Pulling nomic-embed-text..."
    ollama pull nomic-embed-text
else
    echo "   ✅ nomic-embed-text already available"
fi

if ! ollama list | grep -q "llama3.2:3b"; then
    echo "   Pulling llama3.2:3b..."
    ollama pull llama3.2:3b
else
    echo "   ✅ llama3.2:3b already available"
fi

# Check Python virtual environment
echo ""
echo "🐍 Setting up Python environment..."

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

echo "   Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "   Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Generate embeddings if not exists
if [ ! -f "data/embeddings.json" ]; then
    echo ""
    echo "📊 Generating embeddings..."
    echo "   This will take 5-10 minutes..."
    python scripts/generate_embeddings.py
else
    echo ""
    echo "✅ Embeddings already exist"
fi

# Start the application
echo ""
echo "=================================="
echo "🎉 Starting Flask Application..."
echo "=================================="
echo ""

python run.py