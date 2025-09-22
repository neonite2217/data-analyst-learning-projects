#!/bin/bash
# Enhanced Stock Analyzer - Quick Setup & Launch
# Made by Neonite - Production Version

echo "📈 Enhanced Stock Analyzer"
echo "Made by Neonite"
echo "========================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Launch
if [ $? -eq 0 ]; then
    echo "✅ Setup complete!"
    echo ""
    echo "🚀 Launching GUI with Interactive Charts..."
    python gui.py
else
    echo "❌ Installation failed. Check internet connection."
    exit 1
fi
