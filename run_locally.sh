#!/bin/bash
# Script to run crypto trading bot locally on a non-US server
# This script is optimized for Linux/macOS environments

echo "🤖 Cryptocurrency Trading Bot Local Setup 🤖"
echo "============================================="
echo

# Create virtual environment
echo "📦 Setting up virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "🔑 Creating .env file for your API keys..."
    cat > .env << EOF
# KuCoin API Credentials
KUCOIN_API_KEY=your_api_key
KUCOIN_API_SECRET=your_api_secret
KUCOIN_PASSPHRASE=your_passphrase

# Database Configuration
DATABASE_URL=sqlite:///crypto_bot.db

# Flask Configuration
FLASK_SECRET_KEY=$(openssl rand -hex 24)
FLASK_DEBUG=0
EOF
    echo "⚠️ Please edit the .env file to add your KuCoin API credentials"
    echo "📝 .env file created successfully!"
fi

# Load environment variables
echo "🔄 Loading environment variables..."
export $(grep -v '^#' .env | xargs)

# Reset and initialize the database
echo "🗄️ Initializing database..."
python reset_db_and_settings.py

# Run the application
echo "🚀 Starting the trading bot..."
echo "🌐 The application will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the bot"
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app