@echo off
REM Script to run crypto trading bot locally on a Windows machine

echo 🤖 Cryptocurrency Trading Bot Local Setup 🤖
echo =============================================
echo.

REM Create virtual environment
echo 📦 Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
IF NOT EXIST .env (
    echo 🔑 Creating .env file for your API keys...
    (
        echo # KuCoin API Credentials
        echo KUCOIN_API_KEY=your_api_key
        echo KUCOIN_API_SECRET=your_api_secret
        echo KUCOIN_PASSPHRASE=your_passphrase
        echo.
        echo # Database Configuration
        echo DATABASE_URL=sqlite:///crypto_bot.db
        echo.
        echo # Flask Configuration
        echo FLASK_SECRET_KEY=your_secret_key
        echo FLASK_DEBUG=0
    ) > .env
    echo ⚠️ Please edit the .env file to add your KuCoin API credentials
    echo 📝 .env file created successfully!
)

REM Reset and initialize the database
echo 🗄️ Initializing database...
python reset_db_and_settings.py

REM Run the application
echo 🚀 Starting the trading bot...
echo 🌐 The application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the bot
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app