# Cryptocurrency Trading Bot - Local Installation Guide

This guide will help you set up and run the cryptocurrency trading bot on your local machine in Brazil, which should allow you to connect to KuCoin without geographical restrictions.

## Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)
- KuCoin API credentials (API key, secret, and passphrase)
- Internet connection

## Installation Steps

### 1. Download the Code

First, download the bot code from Replit or clone the repository if it's available on GitHub.

### 2. Setup the Environment

#### For Linux/macOS:

1. Make the setup script executable:
   ```bash
   chmod +x run_locally.sh
   ```

2. Run the script:
   ```bash
   ./run_locally.sh
   ```

#### For Windows:

1. Run the Windows setup script:
   ```
   setup_windows.bat
   ```

### 3. Configure Your API Keys

Edit the `.env` file that was created during setup and add your KuCoin API credentials:

```
# KuCoin API Credentials
KUCOIN_API_KEY=your_api_key
KUCOIN_API_SECRET=your_api_secret
KUCOIN_PASSPHRASE=your_passphrase

# Database Configuration
DATABASE_URL=sqlite:///crypto_bot.db

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key
FLASK_DEBUG=0
```

### 4. Manual Installation (if the scripts don't work)

If the automated scripts don't work, you can manually set up the environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r bot_local_packages.txt
   ```

4. Create a `.env` file with your API credentials

5. Run the bot:
   ```bash
   gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
   ```

## Usage

1. After running the setup script and configuring your API keys, the bot will start and be accessible at:
   ```
   http://localhost:5000
   ```

2. Open this URL in your web browser to access the trading bot interface.

3. The bot should automatically connect to KuCoin using your Brazilian IP address without geographical restrictions.

## Troubleshooting

- **API Connection Issues**: Ensure your KuCoin API credentials are correctly entered in the `.env` file.
- **Installation Problems**: Make sure you have Python 3.8+ installed and accessible from your command line.
- **Database Errors**: If you encounter database errors, try running `python reset_db_and_settings.py` to reset the database.

## Important Notes

- Since you're running this from Brazil, you should be able to connect to KuCoin without the US region restrictions.
- The bot will use your KuCoin API credentials to access real market data and potentially execute trades.
- Make sure your API keys have the appropriate permissions required for your trading strategy.
- Always start with paper trading enabled until you're confident in the bot's performance.