# Cryptocurrency Trading Bot - Executable Version

This is the standalone executable version of the Cryptocurrency Trading Bot. This application includes all necessary dependencies and runs without requiring Python installation.

## Building the Executable

### Windows
1. Make sure you have Python 3.8+ installed
2. Open Command Prompt and navigate to the project folder
3. Run: `pip install -r bot_requirements.txt`
4. Run: `build_executable.bat` or `python build_robust_exe.py`
5. The executable will be created in the `dist` folder

### macOS/Linux
1. Make sure you have Python 3.8+ installed
2. Open Terminal and navigate to the project folder
3. Run: `pip install -r bot_requirements.txt`
4. Run: `bash build_executable.sh` or `python build_robust_exe.py`
5. The executable will be created in the `dist` folder

## Getting Started

1. **Launch the Application**: Simply double-click `TradingBot.exe` (Windows) or `TradingBot` (macOS/Linux) to start the application.
2. **Access the Web Interface**: A browser window will automatically open to `http://127.0.0.1:5000` once the application is started.
3. **Configure API Keys**: To enable trading, you'll need to set up your exchange API keys in the web interface.
4. **Start Trading**: Configure your strategy parameters, select trading pairs, and enable the bot.

## Folders and Files

- `data/` - Contains the SQLite database for storing application data
- `model/` - Contains machine learning model files
- `logs/` - Application logs are stored here

## Troubleshooting

### "Invalid async_mode specified"

If you encounter the error "Invalid async_mode specified" when running the executable, try these steps:

1. Make sure no other instance of the application is running
2. Try running the application as administrator
3. If the issue persists, check the logs folder for more detailed error information

### Antivirus/Firewall Blocking

Some antivirus software may block the executable. You may need to add an exception for the application.

### Port Already in Use

If port 5000 is already in use by another application, the trading bot will not start correctly. Ensure no other applications are using this port.

## Data Storage

All data is stored locally in the SQLite database in the `data` folder. This includes:

- Trading configuration
- Historical market data
- Trade records
- Performance metrics

## Features of Executable Version

- **Stand-alone Application**: Runs without Python installation on any platform
- **Web-based Interface**: Access all features through a browser
- **Local Database**: SQLite database for private data storage
- **Automatic Trading**: Run in paper trading or live trading mode
- **Technical Analysis**: Full suite of indicators (RSI, Bollinger Bands, MACD, etc.)
- **Machine Learning**: Predictive modeling for trade decisions
- **Multi-exchange Support**: Binance, KuCoin, Kraken, Coinbase
- **Performance Metrics**: Track results and optimize strategies
- **Backtesting**: Test strategies against historical data

## API Key Setup

To use live trading features, you'll need to set up API keys from your preferred exchange:

1. Create an account on your preferred exchange (Binance, KuCoin, Kraken, Coinbase)
2. Navigate to the API management section
3. Create a new API key (with trading permissions, but NOT withdrawal permissions)
4. Enter the API key and secret in the settings page of the bot interface

## Technical Support

For technical support or to report issues, please visit the repository at [GitHub Repository URL].

## Trading Disclaimer

This software is for educational and research purposes only. Use at your own risk. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor.

---

Copyright © 2025 | All Rights Reserved