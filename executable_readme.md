# Crypto Trading Bot - Executable Guide

This executable version of the trading bot provides a standalone application that doesn't require Python to be installed on the system.

## Running the Application

Simply double-click `TradingBot.exe` to run the application. A web browser should automatically open to http://localhost:5000 after a few seconds.

If the windowed version doesn't work or you want to see error messages, try running `TradingBot_Console.exe` instead, which will show a console window with detailed logs.

## First-Time Setup

On first launch, the application will:
1. Create a local SQLite database in the same folder
2. Set up default configuration values
3. Start in "paper trading" mode (no real money at risk)

## API Keys

To connect to exchanges:
1. Go to Settings in the web interface
2. Enter your API key and secret for your preferred exchange
3. Save the configuration

The application can work in read-only mode without API keys, showing market data and indicators but not executing trades.

## Troubleshooting

If you encounter issues:

1. **Application doesn't start**: Try running the console version to see error messages
2. **Browser doesn't open**: Manually navigate to http://localhost:5000
3. **Can't connect to exchange**: Verify your API keys are entered correctly
4. **Missing data files**: Ensure the `model` and `data` folders exist in the same directory as the executable

## Known Limitations

- The executable is optimized for Windows systems
- Some antivirus software may flag the executable - you may need to add an exception
- Default database is SQLite for simplicity, which may be slower than PostgreSQL for large datasets

## Common Issues

- **Windows Defender SmartScreen**: You may see a warning when first running the executable. Click "More info" and then "Run anyway" to proceed.
- **Firewall alerts**: The application needs to access the internet to download market data. Allow access when prompted.
- **Port conflicts**: If port 5000 is already in use by another application, it will fail to start. Close other applications that might be using this port.