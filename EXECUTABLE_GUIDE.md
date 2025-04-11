# Crypto Trading Bot Executable Guide

This guide provides detailed instructions on creating an executable (.exe) file for the Crypto Trading Bot and distributing it to end users.

## Building the Executable

### Prerequisites
- Python 3.7+ installed on your system
- All dependencies installed with `pip install -r bot_requirements.txt`
- PyInstaller package: `pip install pyinstaller`

### Build Process Options

#### Option 1: Simple Build
For a basic build with minimal configuration:

1. Open a command prompt/terminal in the project directory
2. Run: `python simple_build_exe.py`
3. Wait for the process to complete (may take several minutes)
4. The executable will be created in the `dist` folder as `TradingBot.exe`

#### Option 2: Advanced Build
For a more comprehensive build with additional configuration:

1. Open a command prompt/terminal in the project directory
2. Run: `python build_exe.py`
3. Wait for the process to complete (may take several minutes)
4. The executable will be created in the `dist` folder with additional resources

### What the Build Does
The PyInstaller process:
- Analyzes all code dependencies
- Includes necessary Python libraries
- Packages the web interface (templates and static files)
- Creates a standalone executable that doesn't require Python

## Distributing the Executable

### What to Distribute
At minimum, you'll need to share:
- The `TradingBot.exe` file
- The `model` folder (containing ML model files)

For a more complete distribution, consider including:
- A default `config.json` file
- A `README` file with setup instructions
- Documentation on API keys required for trading

### Setting Up Environment
End users will need:
1. A Binance account (or other supported exchange)
2. API keys with the appropriate permissions
3. Network access to exchange APIs

### Configuration Options

Users can configure the trading bot through:
1. The web interface after starting the application
2. A `config.json` file in the same directory as the executable

## Useful Information for End Users

### Running the Executable
1. Double-click `TradingBot.exe`
2. The application will open a browser window automatically
3. The trading bot interface will be available at http://127.0.0.1:5000

### Data Storage
The executable version uses SQLite for data storage:
- Database file is created in a `data` folder
- Trading history and configuration are stored locally
- ML models are loaded from the `model` folder

### Security Considerations
Important for end users to know:
- API keys should be protected and never shared
- The application stores API keys in the local database
- Paper trading mode is enabled by default (no real money at risk)

### Troubleshooting Common Issues

#### Executable Won't Start
- Check Windows Defender or antivirus (may block executable)
- Run as administrator if accessing system resources
- Check logs in the `logs` folder

#### Connection Issues
- Verify network connectivity to exchange APIs
- Check if API keys have appropriate permissions
- Verify firewall isn't blocking outgoing connections

#### Database Errors
- The application creates a new database if none exists
- If database corruption occurs, delete the `data` folder to reset

## Technical Information

### Architecture Overview
- Flask-based web interface
- SQLite database for standalone mode
- Real-time data from exchange APIs
- Machine learning predictions for trading signals

### Performance Considerations
- ML model loading may cause brief startup delay
- WebSocket connections for real-time updates
- Local processing of market data reduces API calls

### System Requirements
- Windows 7/10/11 (for Windows executable)
- 4GB RAM minimum (8GB recommended)
- Internet connection for exchange API access