import os
import logging
import time
import pandas as pd
import numpy as np
import ccxt
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from app import db
from models import Trade, MarketData
from config import get_config

# Configure logging
logger = logging.getLogger(__name__)

# Global exchange object
exchange = None

def initialize_exchange() -> ccxt.Exchange:
    """Initialize the exchange connection"""
    global exchange
    
    config = get_config()
    
    # Get KuCoin API credentials (our primary exchange)
    kucoin_api_key = os.environ.get('KUCOIN_API_KEY', '') or config.get('kucoin_api_key', '') or config.get('api_key', '')
    kucoin_api_secret = os.environ.get('KUCOIN_API_SECRET', '') or config.get('kucoin_api_secret', '') or config.get('api_secret', '')
    kucoin_passphrase = os.environ.get('KUCOIN_PASSPHRASE', '') or config.get('kucoin_passphrase', '')
    
    # Log API key status (masking the actual key)
    if kucoin_api_key:
        masked_key = kucoin_api_key[:4] + '*' * (len(kucoin_api_key) - 8) + kucoin_api_key[-4:] if len(kucoin_api_key) > 8 else '****'
        logger.info(f"Found KuCoin API key: {masked_key}")
    else:
        logger.warning("No KuCoin API key found")
    
    # For backward compatibility, check Binance credentials too
    binance_api_key = os.environ.get('BINANCE_API_KEY', '') or config.get('binance_api_key', '')
    binance_api_secret = os.environ.get('BINANCE_API_SECRET', '') or config.get('binance_api_secret', '')
    
    if binance_api_key:
        logger.info("Found Binance API credentials as well")
    
    # When paper trading is disabled, prioritize exchanges with API keys
    is_paper_trading = config.get('paper_trading', True)
    
    # ALWAYS prioritize KuCoin if we have API keys, regardless of paper trading mode
    # This ensures we can display real balance even when paper trading is enabled
    if kucoin_api_key and kucoin_api_secret:
        exchanges_to_try = [
            'kucoin',  # Prioritize KuCoin if we have credentials
            'kraken',
            'coinbase',
            'binance',
        ]
        logger.info("Using KuCoin as primary exchange due to available API keys")
    else:
        exchanges_to_try = [
            'kucoin',
            'kraken',
            'coinbase',
            'binance',
        ]
        
    logger.info(f"Exchange priority order: {', '.join(exchanges_to_try)}")
    
    # Try each exchange in sequence until one works
    for exchange_id in exchanges_to_try:
        try:
            logger.info(f"Attempting to connect to {exchange_id}...")
            
            # Initialize with API keys if available for the exchange
            if exchange_id == 'kucoin' and kucoin_api_key and kucoin_api_secret:
                logger.info("Using KuCoin API credentials")
                exchange = getattr(ccxt, exchange_id)({
                    'apiKey': kucoin_api_key,
                    'secret': kucoin_api_secret,
                    'password': kucoin_passphrase,  # KuCoin requires passphrase
                    'enableRateLimit': True,
                })
            elif exchange_id == 'binance' and binance_api_key and binance_api_secret:
                logger.info("Using Binance API credentials")
                exchange = getattr(ccxt, exchange_id)({
                    'apiKey': binance_api_key,
                    'secret': binance_api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'adjustForTimeDifference': True,
                    }
                })
            else:
                # Use public API for exchanges without credentials
                logger.info(f"Using public API for {exchange_id}")
                exchange = getattr(ccxt, exchange_id)({
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            logger.info(f"Successfully connected to {exchange_id}")
            return exchange
            
        except Exception as e:
            error_msg = str(e)
            if exchange_id == 'kucoin' and ('unavailable in the U.S' in error_msg or 'restricted' in error_msg):
                logger.warning(f"KuCoin connection failed due to region restrictions: {error_msg}")
                logger.warning("KuCoin services are unavailable in your current region. Using fallback exchange...")
            elif exchange_id == 'binance' and ('IP' in error_msg or 'restricted location' in error_msg):
                logger.warning(f"Binance connection failed due to IP restrictions: {error_msg}")
                logger.warning("To use Binance API, you need to add the current IP to your Binance API allowlist")
            else:
                logger.warning(f"Failed to connect to {exchange_id}: {error_msg}")
            continue
    
    # If all exchanges fail
    logger.error("Failed to connect to any exchange. Check if your API keys are valid and allow access from this IP address.")
    return None

def get_exchange() -> ccxt.Exchange:
    """Get the exchange object, initializing it if necessary"""
    global exchange
    
    if exchange is None:
        exchange = initialize_exchange()
    
    return exchange

def get_exchange_name() -> str:
    """Get the name of the current exchange"""
    global exchange
    
    if exchange is None:
        exchange = initialize_exchange()
    
    if exchange:
        return exchange.id.capitalize()
    else:
        return "Not Connected"

def fetch_market_data(symbol: str, timeframe: str, limit: int = 100, since: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Fetch market data from the exchange
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for the data (e.g., '5m', '1h', '1d')
        limit: Number of candles to fetch
        since: Fetch data since this datetime (optional)
        
    Returns:
        DataFrame with OHLCV data if successful, None otherwise
    """
    try:
        logger.info(f"Fetching market data for {symbol} on {timeframe} timeframe, limit={limit}")
        
        # Try to fetch data from exchange
        exchange = get_exchange()
        if exchange is None:
            logger.error("Exchange not initialized")
            # Try fetching from CryptoCompare API
            return fetch_crypto_compare_data(symbol, timeframe, limit, since)
        
        # Try to fetch from exchange first
        try:
            # Convert since to milliseconds timestamp if provided
            since_ts = int(since.timestamp() * 1000) if since else None
            
            # Fetch OHLCV data with timeout handling
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(exchange.fetch_ohlcv, symbol, timeframe, since=since_ts, limit=limit)
                try:
                    ohlcv = future.result(timeout=20)  # 20 second timeout
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout fetching data from exchange for {symbol}")
                    # Try fetching from CryptoCompare API
                    return fetch_crypto_compare_data(symbol, timeframe, limit, since)
                except Exception as fetch_error:
                    logger.warning(f"Error in exchange.fetch_ohlcv: {str(fetch_error)}")
                    return fetch_crypto_compare_data(symbol, timeframe, limit, since)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data returned from exchange for {symbol}")
                # Try fetching from CryptoCompare API
                return fetch_crypto_compare_data(symbol, timeframe, limit, since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Skip database storage to prevent server crashes
            # store_market_data(df, symbol)  # This line is commented out to bypass database operations
            
            return df
            
        except Exception as e:
            logger.warning(f"Error processing exchange data: {str(e)}")
            # Fall back to CryptoCompare API
            return fetch_crypto_compare_data(symbol, timeframe, limit, since)
    
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

def store_market_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Store market data in the database with maximum reliability
    
    This ultra-simplified version focuses on stability over efficiency,
    handling one row at a time to prevent application crashes.
    """
    if df is None or df.empty:
        logger.warning("No data to store in database")
        return
    
    # Import required modules
    from app import db
    from models import MarketData
    import sqlalchemy
    from sqlalchemy import text
    
    # Only keep essential columns to avoid errors
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check if all required columns exist
    if not all(col in df.columns for col in required_columns):
        logger.error(f"DataFrame is missing required columns. Available columns: {df.columns.tolist()}")
        return
    
    # Ensure proper data types to prevent database errors
    try:
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric columns are float type
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
    except Exception as type_error:
        logger.error(f"Error converting data types: {str(type_error)}")
        return
    
    # Process in very small batches (essentially one at a time) for maximum reliability
    total_rows = len(df)
    total_stored = 0
    
    for index, row in df.iterrows():
        try:
            # Skip any rows with NaN values
            if any(pd.isna(row[col]) for col in required_columns):
                logger.debug(f"Skipping row {index} due to NaN values")
                continue
                
            # Process one row at a time with its own transaction
            timestamp_val = row['timestamp']
            
            # Skip rows with invalid timestamps
            if timestamp_val is None:
                logger.debug(f"Skipping row {index} due to invalid timestamp")
                continue
                
            # Use model-based insert instead of raw SQL for better type safety
            # This provides better compatibility across database types
            try:
                # Get timestamp as string in ISO format for consistency
                timestamp_str = timestamp_val.isoformat() if hasattr(timestamp_val, 'isoformat') else str(timestamp_val)
                
                # Use the ORM model directly, skipping the check for existing records
                # to simplify the operation and reduce potential errors
                market_data = MarketData(
                    trading_pair=symbol,
                    timestamp=timestamp_val,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                
                # Use a new session for each insert to isolate transactions
                from sqlalchemy.orm import Session
                with Session(db.engine) as session:
                    session.add(market_data)
                    session.commit()
                    total_stored += 1
                    
                    if total_stored % 50 == 0:
                        logger.info(f"Stored {total_stored}/{total_rows} market data records")
            except sqlalchemy.exc.IntegrityError:
                # Record already exists (primary key or unique constraint violation)
                # This is expected and not an error - skip silently
                pass
            except Exception as insert_error:
                logger.debug(f"Insert error: {str(insert_error)}")
        except Exception as row_error:
            logger.debug(f"Row processing error at index {index}: {str(row_error)}")
            # Continue with next row regardless of errors
    
    logger.info(f"Successfully stored {total_stored}/{total_rows} market data records for {symbol}")
    
    # No need to rollback anything since we're using separate sessions for each insert

def execute_order(symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict]:
    """
    Execute an order on the exchange
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        order_type: Type of order ('market' or 'limit')
        side: Order side ('buy' or 'sell')
        amount: Amount to trade
        price: Price for limit orders
        
    Returns:
        Order details if successful, None otherwise
    """
    config = get_config()
    
    # Check if paper trading is enabled
    if config['paper_trading']:
        return create_paper_trade(symbol, order_type, side, amount, price)
    
    try:
        exchange = get_exchange()
        if exchange is None:
            logger.error("Exchange not initialized")
            return None
        
        # Check if API key is available for trading
        if not exchange.apiKey or not exchange.secret:
            logger.error("API credentials not provided, cannot execute real trades")
            return None
        
        # Execute the order
        if order_type == 'market':
            order = exchange.create_market_order(symbol, side, amount)
        else:  # limit
            if price is None:
                logger.error("Price must be provided for limit orders")
                return None
            order = exchange.create_limit_order(symbol, side, amount, price)
        
        logger.info(f"Order executed: {order}")
        
        # Record the trade
        record_trade(order, symbol, side, amount, price, False)
        
        return order
    
    except Exception as e:
        logger.error(f"Error executing order: {str(e)}")
        return None

def create_paper_trade(symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
    """
    Create a paper trade (simulated trade)
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        order_type: Type of order ('market' or 'limit')
        side: Order side ('buy' or 'sell')
        amount: Amount to trade
        price: Price for limit orders
        
    Returns:
        Simulated order details
    """
    try:
        exchange = get_exchange()
        if exchange is None:
            logger.error("Exchange not initialized")
            return None
        
        # Get current market price if not provided
        if price is None:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
        
        # Generate a paper trade
        trade_id = f"paper-{int(time.time())}-{hash(symbol+side)}"
        order = {
            'id': trade_id,
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'price': price,
            'amount': amount,
            'cost': price * amount,
            'fee': {'cost': price * amount * 0.001, 'currency': symbol.split('/')[1]},  # Simulated 0.1% fee
            'status': 'closed',
            'info': {'paper_trade': True},
        }
        
        logger.info(f"Paper trade created: {order}")
        
        # Record the paper trade
        record_trade(order, symbol, side, amount, price, True)
        
        return order
    
    except Exception as e:
        logger.error(f"Error creating paper trade: {str(e)}")
        return None

def fetch_historical_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 30) -> Optional[List[List[float]]]:
    """
    Fetch historical OHLCV data directly from exchange for a specific symbol
    
    Args:
        symbol: Trading pair symbol (e.g. 'BTC/USDT')
        timeframe: Timeframe for candles (e.g. '1h', '15m')
        limit: Maximum number of candles to fetch
        
    Returns:
        List of OHLCV candles in format [timestamp, open, high, low, close, volume]
        or None if there was an error
    """
    try:
        logger.info(f"Fetching historical OHLCV data for {symbol} on {timeframe} timeframe, limit={limit}")
        
        # Get exchange instance
        exchange_instance = get_exchange()
        if not exchange_instance:
            logger.error("No exchange available to fetch historical data")
            return None
            
        # Fetch OHLCV data with timeout safety
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(exchange_instance.fetch_ohlcv, symbol, timeframe, limit=limit)
            try:
                ohlcv_data = future.result(timeout=15)  # 15 second timeout
                
                if not ohlcv_data or len(ohlcv_data) == 0:
                    logger.warning(f"No OHLCV data returned for {symbol} on {timeframe}")
                    return None
                    
                logger.info(f"Successfully fetched {len(ohlcv_data)} candles for {symbol}")
                return ohlcv_data
                
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout fetching OHLCV data for {symbol}")
                return None
            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to fetch historical OHLCV data: {str(e)}")
        return None

def record_trade(order: Dict, symbol: str, side: str, amount: float, price: float, paper_trade: bool) -> None:
    """Record a trade in the database"""
    try:
        # Get market data and indicators at time of trade
        market_data = MarketData.query.filter_by(trading_pair=symbol).order_by(MarketData.timestamp.desc()).first()
        
        trade = Trade(
            trading_pair=symbol,
            order_id=order['id'],
            trade_type=side,
            entry_price=price,
            amount=amount,
            fee=order.get('fee', {}).get('cost', 0),
            status='open' if side == 'buy' else 'closed',
            paper_trade=paper_trade,
            reason="Trading signal or manual order"
        )
        
        # Add indicator values if available
        if market_data:
            trade.rsi_value = market_data.rsi
            trade.bb_upper = market_data.bb_upper
            trade.bb_middle = market_data.bb_middle
            trade.bb_lower = market_data.bb_lower
            trade.macd = market_data.macd
            trade.macd_signal = market_data.macd_signal
            trade.macd_hist = market_data.macd_hist
            trade.stoch_k = market_data.stoch_k
            trade.stoch_d = market_data.stoch_d
        
        # Set stop loss and take profit
        config = get_config()
        if side == 'buy':
            trade.stop_loss = price * (1 - config['stop_loss_pct'] / 100)
            trade.take_profit = price * (1 + config['take_profit_pct'] / 100)
        
        db.session.add(trade)
        db.session.commit()
    
    except Exception as e:
        logger.error(f"Error recording trade: {str(e)}")
        db.session.rollback()

def close_trade(trade_id: int, price: float, reason: str) -> Optional[Dict]:
    """
    Close a trade
    
    Args:
        trade_id: ID of the trade to close
        price: Closing price
        reason: Reason for closing the trade
        
    Returns:
        Updated trade details if successful, None otherwise
    """
    try:
        trade = Trade.query.get(trade_id)
        if not trade:
            logger.error(f"Trade with ID {trade_id} not found")
            return None
        
        # If trade is already closed
        if trade.status == 'closed':
            logger.warning(f"Trade {trade_id} is already closed")
            return None
        
        # Close the trade
        trade.exit_price = price
        trade.exit_time = datetime.utcnow()
        trade.status = 'closed'
        trade.reason = reason
        
        # Calculate profit/loss
        if trade.trade_type == 'buy':
            trade.profit_loss = (price - trade.entry_price) * trade.amount - trade.fee
            trade.profit_loss_pct = (price / trade.entry_price - 1) * 100
        else:  # sell
            trade.profit_loss = (trade.entry_price - price) * trade.amount - trade.fee
            trade.profit_loss_pct = (trade.entry_price / price - 1) * 100
        
        db.session.commit()
        
        logger.info(f"Trade {trade_id} closed with P/L: {trade.profit_loss:.2f} ({trade.profit_loss_pct:.2f}%)")
        
        return {
            'id': trade.id,
            'trading_pair': trade.trading_pair,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'amount': trade.amount,
            'profit_loss': trade.profit_loss,
            'profit_loss_pct': trade.profit_loss_pct
        }
    
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        db.session.rollback()
        return None

def get_account_balance() -> Dict[str, Any]:
    """Get account balance from the exchange
    
    Returns:
        Dict with keys:
        - total: Dict mapping currency to total amount
        - free: Dict mapping currency to available amount
        - used: Dict mapping currency to locked amount
        - total_btc_value: Total portfolio value in BTC
        - total_usdt_value: Total portfolio value in USDT
        - timestamp: Unix timestamp of the balance fetch
        - exchange_name: Name of the exchange
        - paper_trading: Boolean indicating if paper trading is enabled
    """
    config = get_config()
    
    # Define fallback balance structure to ensure consistent returns
    fallback_balance = {
        'free': {'USDT': 10000},
        'used': {'USDT': 0},
        'total': {'USDT': 10000},
        'total_usdt_value': 10000.0,
        'exchange_name': 'Paper Trading',
        'paper_trading': True,
        'timestamp': int(time.time() * 1000)
    }
    
    # Force exchange to be initialized with API keys
    # We'll reinitialize exchange to make sure we have the latest keys
    global exchange
    
    # Reset the exchange to force reinitialization with proper keys
    exchange = None
    
    # Get KuCoin API credentials - our primary exchange
    kucoin_api_key = os.environ.get('KUCOIN_API_KEY', '') or config.get('kucoin_api_key', '') or config.get('api_key', '')
    kucoin_api_secret = os.environ.get('KUCOIN_API_SECRET', '') or config.get('kucoin_api_secret', '') or config.get('api_secret', '')
    kucoin_passphrase = os.environ.get('KUCOIN_PASSPHRASE', '') or config.get('kucoin_passphrase', '')
    
    # Check credentials for different exchanges
    have_kucoin_creds = kucoin_api_key and kucoin_api_secret and kucoin_passphrase
    
    # For backward compatibility, check Binance credentials too
    binance_api_key = os.environ.get('BINANCE_API_KEY', '')
    binance_api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    # Log the presence of API keys (without showing the actual keys)
    if have_kucoin_creds:
        masked_key = kucoin_api_key[:4] + '*' * (len(kucoin_api_key) - 8) + kucoin_api_key[-4:] if len(kucoin_api_key) > 8 else '****'
        logger.info(f"Found KuCoin API credentials with key: {masked_key}")
    else:
        missing = []
        if not kucoin_api_key:
            missing.append("API key")
        if not kucoin_api_secret:
            missing.append("API secret")
        if not kucoin_passphrase:
            missing.append("API passphrase")
        logger.warning(f"Missing KuCoin API credentials: {', '.join(missing)}")
    
    # Get exchange instance (this will trigger a fresh connection)
    exchange_instance = get_exchange()
    
    # Current paper trading setting
    is_paper_trading = config.get('paper_trading', True)
    
    # Check if this is a valid exchange instance with complete API credentials
    exchange_ready = False
    if exchange_instance:
        if exchange_instance.id == 'kucoin':
            exchange_ready = have_kucoin_creds
        elif exchange_instance.id == 'binance':
            exchange_ready = binance_api_key and binance_api_secret
        else:
            # For other exchanges, just check if we have API credentials
            exchange_ready = exchange_instance.apiKey and exchange_instance.secret
    
    # Always try to get real balance if we have a valid exchange connection with API keys
    if exchange_instance and exchange_ready:
        try:
            # Log which exchange we're connected to
            logger.info(f"Attempting to fetch real balance from {exchange_instance.id}")
            
            # Fetch balance with timeout handling
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(exchange_instance.fetch_balance)
                try:
                    # Use a timeout to prevent hanging
                    balance = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout fetching balance from {exchange_instance.id}")
                    raise Exception(f"Timeout fetching balance from {exchange_instance.id}")
            
            # Extract free and used balances for each currency
            result = {
                'free': {},
                'used': {},
                'total': {},
                'exchange_name': exchange_instance.id.capitalize(),
                'paper_trading': is_paper_trading,  # Still show paper_trading status
                'timestamp': int(time.time() * 1000),
                'api_connected': True  # Flag to indicate successful API connection
            }
            
            # Ensure we have at least one currency in the balance
            if not balance or 'total' not in balance or not balance['total']:
                logger.warning("Empty balance returned from exchange - you may need to fund your account")
                result['free'] = {'USDT': 0}
                result['used'] = {'USDT': 0}
                result['total'] = {'USDT': 0}
                result['total_usdt_value'] = 0.0
                
                # Don't fall back to paper trading when explicitly disabled
                # Just return the empty real balance
                if not is_paper_trading:
                    return result
                else:
                    logger.info("Falling back to paper trading balance due to empty exchange balance")
                    return get_paper_trading_balance()
            
            # Extract balances for each currency
            for currency, data in balance['total'].items():
                if data > 0:
                    result['free'][currency] = balance['free'].get(currency, 0)
                    result['used'][currency] = balance['used'].get(currency, 0)
                    result['total'][currency] = data
            
            # Ensure we have at least one currency in the result
            if not result['total']:
                logger.warning("No currencies with positive balance found")
                result['free'] = {'USDT': 0}
                result['used'] = {'USDT': 0}
                result['total'] = {'USDT': 0}
                result['total_usdt_value'] = 0.0
                
                # Don't fall back to paper trading when explicitly disabled
                if not is_paper_trading:
                    return result
                else:
                    return get_paper_trading_balance()
            
            # Calculate total value in USDT if possible
            if 'USDT' in result['total']:
                result['total_usdt_value'] = result['total']['USDT']
            else:
                # Attempt to calculate total value in USDT
                total_usdt = 0
                for currency, amount in result['total'].items():
                    if currency != 'USDT':
                        try:
                            ticker = exchange_instance.fetch_ticker(f"{currency}/USDT")
                            total_usdt += amount * ticker['last']
                        except Exception as e:
                            logger.warning(f"Error calculating USDT value for {currency}: {str(e)}")
                result['total_usdt_value'] = total_usdt
            
            logger.info(f"Successfully fetched real balance from {exchange_instance.id}: {result['total_usdt_value']} USDT value")
            return result
            
        except Exception as fetch_error:
            error_msg = str(fetch_error)
            logger.error(f"Error fetching real balance: {error_msg}")
            
            # If this is a KuCoin error about missing passphrase, provide a more helpful message
            if "kucoin requires \"password\" credential" in error_msg:
                error_msg = "KuCoin requires an API passphrase. Please add your KuCoin API passphrase to KUCOIN_PASSPHRASE environment variable."
            
            # Fall back to paper trading balance only if paper trading is enabled
            if is_paper_trading:
                logger.info("Falling back to paper trading balance due to fetch error")
                return get_paper_trading_balance()
            else:
                logger.error("Paper trading is disabled, but real balance fetch failed")
                # When paper trading is disabled, we need to show a clear error state
                return {
                    'free': {'USDT': 0},
                    'used': {'USDT': 0},
                    'total': {'USDT': 0},
                    'total_usdt_value': 0.0,
                    'exchange_name': exchange_instance.id.capitalize() if exchange_instance else 'Connection Error',
                    'paper_trading': False,
                    'timestamp': int(time.time() * 1000),
                    'error': error_msg,
                    'api_connected': False
                }
    
    # If we get here, either paper trading is enabled or we couldn't connect to exchange with API keys
    if not is_paper_trading:
        # Paper trading is disabled but we failed to get real balance
        error_msg = "No valid API credentials for exchange"
        
        # Provide more specific error messages based on the exchange
        if exchange_instance:
            if exchange_instance.id == 'kucoin':
                missing = []
                if not kucoin_api_key:
                    missing.append("API key")
                if not kucoin_api_secret:
                    missing.append("API secret")
                if not kucoin_passphrase:
                    missing.append("API passphrase")
                
                if missing:
                    error_msg = f"Missing KuCoin {' and '.join(missing)}"
            else:
                error_msg = f"Missing API credentials for {exchange_instance.id}"
        else:
            error_msg = "Failed to connect to any exchange"
            
        logger.error(f"Paper trading disabled but no real exchange connection: {error_msg}")
        return {
            'free': {'USDT': 0},
            'used': {'USDT': 0},
            'total': {'USDT': 0},
            'total_usdt_value': 0.0,
            'exchange_name': 'API Error',
            'paper_trading': False,
            'timestamp': int(time.time() * 1000),
            'error': error_msg,
            'api_connected': False
        }
    else:
        # Paper trading is enabled - use simulated balance
        try:
            logger.info("Using paper trading balance")
            paper_balance = get_paper_trading_balance()
            if not paper_balance or not isinstance(paper_balance, dict):
                logger.error("Invalid paper trading balance returned")
                return fallback_balance
            
            # Ensure all required fields exist
            for field in ['free', 'used', 'total', 'exchange_name', 'paper_trading', 'timestamp', 'total_usdt_value']:
                if field not in paper_balance:
                    logger.warning(f"Missing field {field} in paper trading balance, using default")
                    if field in ['free', 'used', 'total']:
                        paper_balance[field] = {'USDT': 10000 if field == 'total' or field == 'free' else 0}
                    elif field == 'exchange_name':
                        paper_balance[field] = 'Paper Trading'
                    elif field == 'paper_trading':
                        paper_balance[field] = True
                    elif field == 'timestamp':
                        paper_balance[field] = int(time.time() * 1000)
                    elif field == 'total_usdt_value':
                        paper_balance[field] = 10000.0
            
            # Add API connection status
            paper_balance['api_connected'] = exchange_instance is not None
            
            return paper_balance
        except Exception as e:
            logger.error(f"Error getting paper trading balance: {str(e)}")
            return fallback_balance

def get_paper_trading_balance() -> Dict[str, Any]:
    """Get simulated balance for paper trading
    
    Returns:
        Dict with the same structure as get_account_balance
    """
    try:
        # Start with default paper trading balance
        base_balance = 10000  # Default 10,000 USDT
        
        # Calculate balance changes from trades
        trades = Trade.query.filter_by(paper_trade=True).all()
        
        # Initialize structured balance dictionary
        result = {
            'free': {'USDT': base_balance},
            'used': {'USDT': 0},
            'total': {'USDT': base_balance},
            'total_usdt_value': base_balance,
            'exchange_name': 'Paper Trading',
            'paper_trading': True,
            'timestamp': int(time.time() * 1000)
        }
        
        for trade in trades:
            pair_parts = trade.trading_pair.split('/')
            if len(pair_parts) != 2:
                continue
                
            base_currency = pair_parts[0]  # e.g., BTC
            quote_currency = pair_parts[1]  # e.g., USDT
            
            # Initialize balance for base currency if not exists
            if base_currency not in result['free']:
                result['free'][base_currency] = 0
                result['used'][base_currency] = 0
                result['total'][base_currency] = 0
            
            # Update balances based on trades
            if trade.trade_type == 'buy':
                # Buying base currency (e.g., BTC) with quote currency (e.g., USDT)
                cost = trade.entry_price * trade.amount + trade.fee
                if trade.status == 'open':
                    result['free'][base_currency] += trade.amount
                    result['total'][base_currency] += trade.amount
                    result['free'][quote_currency] -= cost
                    result['total'][quote_currency] -= cost
            else:  # sell
                # Selling base currency (e.g., BTC) for quote currency (e.g., USDT)
                proceeds = trade.entry_price * trade.amount - trade.fee
                if trade.status == 'closed' and trade.exit_price:
                    result['free'][base_currency] -= trade.amount
                    result['total'][base_currency] -= trade.amount
                    result['free'][quote_currency] += proceeds
                    result['total'][quote_currency] += proceeds
        
        # Calculate total USDT value of portfolio
        exchange = get_exchange()
        total_usdt_value = 0
        
        if 'USDT' in result['total']:
            total_usdt_value += result['total']['USDT']
            
        # Try to calculate value of other currencies
        if exchange:
            for currency, amount in result['total'].items():
                if currency != 'USDT' and amount > 0:
                    try:
                        ticker = exchange.fetch_ticker(f"{currency}/USDT")
                        currency_value = amount * ticker['last']
                        total_usdt_value += currency_value
                    except:
                        # If we can't get a price, skip this currency
                        pass
        
        result['total_usdt_value'] = max(total_usdt_value, base_balance)
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating paper trading balance: {str(e)}")
        # Return fallback structure with proper formatting for the dashboard
        return {
            'free': {'USDT': 10000},
            'used': {'USDT': 0},
            'total': {'USDT': 10000},
            'total_usdt_value': 10000.0,
            'exchange_name': 'Paper Trading',
            'paper_trading': True,
            'timestamp': int(time.time() * 1000)
        }
        
def get_market_summary(trading_pair: str) -> Dict[str, Any]:
    """Get market summary including ticker data and indicators for a trading pair"""
    try:
        # Get the exchange
        exchange_instance = get_exchange()
        if not exchange_instance:
            logger.warning(f"Exchange not available for {trading_pair}")
            return {"error": "Exchange not available"}
        
        # Get ticker data
        logger.info(f"Getting market summary for {trading_pair}")
        
        # Fetch ticker with timeout handling
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(exchange_instance.fetch_ticker, trading_pair)
            try:
                ticker = future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout fetching ticker for {trading_pair}")
                return {"error": "Timeout fetching ticker"}
            except Exception as e:
                logger.error(f"Error fetching ticker: {e}")
                return {"error": f"Error fetching ticker: {str(e)}"}
        
        # Try to get indicator data from database
        from models import MarketData
        latest_data = MarketData.query.filter_by(trading_pair=trading_pair).order_by(
            MarketData.timestamp.desc()).first()
        
        indicator_data = {}
        if latest_data:
            # Extract indicators from the latest market data
            indicator_data = {
                'rsi': latest_data.rsi,
                'bb_upper': latest_data.bb_upper,
                'bb_middle': latest_data.bb_middle,
                'bb_lower': latest_data.bb_lower,
                'macd': latest_data.macd,
                'macd_signal': latest_data.macd_signal,
                'macd_hist': latest_data.macd_hist,
                'stoch_k': latest_data.stoch_k,
                'stoch_d': latest_data.stoch_d
            }
        
        # Return complete market summary
        return {
            'ticker': ticker,
            'indicators': indicator_data
        }
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return {"error": str(e)}
        
def fetch_crypto_compare_data(symbol: str, timeframe: str, limit: int = 2000, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from CryptoCompare API
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
        limit: Number of data points to fetch
        start_date: Start date for the data
        end_date: End date for the data (defaults to now)
        
    Returns:
        DataFrame with OHLCV data if successful, None otherwise
    """
    try:
        # Parse symbol to get base and quote currencies
        base_currency, quote_currency = symbol.split('/')
        
        # Map timeframe to CryptoCompare API parameter
        timeframe_map = {
            '1m': 'minute',
            '5m': 'minute',
            '15m': 'minute',
            '30m': 'minute',
            '1h': 'hour',
            '4h': 'hour',
            '1d': 'day'
        }
        
        # Determine aggregation value based on timeframe prefix
        aggregation = 1
        if timeframe.startswith('5'):
            aggregation = 5
        elif timeframe.startswith('15'):
            aggregation = 15
        elif timeframe.startswith('30'):
            aggregation = 30
        elif timeframe.startswith('4'):
            aggregation = 4
            
        api_timeframe = timeframe_map.get(timeframe, 'hour')
        
        # CryptoCompare API base URL
        base_url = 'https://min-api.cryptocompare.com/data/v2/histo'
        
        # Construct URL based on timeframe
        url = f"{base_url}{api_timeframe}"
        
        # Set end date to now if not provided
        if not end_date:
            end_date = datetime.now()
            
        # Set start date to appropriate time in the past if not provided
        if not start_date:
            # Default to an appropriate time range based on the timeframe
            if api_timeframe == 'minute':
                start_date = end_date - timedelta(days=1)  # 1 day of minute data
            elif api_timeframe == 'hour':
                start_date = end_date - timedelta(days=30)  # 30 days of hourly data
            else:  # day
                start_date = end_date - timedelta(days=365)  # 1 year of daily data
        
        # Initialize empty list to hold all data chunks
        all_data = []
        
        # Current timestamp for pagination
        current_to_ts = int(end_date.timestamp())
        
        # Calculate the total time span to determine how many chunks we need
        total_time_span = end_date - start_date
        
        # Calculate number of chunks based on timeframe
        if api_timeframe == 'minute':
            # Each chunk can have max 2000 minutes (approx 1.4 days)
            minutes_per_chunk = 2000 * aggregation
            total_minutes = total_time_span.total_seconds() / 60
            num_chunks = max(1, int(total_minutes / minutes_per_chunk) + 1)
        elif api_timeframe == 'hour':
            # Each chunk can have max 2000 hours (approx 83 days)
            hours_per_chunk = 2000 * aggregation
            total_hours = total_time_span.total_seconds() / 3600
            num_chunks = max(1, int(total_hours / hours_per_chunk) + 1)
        else:  # day
            # Each chunk can have max 2000 days (approx 5.5 years)
            days_per_chunk = 2000 * aggregation
            total_days = total_time_span.days
            num_chunks = max(1, int(total_days / days_per_chunk) + 1)
        
        # Limit number of chunks to avoid excessive API calls
        num_chunks = min(num_chunks, 10)  # Maximum 10 chunks to avoid abuse
        
        logger.info(f"Fetching {num_chunks} chunks of historical data for {symbol} from {start_date} to {end_date}")
        
        # Fetch data in chunks, starting from the most recent
        for chunk in range(num_chunks):
            # Parameters for the API request
            params = {
                'fsym': base_currency,
                'tsym': quote_currency,
                'limit': 2000,  # Always fetch maximum allowed
                'toTs': current_to_ts,
                'aggregate': aggregation
            }
            
            logger.info(f"Fetching historical data chunk {chunk+1}/{num_chunks} from CryptoCompare: {url} with params {params}")
            
            try:
                # Make the API request with timeout
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                
                if data.get('Response') == 'Error':
                    logger.error(f"CryptoCompare API error: {data.get('Message')}")
                    # Don't return None immediately, try to use whatever data we've collected
                    break
                
                # Extract the data from the response
                ohlcv_data = data['Data']['Data']
                
                if not ohlcv_data:
                    logger.warning(f"No data returned for chunk {chunk+1}")
                    break
                
                # Add to our collection
                all_data.extend(ohlcv_data)
                
                # Get the oldest timestamp in this chunk to use as the end time for the next chunk
                # Subtract 1 to avoid overlap
                oldest_timestamp = ohlcv_data[0]['time']
                current_to_ts = oldest_timestamp - 1
                
                # If we've gone past the start date, we can stop
                if pd.to_datetime(oldest_timestamp, unit='s') <= start_date:
                    logger.info(f"Reached start date after {chunk+1} chunks")
                    break
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as chunk_error:
                logger.error(f"Error fetching chunk {chunk+1}: {str(chunk_error)}")
                # Continue with the data we have so far
                break
        
        if not all_data:
            logger.warning("No data returned from CryptoCompare API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume'
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Filter to requested date range
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        # Select only the columns we need
        needed_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in needed_columns if col in df.columns]
        
        # Check if we have all necessary columns
        if not all(col in df.columns for col in needed_columns):
            missing = [col for col in needed_columns if col not in df.columns]
            logger.warning(f"Missing columns in CryptoCompare data: {missing}")
            # Try to fill in missing columns with sensible defaults
            for col in missing:
                if col == 'volume':
                    df[col] = 0.0
                elif col in ['open', 'high', 'low', 'close']:
                    # Use another price column if available, otherwise 0
                    for alt_col in ['close', 'open', 'high', 'low']:
                        if alt_col in df.columns:
                            df[col] = df[alt_col]
                            break
                    else:
                        df[col] = 0.0
        
        # Select only the columns we need
        df = df[needed_columns]
        
        # Remove duplicates if any
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Skip database storage to prevent application crashes - we'll return the data directly
        # store_market_data(df, symbol)  # This line is removed to avoid database errors
        
        logger.info(f"Successfully fetched {len(df)} records from CryptoCompare API")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from CryptoCompare: {str(e)}")
        return None
