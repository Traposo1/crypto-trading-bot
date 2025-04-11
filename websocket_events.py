import logging
import threading
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from flask_socketio import emit
from app import socketio, db
from bot.exchange import get_exchange, get_exchange_name
from bot.utils import calculate_performance_metrics, get_market_summary
from config import get_config

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
active_connections = 0
ticker_thread = None
ticker_running = False

def get_ticker_data(trading_pair: str) -> Dict[str, Any]:
    """Get ticker data from the exchange"""
    try:
        # Get the exchange
        exchange = get_exchange()
        if not exchange:
            logger.warning(f"Exchange not available for {trading_pair}")
            return {"error": "Exchange not available"}
        
        # Fetch ticker data with timeout handling
        logger.info(f"Fetching WebSocket ticker data for {trading_pair}")
        
        import concurrent.futures
        
        def fetch_ticker_with_timeout():
            try:
                return exchange.fetch_ticker(trading_pair)
            except Exception as e:
                logger.error(f"Error in fetch_ticker_with_timeout: {str(e)}")
                return None
        
        # Use a thread with timeout to prevent hanging on API calls
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_ticker_with_timeout)
            try:
                ticker = future.result(timeout=10)  # 10 second timeout
                if not ticker:
                    raise Exception("Failed to fetch ticker data")
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout fetching ticker for {trading_pair}")
                raise Exception(f"Timeout fetching ticker for {trading_pair}")
        
        # Handle different volume field names
        volume = 0
        if 'volume' in ticker:
            volume = ticker['volume']
        elif 'baseVolume' in ticker:
            volume = ticker['baseVolume']
        elif 'quoteVolume' in ticker:
            volume = ticker['quoteVolume']
        elif 'vol' in ticker:
            volume = ticker['vol']
        
        # Create data packet
        return {
            'symbol': trading_pair,
            'last': ticker['last'],
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'high': ticker['high'],
            'low': ticker['low'],
            'volume': volume,
            'change': ticker['percentage'] if 'percentage' in ticker else None,
            'change_amount': ticker['change'] if 'change' in ticker else None,
            'timestamp': ticker['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Error fetching WebSocket ticker data: {str(e)}")
        # Try to get fallback data from database
        try:
            from models import MarketData
            from app import db
            
            # Use retry mechanism for database access
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    market_data = MarketData.query.filter_by(trading_pair=trading_pair).order_by(MarketData.timestamp.desc()).first()
                    
                    if market_data:
                        # Found market data, use it as fallback
                        market_summary = get_market_summary(trading_pair)
                        price_change_pct = market_summary.get('price_change_pct', 0) if market_summary and 'price_change_pct' in market_summary else 0
                        
                        return {
                            'symbol': trading_pair,
                            'last': market_data.close,
                            'bid': market_data.close * 0.999,  # Simulated
                            'ask': market_data.close * 1.001,  # Simulated
                            'high': market_data.high,
                            'low': market_data.low,
                            'volume': market_data.volume,
                            'change': price_change_pct,
                            'change_amount': market_summary.get('price_change', 0) if market_summary else 0,
                            'timestamp': int(market_data.timestamp.timestamp() * 1000),
                            'source': 'database_fallback'
                        }
                    break  # No data or success, break out of retry loop
                    
                except Exception as db_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to get market data after {max_retries} attempts: {str(db_error)}")
                    else:
                        logger.warning(f"Database access attempt {retry_count} failed: {str(db_error)}. Retrying...")
                        time.sleep(0.5 * retry_count)
                        
                        # Rollback any failed transaction
                        try:
                            db.session.rollback()
                        except:
                            pass
        except Exception as fallback_error:
            logger.error(f"Error getting fallback data: {str(fallback_error)}")
            
        # Return error if all attempts failed
        return {"error": str(e), "source": "api_error"}

def get_indicator_data(trading_pair: str, force_recalculate: bool = False) -> Dict[str, Any]:
    """Get indicator data from market summary or calculate on the fly
    
    Args:
        trading_pair: Trading pair symbol (BTC/USDT, etc)
        force_recalculate: Force recalculation of indicators even if stored values exist
        
    Returns:
        Dictionary of indicator values, including candlestick patterns if detected
    """
    try:
        # If not forcing recalculation, try to get already calculated indicators from market summary
        if not force_recalculate:
            market_summary = get_market_summary(trading_pair)
            if market_summary and 'indicators' in market_summary and all(
                market_summary['indicators'].get(key) is not None 
                for key in ['rsi', 'macd', 'bb_upper']
            ):
                logger.info(f"Using stored indicators for {trading_pair}")
                return market_summary['indicators']
        
        # If indicators are missing, null, or force_recalculate is True, calculate them from market data
        try:
            from bot import indicators
            from models import MarketData
            
            logger.info(f"{'Forcing recalculation' if force_recalculate else 'Calculating'} of indicators for {trading_pair}")
            
            # First check if we need to fetch data
            candles = MarketData.query.filter_by(trading_pair=trading_pair).order_by(
                MarketData.timestamp.desc()).limit(100).all()

            # If not enough data, try to fetch it from exchange
            if len(candles) < 15:
                try:
                    logger.warning(f"Not enough market data for {trading_pair}, attempting to fetch from exchange")
                    # Get the exchange
                    exchange = get_exchange()
                    if not exchange:
                        logger.error("No exchange available to fetch historical data")
                        raise Exception("Exchange not available")
                        
                    # Fetch OHLCV data with timeout safety
                    import concurrent.futures
                    ohlcv_data = None
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(exchange.fetch_ohlcv, trading_pair, "1h", limit=30)
                        try:
                            ohlcv_data = future.result(timeout=15)  # 15 second timeout
                            
                            if not ohlcv_data or len(ohlcv_data) == 0:
                                logger.warning(f"No OHLCV data returned for {trading_pair}")
                                raise Exception("No data returned from exchange")
                                
                            logger.info(f"Successfully fetched {len(ohlcv_data)} candles for {trading_pair}")
                            
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout fetching OHLCV data for {trading_pair}")
                            raise Exception("Timeout fetching data")
                        except Exception as e:
                            logger.error(f"Error fetching OHLCV data: {str(e)}")
                            raise
                    
                    # We already have ohlcv_data from the code above, no need to call fetch_historical_ohlcv
                    
                    if ohlcv_data is not None and len(ohlcv_data) > 0:
                        logger.info(f"Successfully fetched {len(ohlcv_data)} candles from exchange")
                        
                        # Save to database - but don't commit immediately to avoid blocking the response
                        for candle in ohlcv_data:
                            timestamp, open_price, high, low, close, volume = candle
                            
                            # Check if this candle already exists
                            existing = MarketData.query.filter_by(
                                trading_pair=trading_pair, 
                                timestamp=datetime.fromtimestamp(timestamp/1000)
                            ).first()
                            
                            if not existing:
                                new_candle = MarketData(
                                    trading_pair=trading_pair,
                                    timestamp=datetime.fromtimestamp(timestamp/1000),
                                    open=open_price,
                                    high=high,
                                    low=low,
                                    close=close,
                                    volume=volume
                                )
                                db.session.add(new_candle)
                        
                        # Commit in a separate thread to avoid blocking
                        import threading
                        def commit_data():
                            try:
                                db.session.commit()
                                logger.info(f"Successfully saved {len(ohlcv_data)} candles to database")
                            except Exception as e:
                                db.session.rollback()
                                logger.error(f"Error saving candles to database: {str(e)}")
                        
                        threading.Thread(target=commit_data).start()
                        
                        # Refresh our candles
                        candles = MarketData.query.filter_by(trading_pair=trading_pair).order_by(
                            MarketData.timestamp.desc()).limit(100).all()
                except Exception as fetch_error:
                    logger.error(f"Error fetching market data: {str(fetch_error)}")
            
            # Now try to calculate with available data, using fewer periods if necessary
            if candles and len(candles) >= 5:  # Reduce minimum required candles to 5
                # Sort by timestamp ascending for indicator calculation
                candles = sorted(candles, key=lambda x: x.timestamp)
                
                # Extract price data
                closes = [c.close for c in candles if c.close]
                highs = [c.high for c in candles if c.high]
                lows = [c.low for c in candles if c.low]
                
                # Use variable periods based on available data
                data_length = len(closes)
                logger.info(f"Calculating indicators with {data_length} data points for {trading_pair}")
                
                # Adjust periods based on available data
                rsi_period = min(14, max(2, data_length // 3))
                bb_period = min(20, max(2, data_length // 3))
                macd_fast = min(12, max(2, data_length // 6))
                macd_slow = min(26, max(macd_fast + 1, data_length // 3))
                macd_signal = min(9, max(2, data_length // 8))
                stoch_k_period = min(14, max(2, data_length // 3))
                stoch_d_period = min(3, max(2, data_length // 10))
                
                # Calculate indicators with adjusted periods
                try:
                    rsi = indicators.calculate_rsi(closes, period=rsi_period)
                    bb_upper, bb_middle, bb_lower = indicators.calculate_bollinger_bands(closes, period=bb_period, std_dev=2)
                    macd, macd_signal, macd_hist = indicators.calculate_macd(closes, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                    stoch_k, stoch_d = indicators.calculate_stochastic(highs, lows, closes, k_period=stoch_k_period, d_period=stoch_d_period)
                    
                    # Create result dictionary with standard indicators
                    result = {
                        'rsi': rsi[-1] if rsi and len(rsi) > 0 else 50.0,
                        'bb_upper': bb_upper[-1] if bb_upper and len(bb_upper) > 0 else closes[-1] * 1.02,
                        'bb_middle': bb_middle[-1] if bb_middle and len(bb_middle) > 0 else closes[-1],
                        'bb_lower': bb_lower[-1] if bb_lower and len(bb_lower) > 0 else closes[-1] * 0.98,
                        'macd': macd[-1] if macd and len(macd) > 0 else 0.0,
                        'macd_signal': macd_signal[-1] if macd_signal and len(macd_signal) > 0 else 0.0,
                        'macd_hist': macd_hist[-1] if macd_hist and len(macd_hist) > 0 else 0.0,
                        'stoch_k': stoch_k[-1] if stoch_k and len(stoch_k) > 0 else 50.0,
                        'stoch_d': stoch_d[-1] if stoch_d and len(stoch_d) > 0 else 50.0,
                    }
                    
                    # Log the calculated indicators
                    logger.info(f"Successfully calculated indicators for {trading_pair} with {data_length} data points")
                except Exception as indicator_error:
                    logger.error(f"Error calculating indicators: {str(indicator_error)}")
                    # Create default values if calculation fails
                    result = {
                        'rsi': 50.0,
                        'bb_upper': closes[-1] * 1.02 if closes else 0,
                        'bb_middle': closes[-1] if closes else 0, 
                        'bb_lower': closes[-1] * 0.98 if closes else 0,
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_hist': 0.0,
                        'stoch_k': 50.0,
                        'stoch_d': 50.0,
                    }
                    
                    # Add candlestick pattern detection if there's enough data
                    try:
                        import pandas as pd
                        from bot.candlestick_patterns import detect_candlestick_patterns
                        
                        # Extract OHLC data for pattern detection
                        opens = [c.open for c in candles if c.open]
                        
                        # Verify we have enough data for pattern detection
                        logger.info(f"Attempting pattern detection for {trading_pair} with {len(opens)} data points")
                        
                        if len(opens) >= 10:  # Need at least 10 candles for meaningful patterns
                            # Create DataFrame for last 30 candles (or all we have, sufficient for pattern detection)
                            last_n_candles = pd.DataFrame({
                                'open': opens[-30:] if len(opens) >= 30 else opens,
                                'high': highs[-30:] if len(highs) >= 30 else highs,
                                'low': lows[-30:] if len(lows) >= 30 else lows,
                                'close': closes[-30:] if len(closes) >= 30 else closes
                            })
                            
                            # Log data for debugging
                            logger.info(f"Pattern detection dataframe for {trading_pair} has shape: {last_n_candles.shape}")
                            
                            # TEMPORARY: For testing, create a mock pattern result for debugging purposes
                            # We're doing this because we need to debug the structure that should be returned
                            if force_recalculate:
                                logger.info("Creating test pattern data for debugging")
                                # Match the structure in CandlestickPatterns.get_trade_signals() return value
                                pattern_data = {
                                    "action": "buy",
                                    "confidence": 0.7,
                                    "explanation": "Test pattern detection",
                                    "patterns": ["bullish_engulfing", "three_white_soldiers"],
                                    "overall_sentiment": "bullish"
                                }
                                logger.info(f"Created test pattern data: {pattern_data}")
                            else:
                                # Normal path - use actual pattern detection
                                pattern_data = detect_candlestick_patterns(last_n_candles)
                            
                            # Log pattern detection results
                            logger.info(f"Pattern detection results for {trading_pair}: {pattern_data}")
                            
                            # Add pattern data to results
                            if pattern_data:
                                # Make sure to display the pattern data for debugging
                                logger.info(f"Pattern detection raw data: {pattern_data}")
                                
                                # Add pattern data to results
                                result.update({
                                    'candlestick_patterns': pattern_data.get('patterns', []),
                                    'pattern_action': pattern_data.get('action', 'hold'),
                                    'pattern_confidence': pattern_data.get('confidence', 0.0),
                                    'pattern_sentiment': pattern_data.get('overall_sentiment', 'neutral')
                                })
                                logger.info(f"Added candlestick pattern data for {trading_pair}: {pattern_data.get('patterns', [])}")
                            else:
                                logger.warning(f"No pattern data returned for {trading_pair}")
                        else:
                            logger.warning(f"Not enough data points for pattern detection: {len(opens)} < 10 required")
                    except Exception as pattern_error:
                        logger.error(f"Error detecting candlestick patterns: {str(pattern_error)}")
                        import traceback
                        logger.error(f"Pattern detection traceback: {traceback.format_exc()}")
                    
                    # Return all indicator data
                    logger.info(f"Calculated fresh indicators for {trading_pair}")
                    return result
        except Exception as calc_error:
            logger.error(f"Error calculating indicators on-the-fly: {str(calc_error)}")
        
        # If calculation failed, try to get market summary again
        market_summary = get_market_summary(trading_pair)
        
        # If no market data found or calculation failed, log the issue and return empty data
        if market_summary and 'error' in market_summary and 'No market data found' in market_summary['error']:
            logger.warning(f"No market data for {trading_pair}, unable to provide indicators")
            return {}
        
        # Return either the existing indicators (which may have some nulls) or empty dict
        return market_summary.get('indicators', {}) if market_summary else {}
    except Exception as e:
        logger.error(f"Error fetching WebSocket indicator data: {str(e)}")
        return {}

def ticker_update_loop():
    """Background thread to send ticker updates"""
    global ticker_running
    
    logger.info("Starting WebSocket ticker update loop")
    ticker_running = True
    
    # Need to import here to avoid circular import
    from app import app
    
    # Run within application context
    with app.app_context():
        try:
            while ticker_running and active_connections > 0:
                try:
                    # Get configurations
                    config = get_config()
                    # Get all active trading pairs
                    trading_pairs = config.get('active_pairs', 'BTC/USDT').split(',')
                    
                    # Fetch and emit data for each pair
                    for pair in trading_pairs:
                        pair = pair.strip()
                        if not pair:
                            continue
                        
                        ticker_data = get_ticker_data(pair)
                        indicator_data = get_indicator_data(pair)
                        
                        # Only emit if we have both ticker and indicator data
                        if ticker_data and not ticker_data.get('error'):
                            # Emit ticker update
                            socketio.emit('market_update', {
                                'type': 'ticker_update',
                                'data': ticker_data
                            })
                            
                            # Only emit indicators if we have indicator data
                            if indicator_data:
                                socketio.emit('market_update', {
                                    'type': 'indicator_update',
                                    'data': {
                                        'symbol': pair,
                                        'indicators': indicator_data
                                    }
                                })
                        
                        # Brief pause between pairs to prevent rate limiting
                        time.sleep(0.2)
                        
                    # Wait before next update (5 seconds)
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in ticker update loop: {str(e)}")
                    time.sleep(5)  # Wait before retry on error
                    
        except Exception as e:
            logger.error(f"Ticker update loop terminated with error: {str(e)}")
        
        finally:
            ticker_running = False
            logger.info("WebSocket ticker update loop stopped")

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    global active_connections, ticker_thread, ticker_running
    
    active_connections += 1
    logger.info(f"Client connected. Active connections: {active_connections}")
    
    # Start the ticker thread if not already running
    if ticker_thread is None or not ticker_thread.is_alive():
        ticker_thread = threading.Thread(target=ticker_update_loop)
        ticker_thread.daemon = True
        ticker_thread.start()
        logger.info("Started ticker update thread")
    
    # Send initial status
    emit('connection_status', {
        'connected': True,
        'exchange': get_exchange_name(),
        'time': int(time.time() * 1000)
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    global active_connections
    
    active_connections = max(0, active_connections - 1)
    logger.info(f"Client disconnected. Active connections: {active_connections}")
    
    # No need to stop the thread as it will exit when there are no active connections

@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription to a specific trading pair"""
    trading_pair = data.get('pair', data.get('symbol', 'BTC/USDT'))
    logger.info(f"Client subscribed to {trading_pair}")
    
    # Send immediate update for the subscribed pair
    ticker_data = get_ticker_data(trading_pair)
    indicator_data = get_indicator_data(trading_pair)
    
    if ticker_data and not ticker_data.get('error'):
        emit('market_update', {
            'type': 'ticker_update',
            'data': ticker_data
        })
        
    if indicator_data:
        emit('market_update', {
            'type': 'indicator_update',
            'data': {
                'symbol': trading_pair,
                'indicators': indicator_data
            }
        })

@socketio.on('refresh_indicators')
def handle_refresh_indicators(data):
    """Handle request to refresh indicators for a specific trading pair"""
    trading_pair = data.get('pair', data.get('symbol', 'BTC/USDT'))
    force_refresh = data.get('force_refresh', True)  # Default to True
    test_patterns = data.get('test_patterns', False)  # Check if this is a pattern test request
    
    logger.info(f"Client requested indicator refresh for {trading_pair} (force={force_refresh}, test_patterns={test_patterns})")
    
    try:
        # Get fresh indicator data with force_recalculate=True
        indicator_data = get_indicator_data(trading_pair, force_recalculate=force_refresh)
        
        # If this is a pattern test request and no patterns were detected
        if test_patterns and (not indicator_data.get('candlestick_patterns') or len(indicator_data.get('candlestick_patterns', [])) == 0):
            logger.info(f"Pattern test requested but no patterns detected. Creating test pattern data for {trading_pair}")
            
            # Add test pattern data for debugging
            indicator_data.update({
                'candlestick_patterns': ['bullish_engulfing', 'three_white_soldiers'],
                'pattern_action': 'buy',
                'pattern_confidence': 0.75,
                'pattern_sentiment': 'bullish'
            })
            logger.info(f"Added test pattern data: {indicator_data}")
        
        if indicator_data:
            # Send the updated indicators back
            emit('market_update', {
                'type': 'indicator_update',
                'data': {
                    'symbol': trading_pair,
                    'indicators': indicator_data,
                    'refreshed': True,
                    'patterns_tested': test_patterns
                }
            })
            logger.info(f"Sent refreshed indicators for {trading_pair}")
            return True
        else:
            logger.warning(f"Failed to refresh indicators for {trading_pair} - no data returned")
            return False
    except Exception as e:
        logger.error(f"Error refreshing indicators via WebSocket: {str(e)}")
        return False