"""
Technical indicator calculations for trading bot
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from .candlestick_patterns import CandlestickPatterns, detect_candlestick_patterns

logger = logging.getLogger(__name__)

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)"""
    try:
        # Convert to numpy array to ensure calculations work
        prices = np.array(prices)
        
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        # Calculate RSI for remaining prices
        for i in range(period, len(prices)):
            delta = deltas[i-1]  # The diff is 1 shorter
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return np.zeros_like(prices)

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        # Convert to numpy array to ensure calculations work
        prices = np.array(prices)
        
        # Calculate Bollinger Bands
        middle_band = np.zeros_like(prices)
        upper_band = np.zeros_like(prices)
        lower_band = np.zeros_like(prices)
        
        # Calculate for each point using rolling window
        for i in range(len(prices)):
            if i < period - 1:
                # Not enough data for calculation
                middle_band[i] = prices[i]
                upper_band[i] = prices[i]
                lower_band[i] = prices[i]
            else:
                # Calculate based on previous N prices
                window = prices[i-(period-1):i+1]
                middle_band[i] = np.mean(window)
                band_width = np.std(window) * std_dev
                upper_band[i] = middle_band[i] + band_width
                lower_band[i] = middle_band[i] - band_width
                
        return upper_band, middle_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    try:
        # Convert to numpy array to ensure calculations work
        prices = np.array(prices)
        
        # Calculate EMAs
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        signal_line = calculate_ema(macd_line, signal)
        
        # Calculate histogram (MACD line - signal line)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average (EMA)"""
    try:
        # Convert to numpy array to ensure calculations work
        prices = np.array(prices)
        
        # Calculate EMA
        ema = np.zeros_like(prices)
        
        # First value is a simple average
        ema[period-1] = np.mean(prices[:period])
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
        return ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return np.zeros_like(prices)

def calculate_sma(prices, period):
    """Calculate Simple Moving Average (SMA)"""
    try:
        # Convert to numpy array to ensure calculations work
        prices = np.array(prices)
        
        # Calculate SMA
        sma = np.zeros_like(prices)
        
        # Calculate for each point using rolling window
        for i in range(len(prices)):
            if i < period - 1:
                # Not enough data for calculation
                sma[i] = np.nan
            else:
                # Calculate based on previous N prices
                sma[i] = np.mean(prices[i-(period-1):i+1])
                
        return sma
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return np.zeros_like(prices)

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    try:
        # Convert to numpy arrays to ensure calculations work
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        
        # Initialize K and D line arrays
        k_line = np.zeros_like(closes)
        d_line = np.zeros_like(closes)
        
        # Calculate %K for each point
        for i in range(len(closes)):
            if i < k_period - 1:
                # Not enough data for calculation
                k_line[i] = 50  # Default neutral value
            else:
                # Get high/low/close for window
                window_highs = highs[i-(k_period-1):i+1]
                window_lows = lows[i-(k_period-1):i+1]
                current_close = closes[i]
                
                # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
                highest_high = np.max(window_highs)
                lowest_low = np.min(window_lows)
                
                if highest_high == lowest_low:
                    k_line[i] = 50  # Avoid division by zero
                else:
                    k_line[i] = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
                    
        # Calculate %D (SMA of %K)
        for i in range(len(k_line)):
            if i < k_period - 1 + d_period - 1:
                # Not enough data for calculation
                d_line[i] = 50  # Default neutral value
            else:
                # Calculate simple average of %K values
                d_line[i] = np.mean(k_line[i-(d_period-1):i+1])
                
        return k_line, d_line
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        return np.zeros_like(closes), np.zeros_like(closes)

def calculate_all_indicators(candles, return_full_arrays=False):
    """
    Calculate all indicators from OHLCV candles
    
    Args:
        candles: DataFrame with OHLCV data or list of candle objects
        return_full_arrays: If True, return full arrays for backtesting instead of just the last value
        
    Returns:
        Dictionary of indicator values
    """
    try:
        # Extract price arrays - handle both object attributes and DataFrame/dict-like access
        if isinstance(candles, pd.DataFrame):
            # Handle DataFrame input
            closes = np.array(candles['close'])
            highs = np.array(candles['high'])
            lows = np.array(candles['low'])
            opens = np.array(candles['open'])
        else:
            # Handle object list input (like MarketData objects)
            closes = np.array([c.close for c in candles])
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            opens = np.array([c.open for c in candles])
        
        # Ensure we have enough data for calculations 
        if len(closes) < 14:  # Minimum data for RSI
            logger.warning(f"Not enough data for indicator calculations, only {len(closes)} candles available")
            # Provide empty arrays with matching length for robustness
            zeros = np.zeros(len(closes))
            if return_full_arrays:
                return {
                    'rsi': zeros, 
                    'bb_upper': zeros, 
                    'bb_middle': zeros,
                    'bb_lower': zeros, 
                    'macd': zeros, 
                    'macd_signal': zeros,
                    'macd_hist': zeros, 
                    'stoch_k': zeros, 
                    'stoch_d': zeros
                }
            else:
                # Return last zero value for each
                return {
                    'rsi': 50, 'bb_upper': closes[-1], 'bb_middle': closes[-1],
                    'bb_lower': closes[-1], 'macd': 0, 'macd_signal': 0,
                    'macd_hist': 0, 'stoch_k': 50, 'stoch_d': 50
                }
        
        # Calculate indicators
        try:
            rsi = calculate_rsi(closes)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            rsi = np.zeros(len(closes))
            
        try:
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            bb_upper = bb_middle = bb_lower = np.zeros(len(closes))
            
        try:
            macd, macd_signal, macd_hist = calculate_macd(closes)
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            macd = macd_signal = macd_hist = np.zeros(len(closes))
            
        try:
            stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            stoch_k = stoch_d = np.zeros(len(closes))
        
        # Detect candlestick patterns if we have enough candles
        pattern_signals = {}
        if len(candles) >= 3:
            try:
                # Create DataFrame for candlestick pattern detection
                ohlc_df = pd.DataFrame({
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes
                })
                
                # Get candlestick pattern signals
                pattern_data = detect_candlestick_patterns(ohlc_df)
                
                # Add pattern data to indicators
                pattern_signals = {
                    'candlestick_patterns': pattern_data.get('patterns', ['' for _ in range(len(closes))]),
                    'pattern_action': pattern_data.get('action', 'hold'),
                    'pattern_confidence': pattern_data.get('confidence', 0.0),
                    'pattern_sentiment': pattern_data.get('overall_sentiment', 'neutral')
                }
            except Exception as pattern_e:
                logger.error(f"Error detecting candlestick patterns: {pattern_e}")
        
        # For backtesting, return the full arrays
        if return_full_arrays:
            result = {
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            }
            
            # Add pattern signals if available
            if pattern_signals:
                result.update(pattern_signals)
                
            return result
        else:
            # For regular live trading, return just the latest values
            result = {
                'rsi': rsi[-1] if len(rsi) > 0 else 50.0,
                'bb_upper': bb_upper[-1] if len(bb_upper) > 0 else closes[-1],
                'bb_middle': bb_middle[-1] if len(bb_middle) > 0 else closes[-1],
                'bb_lower': bb_lower[-1] if len(bb_lower) > 0 else closes[-1],
                'macd': macd[-1] if len(macd) > 0 else 0.0,
                'macd_signal': macd_signal[-1] if len(macd_signal) > 0 else 0.0,
                'macd_hist': macd_hist[-1] if len(macd_hist) > 0 else 0.0,
                'stoch_k': stoch_k[-1] if len(stoch_k) > 0 else 50.0,
                'stoch_d': stoch_d[-1] if len(stoch_d) > 0 else 50.0
            }
            
            # Add pattern signals if available
            if pattern_signals:
                # Only take the most recent pattern for live trading
                result.update({
                    'candlestick_patterns': pattern_signals.get('candlestick_patterns', [])[-1] 
                        if isinstance(pattern_signals.get('candlestick_patterns', []), list) 
                           and len(pattern_signals.get('candlestick_patterns', [])) > 0 
                        else [],
                    'pattern_action': pattern_signals.get('pattern_action', 'hold'),
                    'pattern_confidence': pattern_signals.get('pattern_confidence', 0.0),
                    'pattern_sentiment': pattern_signals.get('pattern_sentiment', 'neutral')
                })
                
            return result
    except Exception as e:
        logger.error(f"Error calculating all indicators: {e}")
        # Return empty result with better error handling
        if return_full_arrays and len(closes) > 0:
            zeros = np.zeros(len(closes))
            return {
                'rsi': zeros, 'bb_upper': zeros, 'bb_middle': zeros,
                'bb_lower': zeros, 'macd': zeros, 'macd_signal': zeros,
                'macd_hist': zeros, 'stoch_k': zeros, 'stoch_d': zeros,
                'error': str(e)
            }
        else:
            return {'error': str(e)}

# Alias functions for backward compatibility
calculate_indicators = calculate_all_indicators

def get_indicator_signals(indicators, config=None):
    """Get trading signals from indicators"""
    if not indicators or not isinstance(indicators, dict):
        return {}
    
    # Default config values if not provided
    if not config:
        config = {
            'rsi_oversold': 40.0,  # Increased from 30.0 for more aggressive scalping
            'rsi_overbought': 60.0,  # Decreased from 70.0 for more aggressive scalping
            'stoch_oversold': 30.0,  # Increased from 20.0 for more aggressive scalping
            'stoch_overbought': 70.0,  # Decreased from 80.0 for more aggressive scalping
            'pattern_weight': 0.45,  # Increased from 0.35 for more emphasis on patterns
            'scalping_mode': True  # Enable scalping mode by default
        }
    
    # Extract values with defaults
    rsi_oversold = config.get('rsi_oversold', 40.0)  # Default to scalping values
    rsi_overbought = config.get('rsi_overbought', 60.0)  # Default to scalping values
    stoch_oversold = config.get('stoch_oversold', 30.0)  # Default to scalping values
    stoch_overbought = config.get('stoch_overbought', 70.0)  # Default to scalping values
    pattern_weight = config.get('pattern_weight', 0.45)  # Default to scalping value
    
    # Initialize signals
    signals = {
        'rsi_signal': 'neutral',
        'macd_signal': 'neutral',
        'bb_signal': 'neutral',
        'stoch_signal': 'neutral',
        'pattern_signal': 'neutral',
        'overall_signal': 'neutral',
        'signal_strength': 0.0
    }
    
    # RSI signal
    rsi = indicators.get('rsi')
    if rsi is not None:
        if rsi < rsi_oversold:
            signals['rsi_signal'] = 'buy'
        elif rsi > rsi_overbought:
            signals['rsi_signal'] = 'sell'
    
    # MACD signal
    macd = indicators.get('macd')
    macd_signal_val = indicators.get('macd_signal')
    if macd is not None and macd_signal_val is not None:
        if macd > macd_signal_val:
            signals['macd_signal'] = 'buy'
        elif macd < macd_signal_val:
            signals['macd_signal'] = 'sell'
    
    # Bollinger Bands signal
    price = indicators.get('price', indicators.get('close'))
    bb_lower = indicators.get('bb_lower')
    bb_upper = indicators.get('bb_upper')
    if price is not None and bb_lower is not None and bb_upper is not None:
        if price < bb_lower:
            signals['bb_signal'] = 'buy'
        elif price > bb_upper:
            signals['bb_signal'] = 'sell'
    
    # Stochastic signal
    stoch_k = indicators.get('stoch_k')
    stoch_d = indicators.get('stoch_d')
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < stoch_oversold and stoch_d < stoch_oversold:
            signals['stoch_signal'] = 'buy'
        elif stoch_k > stoch_overbought and stoch_d > stoch_overbought:
            signals['stoch_signal'] = 'sell'
        # Crossover signals (more advanced)
        elif stoch_k > stoch_d and stoch_k < 50:
            signals['stoch_signal'] = 'weak_buy'
        elif stoch_k < stoch_d and stoch_k > 50:
            signals['stoch_signal'] = 'weak_sell'
    
    # Candlestick pattern signal - lower thresholds for scalping
    pattern_action = indicators.get('pattern_action')
    pattern_confidence = indicators.get('pattern_confidence', 0.0)
    if pattern_action:
        if pattern_action == 'buy' and pattern_confidence >= 0.2:  # Reduced from 0.3 to 0.2
            signals['pattern_signal'] = 'buy'
        elif pattern_action == 'sell' and pattern_confidence >= 0.2:  # Reduced from 0.3 to 0.2
            signals['pattern_signal'] = 'sell'
        elif pattern_action == 'buy' and pattern_confidence >= 0.05:  # Reduced from 0.1 to 0.05
            signals['pattern_signal'] = 'weak_buy'
        elif pattern_action == 'sell' and pattern_confidence >= 0.05:  # Reduced from 0.1 to 0.05
            signals['pattern_signal'] = 'weak_sell'
    
    # Add pattern details if available
    pattern_details = {}
    if 'candlestick_patterns' in indicators:
        pattern_details['patterns'] = indicators['candlestick_patterns']
    if 'pattern_sentiment' in indicators:
        pattern_details['sentiment'] = indicators['pattern_sentiment']
    if pattern_details:
        signals['pattern_details'] = pattern_details
    
    # Calculate overall signal
    buy_signals = sum(1 for k, v in signals.items() if k.endswith('_signal') and v == 'buy')
    sell_signals = sum(1 for k, v in signals.items() if k.endswith('_signal') and v == 'sell')
    weak_buy_signals = sum(1 for k, v in signals.items() if k.endswith('_signal') and v == 'weak_buy')
    weak_sell_signals = sum(1 for k, v in signals.items() if k.endswith('_signal') and v == 'weak_sell')
    
    # Weight the signals (with higher weight for pattern signals)
    pattern_signal_value = 0
    if signals['pattern_signal'] == 'buy':
        pattern_signal_value = pattern_weight
    elif signals['pattern_signal'] == 'sell':
        pattern_signal_value = -pattern_weight
    elif signals['pattern_signal'] == 'weak_buy':
        pattern_signal_value = pattern_weight * 0.5
    elif signals['pattern_signal'] == 'weak_sell':
        pattern_signal_value = -pattern_weight * 0.5
    
    # Standard signals have weight 0.25 each (consistent with previous implementation)
    standard_signals_weight = (buy_signals * 0.25) + (weak_buy_signals * 0.1) - (sell_signals * 0.25) - (weak_sell_signals * 0.1)
    
    # Calculate total strength including pattern signals
    total_strength = standard_signals_weight + pattern_signal_value
    signals['signal_strength'] = total_strength
    
    # MODIFIED: Lower thresholds for scalping bot to generate more trade signals
    if total_strength >= 0.2:  # Reduced from 0.4 to 0.2
        signals['overall_signal'] = 'buy'
    elif total_strength <= -0.2:  # Reduced from -0.4 to -0.2
        signals['overall_signal'] = 'sell'
    else:
        signals['overall_signal'] = 'neutral'
    
    return signals

def calculate_and_update_all_indicators(candles, market_data=None):
    """Calculate all indicators and update the market data if provided"""
    try:
        # Calculate indicators
        indicators = calculate_all_indicators(candles)
        
        # Update market data if provided
        if market_data and indicators:
            market_data.rsi = indicators.get('rsi')
            market_data.bb_upper = indicators.get('bb_upper')
            market_data.bb_middle = indicators.get('bb_middle')
            market_data.bb_lower = indicators.get('bb_lower')
            market_data.macd = indicators.get('macd')
            market_data.macd_signal = indicators.get('macd_signal')
            market_data.macd_hist = indicators.get('macd_hist')
            market_data.stoch_k = indicators.get('stoch_k')
            market_data.stoch_d = indicators.get('stoch_d')
        
        return indicators
    except Exception as e:
        logger.error(f"Error calculating and updating indicators: {e}")
        return {}

def get_indicator_features(candles):
    """Extract all indicator features for machine learning model"""
    try:
        # Make sure we have at least 20 candles for reliable indicators (reduced from 30)
        if candles is None:
            logger.warning("No candles provided for feature extraction")
            return pd.DataFrame()  # Return empty DataFrame instead of dict for better compatibility
        
        # Check if we have enough data points
        if isinstance(candles, pd.DataFrame):
            if len(candles) < 20:
                logger.warning(f"Not enough candles for feature extraction: {len(candles)} < 20")
                return pd.DataFrame()
        else:
            if not candles or len(candles) < 20:
                logger.warning(f"Not enough candles for feature extraction: {len(candles) if candles else 0} < 20")
                return pd.DataFrame()
        
        # Extract price arrays - handle both object attributes and DataFrame/dict-like access
        try:
            if isinstance(candles, pd.DataFrame):
                # Handle DataFrame input
                closes = np.array(candles['close'].values) if 'close' in candles else np.array([])
                highs = np.array(candles['high'].values) if 'high' in candles else np.array([])
                lows = np.array(candles['low'].values) if 'low' in candles else np.array([])
                volumes = np.array(candles['volume'].values) if 'volume' in candles else np.array([])
            else:
                # Handle object list input (like MarketData objects)
                closes = np.array([getattr(c, 'close', None) for c in candles if hasattr(c, 'close') and getattr(c, 'close') is not None])
                highs = np.array([getattr(c, 'high', None) for c in candles if hasattr(c, 'high') and getattr(c, 'high') is not None])
                lows = np.array([getattr(c, 'low', None) for c in candles if hasattr(c, 'low') and getattr(c, 'low') is not None])
                volumes = np.array([getattr(c, 'volume', None) for c in candles if hasattr(c, 'volume') and getattr(c, 'volume') is not None])
        except Exception as e:
            logger.error(f"Error extracting price arrays: {e}")
            return pd.DataFrame()
        
        # Verify we have valid price data
        if len(closes) < 20:  # Reduced from 30 to increase feature extraction success rate
            logger.warning(f"Not enough valid close prices for feature extraction: {len(closes)} < 20")
            return pd.DataFrame()
        
        # Calculate indicators
        rsi_values = calculate_rsi(closes)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
        macd_line, macd_signal, macd_hist = calculate_macd(closes)
        stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
        
        # Calculate price features
        returns = np.diff(closes) / closes[:-1]  # Percentage returns
        log_returns = np.diff(np.log(closes))  # Log returns
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        # Calculate volume features
        volume_ma = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
        volume_change = volumes[-1] / volume_ma if volume_ma != 0 else 1.0
        
        # Price trend features
        short_trend = closes[-1] / np.mean(closes[-5:]) if len(closes) >= 5 else 1.0
        medium_trend = closes[-1] / np.mean(closes[-10:]) if len(closes) >= 10 else 1.0
        long_trend = closes[-1] / np.mean(closes[-20:]) if len(closes) >= 20 else 1.0
        
        # Latest indicator values
        latest_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50
        latest_bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if len(bb_middle) > 0 and bb_middle[-1] != 0 else 0
        latest_bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if len(bb_upper) > 0 and (bb_upper[-1] - bb_lower[-1]) != 0 else 0.5
        latest_macd = macd_line[-1] if len(macd_line) > 0 else 0
        latest_macd_signal = macd_signal[-1] if len(macd_signal) > 0 else 0
        latest_macd_hist = macd_hist[-1] if len(macd_hist) > 0 else 0
        latest_stoch_k = stoch_k[-1] if len(stoch_k) > 0 else 50
        latest_stoch_d = stoch_d[-1] if len(stoch_d) > 0 else 50
        
        # Combine all features
        features = {
            # Standard indicators
            'rsi': latest_rsi,
            'bb_position': latest_bb_position,
            'bb_width': latest_bb_width,
            'macd': latest_macd,
            'macd_signal': latest_macd_signal,
            'macd_hist': latest_macd_hist,
            'stoch_k': latest_stoch_k,
            'stoch_d': latest_stoch_d,
            
            # Price and volume features
            'volatility': volatility,
            'volume_change': volume_change,
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            
            # Calculated signals with scalping thresholds
            'rsi_overbought': 1 if latest_rsi > 60 else 0,  # Reduced from 70 to 60
            'rsi_oversold': 1 if latest_rsi < 40 else 0,  # Increased from 30 to 40
            'macd_positive': 1 if latest_macd > 0 else 0,
            'macd_crossover': 1 if latest_macd > latest_macd_signal else 0,
            'stoch_overbought': 1 if latest_stoch_k > 70 and latest_stoch_d > 70 else 0,  # Reduced from 80 to 70
            'stoch_oversold': 1 if latest_stoch_k < 30 and latest_stoch_d < 30 else 0,  # Increased from 20 to 30
            'price_above_middle_bb': 1 if len(bb_middle) > 0 and closes[-1] > bb_middle[-1] else 0.5,
            
            # Target features (for supervised learning)
            'future_return': 0.0  # Placeholder to be filled by the calling function
        }
        
        # Convert dict to DataFrame for better compatibility with ML functions
        features_df = pd.DataFrame([features])
        return features_df
    except Exception as e:
        logger.error(f"Error extracting indicator features: {e}")
        # Return empty DataFrame instead of empty dict
        return pd.DataFrame()