import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from bot.indicators import get_indicator_signals

# Configure logging
logger = logging.getLogger(__name__)

def generate_signals(df: pd.DataFrame, prediction: Optional[int], confidence: Optional[float], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate trading signals based on technical indicators and ML predictions
    
    Args:
        df: DataFrame with OHLCV data and indicators
        prediction: ML model prediction (1 for up, 0 for down/no change, None if ML is disabled)
        confidence: ML prediction confidence score
        config: Configuration dictionary with indicator parameters
        
    Returns:
        Dictionary with signal information
    """
    try:
        # Enhanced error checking
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning("Empty or None dataframe provided to generate_signals")
            return {
                'buy': False,
                'sell': False,
                'timestamp': None,
                'price': None,
                'indicators': {},
                'ml': {'prediction': None, 'confidence': None, 'signal': None},
                'strength': 0,
                'reasons': ["Error: Empty or invalid dataframe"]
            }
        
        # Check for minimum number of rows required
        if len(df) < 5:  # Need at least a few candles for minimal calculations
            logger.warning(f"Not enough data points for signal generation, only {len(df)} rows available")
            return {
                'buy': False,
                'sell': False,
                'timestamp': None,
                'price': None,
                'indicators': {},
                'ml': {'prediction': None, 'confidence': None, 'signal': None},
                'strength': 0,
                'reasons': [f"Error: Not enough data points ({len(df)} rows)"]
            }
        
        try:
            # Safely extract the last row values as Python native types to avoid Series issues
            latest_data = df.iloc[-1].to_dict()
            
            # Verify required price data is available
            if 'close' not in latest_data or pd.isna(latest_data['close']):
                logger.warning("Missing closing price in latest data point")
                return {
                    'buy': False,
                    'sell': False,
                    'timestamp': None,
                    'price': None,
                    'indicators': {},
                    'ml': {'prediction': None, 'confidence': None, 'signal': None},
                    'strength': 0,
                    'reasons': ["Error: Missing closing price data"]
                }
        except Exception as row_error:
            logger.error(f"Error extracting latest row data: {str(row_error)}")
            return {
                'buy': False,
                'sell': False,
                'timestamp': latest_data.get('timestamp') if 'timestamp' in latest_data else None,
                'price': latest_data.get('close') if 'close' in latest_data else None,
                'strength': 0,
                'reasons': [f"Error extracting row data: {str(row_error)}"]
            }
        
        # Extract indicator values from the DataFrame with proper None handling
        indicators = {
            'rsi': latest_data.get('rsi', None),
            'bb_upper': latest_data.get('bb_upper', None),
            'bb_middle': latest_data.get('bb_middle', None),
            'bb_lower': latest_data.get('bb_lower', None),
            'macd': latest_data.get('macd', None),
            'macd_signal': latest_data.get('macd_signal', None),
            'macd_hist': latest_data.get('macd_hist', None),
            'stoch_k': latest_data.get('stoch_k', None),
            'stoch_d': latest_data.get('stoch_d', None),
            'price': latest_data.get('close', None)
        }
        
        # Add presence checks for debugging
        missing_indicators = [k for k, v in indicators.items() if v is None or pd.isna(v)]
        if missing_indicators:
            logger.warning(f"Missing indicators in signal generation: {', '.join(missing_indicators)}")
        
        # Get signals from each indicator
        indicator_signals = get_indicator_signals(indicators, config)
        
        # Initialize signal object
        signal = {
            'buy': False,
            'sell': False,
            'timestamp': latest_data['timestamp'],
            'price': float(latest_data['close']),
            'indicators': indicator_signals,
            'ml': {
                'prediction': prediction,
                'confidence': confidence,
                'signal': False  # Will be updated below if conditions are met
            },
            'strength': 0,  # Signal strength from 0 to 1
            'reasons': []
        }
        
        # Set ML signal properly - avoiding pandas Series truth value ambiguity
        if prediction is not None and confidence is not None:
            ml_threshold = float(config['ml_confidence_threshold'])
            signal['ml']['signal'] = (prediction == 1 and confidence >= ml_threshold)
        
        # Combine indicator signals
        signal_strength = 0
        buy_count = 0
        sell_count = 0
        
        # RSI signals
        if 'rsi' in indicator_signals:
            rsi_signal = indicator_signals['rsi']
            if rsi_signal.get('oversold', False):
                buy_count += 1
                signal['reasons'].append(f"RSI oversold ({rsi_signal['value']:.2f})")
            elif rsi_signal.get('overbought', False):
                sell_count += 1
                signal['reasons'].append(f"RSI overbought ({rsi_signal['value']:.2f})")
            elif rsi_signal.get('bullish', False):
                buy_count += 0.5
                signal['reasons'].append(f"RSI bullish momentum ({rsi_signal['value']:.2f})")
            elif rsi_signal.get('bearish', False):
                sell_count += 0.5
                signal['reasons'].append(f"RSI bearish momentum ({rsi_signal['value']:.2f})")
        
        # Bollinger Bands signals
        if 'bollinger' in indicator_signals:
            bb_signal = indicator_signals['bollinger']
            if bb_signal.get('price_below_lower', False):
                buy_count += 1
                signal['reasons'].append(f"Price below BB lower band (percent_b: {bb_signal['percent_b']:.2f})")
            elif bb_signal.get('price_above_upper', False):
                sell_count += 1
                signal['reasons'].append(f"Price above BB upper band (percent_b: {bb_signal['percent_b']:.2f})")
        
        # MACD signals
        if 'macd' in indicator_signals:
            macd_signal = indicator_signals['macd']
            macd_bullish = macd_signal.get('bullish_crossover', False)
            macd_bearish = macd_signal.get('bearish_crossover', False)
            macd_positive = macd_signal.get('positive_histogram', False) 
            macd_increasing = macd_signal.get('histogram_increasing', False)
            
            if macd_bullish:
                buy_count += 1
                signal['reasons'].append(f"MACD bullish crossover (histogram: {macd_signal['histogram']:.6f})")
            elif macd_bearish:
                sell_count += 1
                signal['reasons'].append(f"MACD bearish crossover (histogram: {macd_signal['histogram']:.6f})")
            elif macd_positive and macd_increasing:
                buy_count += 0.5
                signal['reasons'].append(f"MACD histogram increasing positive (histogram: {macd_signal['histogram']:.6f})")
            elif not macd_positive and not macd_increasing:
                sell_count += 0.5
                signal['reasons'].append(f"MACD histogram decreasing negative (histogram: {macd_signal['histogram']:.6f})")
        
        # Stochastic signals
        if 'stochastic' in indicator_signals:
            stoch_signal = indicator_signals['stochastic']
            stoch_oversold = stoch_signal.get('oversold', False)
            stoch_overbought = stoch_signal.get('overbought', False)
            stoch_bullish = stoch_signal.get('bullish_crossover', False)
            stoch_bearish = stoch_signal.get('bearish_crossover', False)
            
            if stoch_oversold and stoch_bullish:
                buy_count += 1
                signal['reasons'].append(f"Stochastic oversold bullish crossover (K: {stoch_signal['k']:.2f}, D: {stoch_signal['d']:.2f})")
            elif stoch_overbought and stoch_bearish:
                sell_count += 1
                signal['reasons'].append(f"Stochastic overbought bearish crossover (K: {stoch_signal['k']:.2f}, D: {stoch_signal['d']:.2f})")
        
        # Moving Averages signals
        if 'moving_averages' in indicator_signals:
            ma_signal = indicator_signals['moving_averages']
            
            # Golden/Death Cross (stronger signals)
            if ma_signal.get('golden_cross', False):
                buy_count += 1.25
                signal['reasons'].append(f"Golden Cross (SMA50 > SMA200)")
            elif ma_signal.get('death_cross', False):
                sell_count += 1.25
                signal['reasons'].append(f"Death Cross (SMA50 < SMA200)")
                
            # Fast EMA crossovers (shorter-term signals)
            if ma_signal.get('ema_fast_bullish', False):
                buy_count += 0.75
                signal['reasons'].append(f"Fast EMA bullish (EMA9 > EMA21)")
            
            # Price relative to key moving averages
            if ma_signal.get('price_above_ema_50', False) and ma_signal.get('price_above_ema_200', False):
                buy_count += 0.5
                signal['reasons'].append(f"Price above key EMAs (50 & 200)")
            elif not ma_signal.get('price_above_ema_50', True) and not ma_signal.get('price_above_ema_200', True):
                sell_count += 0.5
                signal['reasons'].append(f"Price below key EMAs (50 & 200)")
                
        # VWAP signals
        if 'vwap' in indicator_signals:
            vwap_signal = indicator_signals['vwap']
            
            if vwap_signal.get('price_above_vwap', False):
                buy_count += 0.5
                signal['reasons'].append(f"Price above VWAP")
            elif vwap_signal.get('price_below_vwap', False):
                sell_count += 0.5
                signal['reasons'].append(f"Price below VWAP")
                
        # ADX signals (trend strength)
        if 'adx' in indicator_signals:
            adx_signal = indicator_signals['adx']
            
            # Strong trend with directional bias
            if adx_signal.get('strong_trend', False):
                if adx_signal.get('bullish_trend', False):
                    buy_count += 1
                    signal['reasons'].append(f"Strong bullish trend (ADX: {adx_signal['value']:.2f})")
                elif adx_signal.get('bearish_trend', False):
                    sell_count += 1
                    signal['reasons'].append(f"Strong bearish trend (ADX: {adx_signal['value']:.2f})")
            
            # Very strong trend signals
            if adx_signal.get('very_strong_trend', False):
                if adx_signal.get('bullish_trend', False):
                    buy_count += 0.5
                    signal['reasons'].append(f"Very strong bullish trend")
                elif adx_signal.get('bearish_trend', False):
                    sell_count += 0.5
                    signal['reasons'].append(f"Very strong bearish trend")
                
        # ML signals - avoid pandas Series truth value ambiguity
        ml_enabled = bool(config.get('ml_enabled', False))
        if ml_enabled and prediction is not None and confidence is not None:
            ml_threshold = float(config['ml_confidence_threshold'])
            if prediction == 1 and confidence >= ml_threshold:
                buy_count += 1.5  # ML prediction has higher weight
                signal['reasons'].append(f"ML predicts price increase (confidence: {confidence:.2f})")
            elif prediction == 0 and confidence >= ml_threshold:
                sell_count += 1.5  # ML prediction has higher weight
                signal['reasons'].append(f"ML predicts price decrease (confidence: {confidence:.2f})")
        
        # Calculate signal strength based on buy and sell counts
        total_possible_signals = 10.0 if ml_enabled else 8.5  # Adjusted for new indicators
        buy_strength = buy_count / total_possible_signals
        sell_strength = sell_count / total_possible_signals
        
        # MODIFIED: Lower thresholds for scalping strategy - a scalping bot should be more aggressive
        # and trade more frequently with smaller profit targets
        if buy_strength >= 0.2 and buy_strength > sell_strength:  # Reduced from 0.4 to 0.2
            signal['buy'] = True
            signal['strength'] = buy_strength
            logger.info(f"BUY signal generated with strength {buy_strength:.2f} at price {signal['price']}")
        elif sell_strength >= 0.2 and sell_strength > buy_strength:  # Reduced from 0.4 to 0.2
            signal['sell'] = True
            signal['strength'] = sell_strength
            logger.info(f"SELL signal generated with strength {sell_strength:.2f} at price {signal['price']}")
        
        return signal
    
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        timestamp = None
        price = None
        
        if not df.empty:
            try:
                timestamp = df.iloc[-1]['timestamp']
                price = float(df.iloc[-1]['close'])
            except:
                pass
                
        return {
            'buy': False,
            'sell': False,
            'timestamp': timestamp,
            'price': price,
            'indicators': {},
            'ml': {'prediction': None, 'confidence': None, 'signal': None},
            'strength': 0,
            'reasons': [f"Error: {str(e)}"]
        }

def filter_signals(signal: Dict[str, Any], open_trades: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter signals based on existing trades and risk management
    
    Args:
        signal: Original signal dictionary
        open_trades: List of open trades
        config: Configuration dictionary
        
    Returns:
        Updated signal dictionary
    """
    filtered_signal = signal.copy()
    
    # Check if maximum number of trades is reached
    if filtered_signal['buy'] and len(open_trades) >= config['max_open_trades']:
        filtered_signal['buy'] = False
        filtered_signal['reasons'].append(f"Maximum open trades reached ({config['max_open_trades']})")
    
    # Check if we already have an open trade for this pair
    for trade in open_trades:
        if trade['trading_pair'] == config['trading_pair'] and trade['trade_type'] == 'buy':
            # Already have an open buy position, can't buy again
            if filtered_signal['buy']:
                filtered_signal['buy'] = False
                filtered_signal['reasons'].append("Already have an open position for this pair")
            break
    
    # Can't sell if we don't have an open buy position
    has_open_position = any(trade['trading_pair'] == config['trading_pair'] and trade['trade_type'] == 'buy' for trade in open_trades)
    if filtered_signal['sell'] and not has_open_position:
        filtered_signal['sell'] = False
        filtered_signal['reasons'].append("No open position to sell")
    
    return filtered_signal

def scalping_strategy(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implement a scalping strategy based on short-term price movements and indicators
    
    Args:
        df: DataFrame with OHLCV data and indicators
        config: Configuration dictionary
        
    Returns:
        Signal dictionary
    """
    try:
        # Ensure we have enough data - but with reduced requirement for aggressive scalping
        # Only require 10 candles instead of 20 to generate signals earlier in the backtest
        if len(df) < 10:
            return {'buy': False, 'sell': False, 'reasons': ["Not enough data for scalping strategy"]}
        
        # Get config values with fallbacks - ultra aggressive for scalping
        rsi_period = int(config.get('rsi_period', 4))
        # Use more extreme thresholds for scalping in backtests
        rsi_oversold_threshold = float(config.get('rsi_oversold', 45.0))  # Increased for more buy signals
        rsi_overbought_threshold = float(config.get('rsi_overbought', 55.0))  # Decreased for more sell signals
        
        # Get latest price data
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest  # Fallback if only one row
        
        # Initialize signal
        signal = {
            'buy': False,
            'sell': False,
            'timestamp': latest['timestamp'] if 'timestamp' in latest else pd.Timestamp.now(),
            'price': latest['close'] if 'close' in latest else 0,
            'strength': 0,
            'reasons': []
        }
        
        # Track signal strength for later decision
        buy_points = 0
        sell_points = 0
        
        # Ultra short-term price movement - using even shorter periods for ultra-responsive scalping
        try:
            # Use ultra-short-term moving averages for scalping
            short_term_ma = df['close'].rolling(2).mean().iloc[-1]  # Ultra-short 2-period MA
            medium_term_ma = df['close'].rolling(5).mean().iloc[-1]  # Short 5-period MA
            long_term_ma = df['close'].rolling(8).mean().iloc[-1]  # Medium 8-period MA
            
            # Price relative to MAs
            price_above_short_ma = latest['close'] > short_term_ma
            price_above_medium_ma = latest['close'] > medium_term_ma
            
            # MA trends
            short_ma_trend_up = short_term_ma > medium_term_ma
            medium_ma_trend_up = medium_term_ma > long_term_ma
            
            # Combined signals
            if price_above_short_ma and short_ma_trend_up:
                buy_points += 1
                signal['reasons'].append(f"Price above short MA and rising")
            elif not price_above_short_ma and not short_ma_trend_up:
                sell_points += 1
                signal['reasons'].append(f"Price below short MA and falling")
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {e}")
            # Fallback values
            price_above_short_ma = False
            price_above_medium_ma = False
            short_ma_trend_up = False
            medium_ma_trend_up = False
        
        # RSI for detecting potential reversals - even wider thresholds for more frequent signals
        # Check if 'rsi' column exists and handle missing values
        if 'rsi' in df.columns:
            try:
                rsi_value = latest.get('rsi')
                if rsi_value is not None and not pd.isna(rsi_value):
                    rsi_oversold = rsi_value < rsi_oversold_threshold
                    rsi_overbought = rsi_value > rsi_overbought_threshold
                else:
                    # Use default values if RSI is missing at this position
                    rsi_oversold = False
                    rsi_overbought = False
                    logger.warning("RSI value missing at the current position, using default signal values")
            except Exception as e:
                logger.warning(f"Error processing RSI value: {e}")
                rsi_oversold = False
                rsi_overbought = False
        else:
            logger.warning("RSI column not found in DataFrame, using default signal values")
            rsi_oversold = False
            rsi_overbought = False
        
        # Volume analysis - more responsive for ultra-quick trades
        volume_increase = latest['volume'] > df['volume'].rolling(2).mean().iloc[-1]  # Further reduced from 3 to 2
        # Check for volume spike - stronger signal for aggressive scalping
        volume_spike = latest['volume'] > df['volume'].rolling(3).mean().iloc[-1] * 1.3  # Reduced multiplier from 1.5 to 1.3
        
        # Price velocity - check if price is moving quickly in one direction
        price_velocity = (latest['close'] - prev['close']) / prev['close']
        rapid_price_increase = price_velocity > 0.001  # 0.1% quick rise
        rapid_price_decrease = price_velocity < -0.001  # 0.1% quick drop
        
        # Bollinger Bands squeeze (narrowing bands indicate potential breakout)
        bb_squeeze = False
        
        # Check if BB values exist
        if all(x in latest and not pd.isna(latest[x]) for x in ['bb_upper', 'bb_lower', 'bb_middle']):
            try:
                bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle'] if latest['bb_middle'] != 0 else 0
                
                # Only calculate if BB columns exist in the DataFrame
                if all(x in df.columns for x in ['bb_upper', 'bb_lower', 'bb_middle']):
                    # Use even shorter period for ultra-fast response to band squeezes
                    avg_upper = df['bb_upper'].rolling(5).mean().iloc[-1]  # Further reduced from 10 to 5
                    avg_lower = df['bb_lower'].rolling(5).mean().iloc[-1]  # Further reduced from 10 to 5
                    avg_middle = df['bb_middle'].rolling(5).mean().iloc[-1]  # Further reduced from 10 to 5
                    avg_width = (avg_upper - avg_lower) / avg_middle if avg_middle != 0 else 0
                    bb_squeeze = bb_width < avg_width * 0.95  # Changed from 0.9 to 0.95 to detect squeezes even earlier
            except Exception as e:
                logger.warning(f"Error calculating BB squeeze: {str(e)}")
                bb_squeeze = False
        else:
            logger.warning("BB values missing, skipping BB squeeze calculation")
        
        # Price near Bollinger Bands - more lenient definitions for more trade signals
        near_lower_bb = False
        near_upper_bb = False
        
        # Only calculate if BB values exist
        if all(x in latest and not pd.isna(latest[x]) for x in ['bb_upper', 'bb_lower']):
            near_lower_bb = latest['close'] < latest['bb_lower'] * 1.02  # Increased from 1.01 to 1.02
            near_upper_bb = latest['close'] > latest['bb_upper'] * 0.98  # Decreased from 0.99 to 0.98
        
        # Bollinger Band breakout detection - stronger signal for aggressive scalping
        bb_breakout_up = False
        bb_breakout_down = False
        
        # Only calculate if BB values exist in both current and previous candles
        if all(x in latest and x in prev and not pd.isna(latest[x]) and not pd.isna(prev[x]) 
               for x in ['bb_upper', 'bb_lower']):
            bb_breakout_up = prev['close'] < prev['bb_upper'] and latest['close'] > latest['bb_upper']
            bb_breakout_down = prev['close'] > prev['bb_lower'] and latest['close'] < latest['bb_lower']
        
        # Generate buy signal - Aggressive but with some quality filters
        # Only need one condition to trigger a buy, but must have some quality check
        buy_condition1 = price_above_short_ma and short_ma_trend_up  # Basic trend following
        buy_condition2 = rsi_oversold  # RSI oversold condition
        buy_condition3 = near_lower_bb and volume_increase  # Bottom bounce with volume
        buy_condition4 = bb_squeeze  # Potential breakout setup
        buy_condition5 = bb_breakout_up  # Breakout above Bollinger Band
        buy_condition6 = rapid_price_increase and volume_increase  # Momentum with volume
        
        # Track reasons for diagnostics
        buy_reasons = []
        
        if price_above_short_ma and short_ma_trend_up:
            buy_reasons.append("Price above short MA and rising")
            
        if price_above_medium_ma:
            buy_reasons.append("Price above medium MA")
            
        if rsi_oversold:
            buy_reasons.append(f"RSI oversold ({latest.get('rsi', 0):.2f} < {rsi_oversold_threshold:.2f})")
            
        if near_lower_bb:
            buy_reasons.append("Price near lower Bollinger Band")
            
        if bb_squeeze:
            buy_reasons.append("Bollinger Band squeeze (potential breakout)")
            
        if volume_spike:
            buy_reasons.append("Volume spike detected")
            
        if bb_breakout_up:
            buy_reasons.append("Bullish Bollinger Band breakout")
            
        if rapid_price_increase:
            buy_reasons.append("Rapid price increase detected")
        
        # Generate buy signal - we need to be aggressive but with basic filtering
        # ORIGINAL strategy that worked better - more aggressive with lower thresholds
        buy_condition = False
        
        # Best original conditions from successful tests
        if price_above_short_ma:
            buy_condition = True
            if not buy_reasons:
                buy_reasons.append("Price above short MA")
        
        if rsi_oversold:
            buy_condition = True
            if "RSI oversold" not in str(buy_reasons):
                buy_reasons.append(f"RSI oversold ({latest.get('rsi', 0):.2f})")
                
        if near_lower_bb:
            buy_condition = True
            if not buy_reasons:
                buy_reasons.append("Price near lower BB")
        
        # Generate buy signal with minimal conditions (very aggressive)
        if buy_condition and buy_reasons:
            signal['buy'] = True
            signal['strength'] = 0.9  # Ultra-high confidence for aggressive strategy
            signal['reasons'] = buy_reasons
        
        # Generate sell signal with higher quality criteria
        # Quality sell conditions with higher win rate potential
        sell_condition1 = not price_above_short_ma and not price_above_medium_ma  # Confirmed downtrend
        sell_condition2 = rsi_overbought and near_upper_bb  # Multiple confirmations
        sell_condition3 = bb_breakout_down and rapid_price_decrease  # Confirmed breakdown
        
        # Track sell strength based on multiple confirmations
        sell_strength = 0
        sell_reasons = []
        
        if not price_above_short_ma and not short_ma_trend_up:
            sell_strength += 0.3
            sell_reasons.append("Price below short MA and falling")
            
        if not price_above_medium_ma and not medium_ma_trend_up:
            sell_strength += 0.2
            sell_reasons.append("Price below medium MA with negative trend")
            
        if rsi_overbought:
            sell_strength += 0.25
            sell_reasons.append(f"RSI overbought ({latest.get('rsi', 0):.2f} > {rsi_overbought_threshold:.2f})")
            
        if near_upper_bb:
            sell_strength += 0.25
            sell_reasons.append("Price near upper Bollinger Band")
            
        if bb_squeeze and not price_above_short_ma:
            sell_strength += 0.2
            sell_reasons.append("Bollinger Band squeeze with bearish bias")
            
        if rapid_price_decrease:
            sell_strength += 0.2
            sell_reasons.append("Rapid price decrease detected")
            
        if bb_breakout_down:
            sell_strength += 0.3
            sell_reasons.append("Bearish Bollinger Band breakout")
        
        # More intelligent profit taking
        take_profit_condition = False
        take_profit_reason = ""
        
        # Check for significant profit opportunity
        for i in range(min(5, len(df)-1)):
            profit_pct = (latest['close'] / df.iloc[-(i+2)]['close'] - 1) * 100
            # Take profit if more than 0.8% gain in past 5 candles
            if profit_pct > 0.8:
                take_profit_condition = True
                take_profit_reason = f"Taking profit after {profit_pct:.2f}% gain"
                sell_strength += 0.4  # Strong incentive to take profit
                break
        
        if take_profit_condition:
            sell_reasons.append(take_profit_reason)
        
        # Generate sell signal only with sufficient confirmation
        # Require higher threshold (0.5) for quality sell signals
        if sell_strength >= 0.5 and (sell_condition1 or sell_condition2 or sell_condition3 or take_profit_condition):
            signal['sell'] = True
            signal['strength'] = min(0.95, sell_strength)  # Cap at 0.95
            signal['reasons'] = sell_reasons
        
        return signal
    
    except Exception as e:
        logger.error(f"Error in scalping strategy: {str(e)}")
        return {'buy': False, 'sell': False, 'reasons': [f"Error in scalping strategy: {str(e)}"]}
