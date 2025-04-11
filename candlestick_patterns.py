"""
Candlestick Pattern Recognition Module

This module provides functions to identify common Japanese candlestick patterns
in price data. It uses rule-based pattern detection with configurable sensitivity.
"""

from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import pandas as pd


class CandlestickPatterns:
    """
    Class to identify candlestick patterns in price data.
    Uses mathematical definitions for pattern identification.
    """
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize the candlestick pattern detector.
        
        Args:
            tolerance: Percentage tolerance for pattern detection (default 5%)
                       Higher values allow more flexibility in pattern recognition
        """
        self.tolerance = tolerance
        self.patterns = {
            "doji": self._is_doji,
            "hammer": self._is_hammer,
            "inverted_hammer": self._is_inverted_hammer,
            "bullish_engulfing": self._is_bullish_engulfing,
            "bearish_engulfing": self._is_bearish_engulfing,
            "morning_star": self._is_morning_star,
            "evening_star": self._is_evening_star,
            "shooting_star": self._is_shooting_star,
            "hanging_man": self._is_hanging_man,
            "three_white_soldiers": self._is_three_white_soldiers,
            "three_black_crows": self._is_three_black_crows,
            "bullish_harami": self._is_bullish_harami,
            "bearish_harami": self._is_bearish_harami,
            "piercing_line": self._is_piercing_line,
            "dark_cloud_cover": self._is_dark_cloud_cover,
        }
        
        # Classification of patterns by bullish/bearish/neutral
        self.bullish_patterns = [
            "hammer", "inverted_hammer", "bullish_engulfing", 
            "morning_star", "three_white_soldiers", "bullish_harami",
            "piercing_line"
        ]
        
        self.bearish_patterns = [
            "shooting_star", "hanging_man", "bearish_engulfing", 
            "evening_star", "three_black_crows", "bearish_harami",
            "dark_cloud_cover"
        ]
        
        self.neutral_patterns = ["doji"]
        
    def _is_doji(self, candle: pd.Series) -> bool:
        """
        Check if the candle is a doji (open and close are very close).
        
        Args:
            candle: Pandas Series with OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        # Doji has a very small body compared to the total range
        if candle_range > 0:
            body_to_range_ratio = body_size / candle_range
            return body_to_range_ratio <= 0.1  # Body is less than 10% of range
        
        return False
    
    def _is_hammer(self, candle: pd.Series, prev_trend: str = 'down') -> bool:
        """
        Check if the candle is a hammer (small body, long lower shadow, small upper shadow).
        
        Args:
            candle: Pandas Series with OHLC data
            prev_trend: Previous market trend, should be 'down' for a valid hammer
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return False
        
        body_to_range_ratio = body_size / candle_range
        
        # Get upper and lower shadows
        if candle['close'] >= candle['open']:  # Bullish candle
            upper_shadow = candle['high'] - candle['close']
            lower_shadow = candle['open'] - candle['low']
        else:  # Bearish candle
            upper_shadow = candle['high'] - candle['open']
            lower_shadow = candle['close'] - candle['low']
        
        # Hammer criteria:
        # 1. Small body (typically less than 30% of total range)
        # 2. Long lower shadow (at least 2x the body size)
        # 3. Small or no upper shadow (less than 10% of total range)
        # 4. Should appear in a downtrend
        
        is_small_body = body_to_range_ratio <= 0.3
        is_long_lower_shadow = lower_shadow >= 2 * body_size
        is_small_upper_shadow = upper_shadow <= 0.1 * candle_range
        
        return (is_small_body and is_long_lower_shadow and 
                is_small_upper_shadow and prev_trend == 'down')
    
    def _is_inverted_hammer(self, candle: pd.Series, prev_trend: str = 'down') -> bool:
        """
        Check if the candle is an inverted hammer (small body, long upper shadow, small lower shadow).
        
        Args:
            candle: Pandas Series with OHLC data
            prev_trend: Previous market trend, should be 'down' for a valid inverted hammer
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return False
        
        body_to_range_ratio = body_size / candle_range
        
        # Get upper and lower shadows
        if candle['close'] >= candle['open']:  # Bullish candle
            upper_shadow = candle['high'] - candle['close']
            lower_shadow = candle['open'] - candle['low']
        else:  # Bearish candle
            upper_shadow = candle['high'] - candle['open']
            lower_shadow = candle['close'] - candle['low']
        
        # Inverted hammer criteria:
        # 1. Small body (typically less than 30% of total range)
        # 2. Long upper shadow (at least 2x the body size)
        # 3. Small or no lower shadow (less than 10% of total range)
        # 4. Should appear in a downtrend
        
        is_small_body = body_to_range_ratio <= 0.3
        is_long_upper_shadow = upper_shadow >= 2 * body_size
        is_small_lower_shadow = lower_shadow <= 0.1 * candle_range
        
        return (is_small_body and is_long_upper_shadow and 
                is_small_lower_shadow and prev_trend == 'down')
    
    def _is_shooting_star(self, candle: pd.Series, prev_trend: str = 'up') -> bool:
        """
        Check if the candle is a shooting star (small body, long upper shadow, small lower shadow).
        
        Args:
            candle: Pandas Series with OHLC data
            prev_trend: Previous market trend, should be 'up' for a valid shooting star
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Shooting star is structurally similar to inverted hammer but appears in an uptrend
        return self._is_inverted_hammer(candle, 'up') and prev_trend == 'up'
    
    def _is_hanging_man(self, candle: pd.Series, prev_trend: str = 'up') -> bool:
        """
        Check if the candle is a hanging man (small body, long lower shadow, small upper shadow).
        
        Args:
            candle: Pandas Series with OHLC data
            prev_trend: Previous market trend, should be 'up' for a valid hanging man
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Hanging man is structurally similar to hammer but appears in an uptrend
        return self._is_hammer(candle, 'up') and prev_trend == 'up'
    
    def _is_bullish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a bullish engulfing pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Bullish engulfing criteria:
        # 1. Previous candle is bearish (close < open)
        # 2. Current candle is bullish (close > open)
        # 3. Current candle's body completely engulfs previous candle's body
        
        if previous['close'] >= previous['open']:  # Previous candle must be bearish
            return False
        
        if current['close'] <= current['open']:  # Current candle must be bullish
            return False
        
        # Check engulfing (current body completely engulfs previous body)
        return (current['open'] <= previous['close'] and 
                current['close'] >= previous['open'])
    
    def _is_bearish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a bearish engulfing pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Bearish engulfing criteria:
        # 1. Previous candle is bullish (close > open)
        # 2. Current candle is bearish (close < open)
        # 3. Current candle's body completely engulfs previous candle's body
        
        if previous['close'] <= previous['open']:  # Previous candle must be bullish
            return False
        
        if current['close'] >= current['open']:  # Current candle must be bearish
            return False
        
        # Check engulfing (current body completely engulfs previous body)
        return (current['open'] >= previous['close'] and 
                current['close'] <= previous['open'])
    
    def _is_morning_star(self, candles: List[pd.Series]) -> bool:
        """
        Check if three candles form a morning star pattern.
        
        Args:
            candles: List of 3 candles' OHLC data (oldest to newest)
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        if len(candles) < 3:
            return False
        
        # Morning star criteria:
        # 1. First candle is bearish with a large body
        # 2. Second candle has a small body and gaps down
        # 3. Third candle is bullish with a large body, closing into the first candle
        
        first, second, third = candles[-3], candles[-2], candles[-1]
        
        # First candle is bearish with large body
        first_bearish = first['close'] < first['open']
        first_body_size = abs(first['close'] - first['open'])
        first_range = first['high'] - first['low']
        first_large_body = first_body_size > 0.5 * first_range
        
        # Second candle has small body
        second_body_size = abs(second['close'] - second['open'])
        second_range = second['high'] - second['low']
        second_small_body = second_body_size < 0.3 * second_range
        
        # Gap down between first and second candles
        gap_down = max(second['open'], second['close']) < min(first['open'], first['close'])
        
        # Third candle is bullish with large body
        third_bullish = third['close'] > third['open']
        third_body_size = abs(third['close'] - third['open'])
        third_range = third['high'] - third['low']
        third_large_body = third_body_size > 0.5 * third_range
        
        # Third candle closes into the first candle's body
        closes_into_first = third['close'] > (first['open'] + first['close']) / 2
        
        return (first_bearish and first_large_body and 
                second_small_body and 
                third_bullish and third_large_body and 
                closes_into_first)
    
    def _is_evening_star(self, candles: List[pd.Series]) -> bool:
        """
        Check if three candles form an evening star pattern.
        
        Args:
            candles: List of 3 candles' OHLC data (oldest to newest)
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        if len(candles) < 3:
            return False
        
        # Evening star criteria:
        # 1. First candle is bullish with a large body
        # 2. Second candle has a small body and gaps up
        # 3. Third candle is bearish with a large body, closing into the first candle
        
        first, second, third = candles[-3], candles[-2], candles[-1]
        
        # First candle is bullish with large body
        first_bullish = first['close'] > first['open']
        first_body_size = abs(first['close'] - first['open'])
        first_range = first['high'] - first['low']
        first_large_body = first_body_size > 0.5 * first_range
        
        # Second candle has small body
        second_body_size = abs(second['close'] - second['open'])
        second_range = second['high'] - second['low']
        second_small_body = second_body_size < 0.3 * second_range
        
        # Gap up between first and second candles
        gap_up = min(second['open'], second['close']) > max(first['open'], first['close'])
        
        # Third candle is bearish with large body
        third_bearish = third['close'] < third['open']
        third_body_size = abs(third['close'] - third['open'])
        third_range = third['high'] - third['low']
        third_large_body = third_body_size > 0.5 * third_range
        
        # Third candle closes into the first candle's body
        closes_into_first = third['close'] < (first['open'] + first['close']) / 2
        
        return (first_bullish and first_large_body and 
                second_small_body and 
                third_bearish and third_large_body and 
                closes_into_first)
    
    def _is_three_white_soldiers(self, candles: List[pd.Series]) -> bool:
        """
        Check if three candles form a three white soldiers pattern.
        
        Args:
            candles: List of 3 candles' OHLC data (oldest to newest)
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        if len(candles) < 3:
            return False
        
        # Three white soldiers criteria:
        # 1. Three consecutive bullish candles with large bodies
        # 2. Each candle opens within the previous candle's body
        # 3. Each candle closes higher than the previous candle
        # 4. Small or no upper shadows
        
        for i in range(3):
            if candles[i]['close'] <= candles[i]['open']:  # Must be bullish
                return False
            
            body_size = candles[i]['close'] - candles[i]['open']
            candle_range = candles[i]['high'] - candles[i]['low']
            
            if body_size < 0.5 * candle_range:  # Must have large body
                return False
            
            upper_shadow = candles[i]['high'] - candles[i]['close']
            if upper_shadow > 0.2 * body_size:  # Small upper shadow
                return False
            
            if i > 0:
                # Opens within previous body
                if not (candles[i]['open'] > candles[i-1]['open'] and 
                        candles[i]['open'] < candles[i-1]['close']):
                    return False
                
                # Closes higher than previous
                if candles[i]['close'] <= candles[i-1]['close']:
                    return False
        
        return True
    
    def _is_three_black_crows(self, candles: List[pd.Series]) -> bool:
        """
        Check if three candles form a three black crows pattern.
        
        Args:
            candles: List of 3 candles' OHLC data (oldest to newest)
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        if len(candles) < 3:
            return False
        
        # Three black crows criteria:
        # 1. Three consecutive bearish candles with large bodies
        # 2. Each candle opens within the previous candle's body
        # 3. Each candle closes lower than the previous candle
        # 4. Small or no lower shadows
        
        for i in range(3):
            if candles[i]['close'] >= candles[i]['open']:  # Must be bearish
                return False
            
            body_size = candles[i]['open'] - candles[i]['close']
            candle_range = candles[i]['high'] - candles[i]['low']
            
            if body_size < 0.5 * candle_range:  # Must have large body
                return False
            
            lower_shadow = candles[i]['close'] - candles[i]['low']
            if lower_shadow > 0.2 * body_size:  # Small lower shadow
                return False
            
            if i > 0:
                # Opens within previous body
                if not (candles[i]['open'] < candles[i-1]['open'] and 
                        candles[i]['open'] > candles[i-1]['close']):
                    return False
                
                # Closes lower than previous
                if candles[i]['close'] >= candles[i-1]['close']:
                    return False
        
        return True
    
    def _is_bullish_harami(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a bullish harami pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Bullish harami criteria:
        # 1. Previous candle is bearish with a large body
        # 2. Current candle is bullish with a small body
        # 3. Current candle's body is completely inside the previous candle's body
        
        if previous['close'] >= previous['open']:  # Previous candle must be bearish
            return False
        
        if current['close'] <= current['open']:  # Current candle must be bullish
            return False
        
        prev_body_size = previous['open'] - previous['close']
        curr_body_size = current['close'] - current['open']
        
        # Current body is smaller than previous
        if curr_body_size >= prev_body_size:
            return False
        
        # Current body is inside previous body
        return (current['open'] > previous['close'] and 
                current['close'] < previous['open'])
    
    def _is_bearish_harami(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a bearish harami pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Bearish harami criteria:
        # 1. Previous candle is bullish with a large body
        # 2. Current candle is bearish with a small body
        # 3. Current candle's body is completely inside the previous candle's body
        
        if previous['close'] <= previous['open']:  # Previous candle must be bullish
            return False
        
        if current['close'] >= current['open']:  # Current candle must be bearish
            return False
        
        prev_body_size = previous['close'] - previous['open']
        curr_body_size = current['open'] - current['close']
        
        # Current body is smaller than previous
        if curr_body_size >= prev_body_size:
            return False
        
        # Current body is inside previous body
        return (current['open'] < previous['close'] and 
                current['close'] > previous['open'])
    
    def _is_piercing_line(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a piercing line pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Piercing line criteria:
        # 1. Previous candle is bearish
        # 2. Current candle is bullish
        # 3. Current candle opens below previous candle's low
        # 4. Current candle closes above the midpoint of previous candle's body
        
        if previous['close'] >= previous['open']:  # Previous candle must be bearish
            return False
        
        if current['close'] <= current['open']:  # Current candle must be bullish
            return False
        
        # Current opens below previous low
        if current['open'] >= previous['low']:
            return False
        
        # Current closes above the midpoint of previous body
        prev_midpoint = (previous['open'] + previous['close']) / 2
        return current['close'] > prev_midpoint
    
    def _is_dark_cloud_cover(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if the current and previous candles form a dark cloud cover pattern.
        
        Args:
            current: Current candle's OHLC data
            previous: Previous candle's OHLC data
            
        Returns:
            True if the pattern is detected, False otherwise
        """
        # Dark cloud cover criteria:
        # 1. Previous candle is bullish
        # 2. Current candle is bearish
        # 3. Current candle opens above previous candle's high
        # 4. Current candle closes below the midpoint of previous candle's body
        
        if previous['close'] <= previous['open']:  # Previous candle must be bullish
            return False
        
        if current['close'] >= current['open']:  # Current candle must be bearish
            return False
        
        # Current opens above previous high
        if current['open'] <= previous['high']:
            return False
        
        # Current closes below the midpoint of previous body
        prev_midpoint = (previous['open'] + previous['close']) / 2
        return current['close'] < prev_midpoint
    
    def identify_patterns(self, ohlc: pd.DataFrame, trend_periods: int = 10) -> Dict[str, List[int]]:
        """
        Identify all candlestick patterns in the given OHLC data.
        
        Args:
            ohlc: DataFrame with OHLC price data
            trend_periods: Number of periods to determine trend
            
        Returns:
            Dictionary mapping pattern names to lists of indices where patterns appear
        """
        if len(ohlc) < 3:
            return {}
        
        # Calculate simple trend (using EMA)
        try:
            ema = ohlc['close'].ewm(span=trend_periods, adjust=False).mean()
            trend = ['up' if ohlc['close'].iloc[i] > ema.iloc[i] else 'down' 
                    for i in range(len(ohlc))]
        except:
            # Fallback to simple moving average if EMA fails
            sma = ohlc['close'].rolling(window=trend_periods, min_periods=1).mean()
            trend = ['up' if ohlc['close'].iloc[i] > sma.iloc[i] else 'down' 
                    for i in range(len(ohlc))]
        
        results = {pattern: [] for pattern in self.patterns.keys()}
        
        # Single-candle patterns
        for i in range(len(ohlc)):
            if i >= 1:  # Need at least one previous candle for context
                current = ohlc.iloc[i]
                prev_trend = trend[i-1]
                
                # Check single-candle patterns
                if self._is_doji(current):
                    results['doji'].append(i)
                
                if self._is_hammer(current, prev_trend):
                    results['hammer'].append(i)
                
                if self._is_inverted_hammer(current, prev_trend):
                    results['inverted_hammer'].append(i)
                
                if self._is_shooting_star(current, prev_trend):
                    results['shooting_star'].append(i)
                
                if self._is_hanging_man(current, prev_trend):
                    results['hanging_man'].append(i)
                
                # Check two-candle patterns
                previous = ohlc.iloc[i-1]
                
                if self._is_bullish_engulfing(current, previous):
                    results['bullish_engulfing'].append(i)
                
                if self._is_bearish_engulfing(current, previous):
                    results['bearish_engulfing'].append(i)
                
                if self._is_bullish_harami(current, previous):
                    results['bullish_harami'].append(i)
                
                if self._is_bearish_harami(current, previous):
                    results['bearish_harami'].append(i)
                
                if self._is_piercing_line(current, previous):
                    results['piercing_line'].append(i)
                
                if self._is_dark_cloud_cover(current, previous):
                    results['dark_cloud_cover'].append(i)
                
            if i >= 2:  # Need at least three candles for some patterns
                # Check three-candle patterns
                candles = [ohlc.iloc[i-2], ohlc.iloc[i-1], ohlc.iloc[i]]
                
                if self._is_morning_star(candles):
                    results['morning_star'].append(i)
                
                if self._is_evening_star(candles):
                    results['evening_star'].append(i)
        
        # Check multi-candle patterns that need specific sequences
        if len(ohlc) >= 3:
            for i in range(2, len(ohlc)):
                candles = [ohlc.iloc[i-2], ohlc.iloc[i-1], ohlc.iloc[i]]
                
                if self._is_three_white_soldiers(candles):
                    results['three_white_soldiers'].append(i)
                
                if self._is_three_black_crows(candles):
                    results['three_black_crows'].append(i)
        
        return results
    
    def get_pattern_signals(self, ohlc: pd.DataFrame) -> Dict[str, Dict[str, Union[List[int], str]]]:
        """
        Get trading signals from candlestick patterns.
        
        Args:
            ohlc: DataFrame with OHLC price data
            
        Returns:
            Dictionary of pattern signals with sentiment and locations
        """
        patterns = self.identify_patterns(ohlc)
        
        signals = {}
        for pattern_name, indices in patterns.items():
            if not indices:  # Skip patterns not found
                continue
            
            # Determine sentiment based on pattern type
            if pattern_name in self.bullish_patterns:
                sentiment = "bullish"
            elif pattern_name in self.bearish_patterns:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            signals[pattern_name] = {
                "sentiment": sentiment,
                "indices": indices,
                "count": len(indices),
                "last_index": indices[-1] if indices else None
            }
        
        return signals
    
    def get_candlestick_summary(self, ohlc: pd.DataFrame) -> Dict[str, any]:
        """
        Generate a summary of candlestick patterns and their signals.
        
        Args:
            ohlc: DataFrame with OHLC price data
            
        Returns:
            Dictionary with pattern summary and overall sentiment
        """
        signals = self.get_pattern_signals(ohlc)
        
        if not signals:
            return {
                "patterns_found": False,
                "summary": "No significant candlestick patterns detected.",
                "overall_sentiment": "neutral",
                "signals": {}
            }
        
        # Count patterns by sentiment
        bullish_count = sum(1 for p in signals.values() if p["sentiment"] == "bullish")
        bearish_count = sum(1 for p in signals.values() if p["sentiment"] == "bearish")
        
        # Overall sentiment is weighted toward more recent patterns
        recent_patterns = {name: info for name, info in signals.items() 
                          if info["last_index"] >= len(ohlc) - 3}
        
        recent_bullish = sum(1 for p in recent_patterns.values() if p["sentiment"] == "bullish")
        recent_bearish = sum(1 for p in recent_patterns.values() if p["sentiment"] == "bearish")
        
        # Determine overall sentiment
        if recent_bullish > recent_bearish:
            overall_sentiment = "bullish"
        elif recent_bearish > recent_bullish:
            overall_sentiment = "bearish"
        else:  # Fall back to all patterns if recent ones are inconclusive
            if bullish_count > bearish_count:
                overall_sentiment = "bullish"
            elif bearish_count > bullish_count:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
        
        # Generate readable summary
        pattern_list = ", ".join([f"{name} ({info['sentiment']})" 
                                 for name, info in signals.items()])
        
        summary = f"Found {len(signals)} candlestick patterns: {pattern_list}. "
        summary += f"Overall sentiment is {overall_sentiment}."
        
        return {
            "patterns_found": True,
            "summary": summary,
            "overall_sentiment": overall_sentiment,
            "signals": signals
        }
    
    def get_trade_signals(self, ohlc: pd.DataFrame) -> Dict[str, any]:
        """
        Get actionable trade signals from candlestick patterns.
        
        Args:
            ohlc: DataFrame with OHLC price data
            
        Returns:
            Dictionary with trade signals and confidence
        """
        summary = self.get_candlestick_summary(ohlc)
        
        if not summary["patterns_found"]:
            return {
                "action": "hold",
                "confidence": 0.0,
                "explanation": "No significant candlestick patterns detected."
            }
        
        # Calculate confidence based on pattern types and counts
        signals = summary["signals"]
        overall_sentiment = summary["overall_sentiment"]
        
        # Higher confidence for:
        # 1. Multiple patterns with the same sentiment
        # 2. Patterns that appear at the end of the data
        # 3. Stronger patterns (engulfing, stars, etc.)
        
        # Define pattern strength (1-10)
        pattern_strength = {
            "doji": 3,
            "hammer": 5,
            "inverted_hammer": 5,
            "bullish_engulfing": 7,
            "bearish_engulfing": 7,
            "morning_star": 8,
            "evening_star": 8,
            "shooting_star": 6,
            "hanging_man": 6,
            "three_white_soldiers": 9,
            "three_black_crows": 9,
            "bullish_harami": 6,
            "bearish_harami": 6,
            "piercing_line": 7,
            "dark_cloud_cover": 7
        }
        
        # Calculate weighted confidence
        total_strength = 0
        sentiment_strength = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        for name, info in signals.items():
            # Get pattern strength
            strength = pattern_strength.get(name, 5)
            
            # Increase strength if at the end of data
            recency_boost = 1.5 if info["last_index"] >= len(ohlc) - 3 else 1.0
            
            # Add to total strength
            weighted_strength = strength * recency_boost
            total_strength += weighted_strength
            sentiment_strength[info["sentiment"]] += weighted_strength
        
        # Normalize to get confidence (0-1)
        max_possible = sum([max(pattern_strength.values()) * 1.5]) * len(signals)
        confidence_level = min(0.9, total_strength / max_possible) if max_possible > 0 else 0
        
        # Determine action based on sentiment and confidence
        if overall_sentiment == "bullish" and confidence_level > 0.2:
            action = "buy"
            confidence = confidence_level
            explanation = f"Bullish candlestick patterns detected with {confidence:.2f} confidence."
            
            # Add details about the strongest bullish pattern
            strongest_bullish = max(
                [name for name, info in signals.items() if info["sentiment"] == "bullish"],
                key=lambda x: pattern_strength.get(x, 0),
                default=None
            )
            if strongest_bullish:
                explanation += f" Most significant pattern: {strongest_bullish}."
            
        elif overall_sentiment == "bearish" and confidence_level > 0.2:
            action = "sell"
            confidence = confidence_level
            explanation = f"Bearish candlestick patterns detected with {confidence:.2f} confidence."
            
            # Add details about the strongest bearish pattern
            strongest_bearish = max(
                [name for name, info in signals.items() if info["sentiment"] == "bearish"],
                key=lambda x: pattern_strength.get(x, 0),
                default=None
            )
            if strongest_bearish:
                explanation += f" Most significant pattern: {strongest_bearish}."
            
        else:
            action = "hold"
            confidence = 0.2
            explanation = "No strong trading signal from candlestick patterns."
        
        return {
            "action": action,
            "confidence": confidence,
            "explanation": explanation,
            "patterns": [name for name in signals.keys()],
            "overall_sentiment": overall_sentiment
        }


def detect_candlestick_patterns(ohlc_data: pd.DataFrame) -> Dict[str, any]:
    """
    Wrapper function to detect candlestick patterns in OHLC data.
    
    Args:
        ohlc_data: DataFrame with columns 'open', 'high', 'low', 'close'
        
    Returns:
        Dictionary with pattern detection results
    """
    detector = CandlestickPatterns()
    signals = detector.get_trade_signals(ohlc_data)
    return signals


def get_pattern_data_for_visualization(ohlc_data: pd.DataFrame) -> Dict[str, List]:
    """
    Get pattern data in a format suitable for visualization on charts.
    
    Args:
        ohlc_data: DataFrame with columns 'open', 'high', 'low', 'close'
        
    Returns:
        Dictionary mapping pattern names to lists of data points for visualization
    """
    detector = CandlestickPatterns()
    patterns = detector.identify_patterns(ohlc_data)
    
    result = {}
    for pattern_name, indices in patterns.items():
        if not indices:
            continue
        
        # Create points for visualization
        points = []
        for idx in indices:
            # Get price at which to place the marker
            candle = ohlc_data.iloc[idx]
            
            # Place marker above or below candle based on pattern type
            if pattern_name in detector.bullish_patterns:
                price = candle['low'] * 0.995  # Slightly below
            elif pattern_name in detector.bearish_patterns:
                price = candle['high'] * 1.005  # Slightly above
            else:
                price = (candle['high'] + candle['low']) / 2  # Middle
            
            # Store index and price for plotting
            points.append({
                "index": idx,
                "price": price,
                "tooltip": f"{pattern_name.replace('_', ' ').title()}"
            })
        
        result[pattern_name] = points
    
    return result