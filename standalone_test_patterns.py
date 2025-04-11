"""
Standalone test script for candlestick pattern detection
"""
import pandas as pd
import numpy as np

# Simple helper functions
def is_bullish_candle(candle):
    return candle['close'] > candle['open']

def is_bearish_candle(candle):
    return candle['close'] < candle['open']

def get_candle_body_size(candle):
    return abs(candle['close'] - candle['open'])

def get_upper_shadow(candle):
    if is_bullish_candle(candle):
        return candle['high'] - candle['close']
    return candle['high'] - candle['open']

def get_lower_shadow(candle):
    if is_bullish_candle(candle):
        return candle['open'] - candle['low']
    return candle['close'] - candle['low']

# Test pattern detection with some simple rules
def test_simple_pattern_detection(df):
    """Test pattern detection with simple rules on the provided data"""
    patterns = []
    sentiment = "neutral"
    
    # Check for bullish three white soldiers pattern
    if len(df) >= 3:
        last_three = df.iloc[-3:].reset_index(drop=True)
        if (all(is_bullish_candle(last_three.iloc[i]) for i in range(3)) and 
            last_three.iloc[0]['close'] < last_three.iloc[1]['close'] < last_three.iloc[2]['close'] and
            last_three.iloc[0]['open'] < last_three.iloc[1]['open'] < last_three.iloc[2]['open']):
            patterns.append("Three White Soldiers")
            sentiment = "bullish"
    
    # Check for bearish three black crows pattern
    if len(df) >= 3:
        last_three = df.iloc[-3:].reset_index(drop=True)
        if (all(is_bearish_candle(last_three.iloc[i]) for i in range(3)) and 
            last_three.iloc[0]['close'] > last_three.iloc[1]['close'] > last_three.iloc[2]['close'] and
            last_three.iloc[0]['open'] > last_three.iloc[1]['open'] > last_three.iloc[2]['open']):
            patterns.append("Three Black Crows")
            sentiment = "bearish"
    
    # Check for bullish engulfing pattern
    if len(df) >= 2:
        prev, current = df.iloc[-2], df.iloc[-1]
        if (is_bearish_candle(prev) and 
            is_bullish_candle(current) and
            current['open'] < prev['close'] and
            current['close'] > prev['open']):
            patterns.append("Bullish Engulfing")
            sentiment = "bullish"
    
    # Check for bearish engulfing pattern
    if len(df) >= 2:
        prev, current = df.iloc[-2], df.iloc[-1]
        if (is_bullish_candle(prev) and 
            is_bearish_candle(current) and
            current['open'] > prev['close'] and
            current['close'] < prev['open']):
            patterns.append("Bearish Engulfing")
            sentiment = "bearish"
    
    # Check for hammer pattern (bullish)
    if len(df) >= 1:
        current = df.iloc[-1]
        body_size = get_candle_body_size(current)
        lower_shadow = get_lower_shadow(current)
        upper_shadow = get_upper_shadow(current)
        
        if (is_bullish_candle(current) and
            lower_shadow > 2 * body_size and
            upper_shadow < 0.2 * body_size):
            patterns.append("Hammer")
            sentiment = "bullish"
    
    # Check for shooting star pattern (bearish)
    if len(df) >= 1:
        current = df.iloc[-1]
        body_size = get_candle_body_size(current)
        lower_shadow = get_lower_shadow(current)
        upper_shadow = get_upper_shadow(current)
        
        if (is_bearish_candle(current) and
            upper_shadow > 2 * body_size and
            lower_shadow < 0.2 * body_size):
            patterns.append("Shooting Star")
            sentiment = "bearish"
    
    # Determine action based on sentiment
    if sentiment == "bullish":
        action = "buy"
        confidence = 0.7 if len(patterns) > 1 else 0.5
    elif sentiment == "bearish":
        action = "sell"
        confidence = 0.7 if len(patterns) > 1 else 0.5
    else:
        action = "hold"
        confidence = 0.0
    
    return {
        "patterns": patterns,
        "action": action,
        "confidence": confidence,
        "overall_sentiment": sentiment
    }

def main():
    """Main function to run the test"""
    print("Testing candlestick pattern detection...")
    
    # Create sample OHLC data with known patterns
    # This data is designed to show a few common patterns
    data = {
        'open':  [100, 102, 105, 103, 106, 107, 108, 110, 112, 111],
        'high':  [105, 106, 107, 106, 109, 110, 112, 115, 116, 113],
        'low':   [98,  100, 103, 100, 104, 105, 106, 108, 108, 105],
        'close': [102, 105, 103, 106, 107, 108, 110, 112, 111, 106]
    }
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Print the sample data
    print("\nSample OHLC data:")
    print(df)
    
    # Detect patterns
    pattern_results = test_simple_pattern_detection(df)
    
    # Print results
    print("\nDetected patterns:")
    print(f"Total patterns found: {len(pattern_results.get('patterns', []))}")
    for pattern in pattern_results.get('patterns', []):
        print(f"- {pattern}")
    
    print(f"\nAction: {pattern_results.get('action', 'None')}")
    print(f"Confidence: {pattern_results.get('confidence', 0)}")
    print(f"Overall sentiment: {pattern_results.get('overall_sentiment', 'neutral')}")
    
    # Create a different dataset with clear three white soldiers pattern
    bullish_data = {
        'open':  [100, 102, 104, 106, 108, 110],
        'high':  [105, 107, 109, 111, 114, 115],
        'low':   [98,  100, 102, 105, 107, 109],
        'close': [102, 104, 106, 108, 110, 112]
    }
    
    bullish_df = pd.DataFrame(bullish_data)
    print("\n\nBullish pattern sample data:")
    print(bullish_df)
    
    bullish_results = test_simple_pattern_detection(bullish_df)
    
    print("\nDetected patterns:")
    print(f"Total patterns found: {len(bullish_results.get('patterns', []))}")
    for pattern in bullish_results.get('patterns', []):
        print(f"- {pattern}")
    
    print(f"\nAction: {bullish_results.get('action', 'None')}")
    print(f"Confidence: {bullish_results.get('confidence', 0)}")
    print(f"Overall sentiment: {bullish_results.get('overall_sentiment', 'neutral')}")
    
    # Create a different dataset with clear bearish pattern
    bearish_data = {
        'open':  [112, 110, 108, 106, 104, 102],
        'high':  [115, 113, 110, 108, 106, 104],
        'low':   [109, 107, 105, 103, 101, 98],
        'close': [110, 108, 106, 104, 102, 100]
    }
    
    bearish_df = pd.DataFrame(bearish_data)
    print("\n\nBearish pattern sample data:")
    print(bearish_df)
    
    bearish_results = test_simple_pattern_detection(bearish_df)
    
    print("\nDetected patterns:")
    print(f"Total patterns found: {len(bearish_results.get('patterns', []))}")
    for pattern in bearish_results.get('patterns', []):
        print(f"- {pattern}")
    
    print(f"\nAction: {bearish_results.get('action', 'None')}")
    print(f"Confidence: {bearish_results.get('confidence', 0)}")
    print(f"Overall sentiment: {bearish_results.get('overall_sentiment', 'neutral')}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()