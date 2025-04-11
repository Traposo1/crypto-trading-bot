"""
Test script for candlestick pattern detection
"""
import pandas as pd
import numpy as np
from bot.candlestick_patterns import detect_candlestick_patterns

def test_pattern_detection():
    """Test pattern detection with sample data"""
    print("Testing candlestick pattern detection...")
    
    # Create sample OHLC data with known patterns
    # This data intentionally creates a few common patterns
    data = {
        'open':  [100, 102, 105, 103, 106, 107, 108, 110, 112, 111],
        'high':  [105, 106, 107, 106, 109, 110, 112, 115, 116, 113],
        'low':   [98,  100, 103, 100, 104, 105, 106, 108, 108, 105],
        'close': [102, 105, 103, 106, 107, 108, 110, 112, 111, 106]
    }
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Detect patterns
    pattern_results = detect_candlestick_patterns(df)
    
    # Print results
    print("\nDetected patterns:")
    print(f"Total patterns found: {len(pattern_results.get('patterns', []))}")
    for pattern in pattern_results.get('patterns', []):
        print(f"- {pattern}")
    
    print(f"\nAction: {pattern_results.get('action', 'None')}")
    print(f"Confidence: {pattern_results.get('confidence', 0)}")
    print(f"Overall sentiment: {pattern_results.get('overall_sentiment', 'neutral')}")
    
    return pattern_results

if __name__ == "__main__":
    # Set up pandas display options
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    
    # Run the test
    results = test_pattern_detection()
    
    print("\nTest completed successfully!")