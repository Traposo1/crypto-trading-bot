from datetime import datetime
from app import db

class BotConfig(db.Model):
    """Configuration settings for the trading bot"""
    id = db.Column(db.Integer, primary_key=True)
    # Legacy API credentials (for backward compatibility)
    api_key = db.Column(db.String(256))
    api_secret = db.Column(db.String(256))
    
    # KuCoin API credentials
    kucoin_api_key = db.Column(db.String(256))
    kucoin_api_secret = db.Column(db.String(256))
    kucoin_passphrase = db.Column(db.String(256))
    
    trading_pair = db.Column(db.String(20), default="BTC/USDT")  # Primary trading pair for backward compatibility
    timeframe = db.Column(db.String(10), default="5m")
    capital_per_trade = db.Column(db.Float, default=100.0)
    max_open_trades = db.Column(db.Integer, default=3)
    paper_trading = db.Column(db.Boolean, default=True)
    active_pairs = db.Column(db.Text, default="BTC/USDT")  # Comma-separated list of active trading pairs
    
    # Technical indicators config
    rsi_period = db.Column(db.Integer, default=14)
    rsi_overbought = db.Column(db.Float, default=70.0)
    rsi_oversold = db.Column(db.Float, default=30.0)
    
    bb_period = db.Column(db.Integer, default=20)
    bb_std_dev = db.Column(db.Float, default=2.0)
    
    macd_fast = db.Column(db.Integer, default=12)
    macd_slow = db.Column(db.Integer, default=26)
    macd_signal = db.Column(db.Integer, default=9)
    
    stoch_k_period = db.Column(db.Integer, default=14)
    stoch_d_period = db.Column(db.Integer, default=3)
    stoch_overbought = db.Column(db.Float, default=80.0)
    stoch_oversold = db.Column(db.Float, default=20.0)
    
    # Risk management
    stop_loss_pct = db.Column(db.Float, default=2.0)
    take_profit_pct = db.Column(db.Float, default=3.0)
    
    # ML model settings
    ml_enabled = db.Column(db.Boolean, default=True)
    ml_confidence_threshold = db.Column(db.Float, default=0.7)
    ml_lookback_periods = db.Column(db.Integer, default=100)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Trade(db.Model):
    """Record of executed trades"""
    id = db.Column(db.Integer, primary_key=True)
    trading_pair = db.Column(db.String(20))
    order_id = db.Column(db.String(100))
    trade_type = db.Column(db.String(10))  # 'buy' or 'sell'
    entry_price = db.Column(db.Float)
    exit_price = db.Column(db.Float, nullable=True)
    amount = db.Column(db.Float)
    fee = db.Column(db.Float, default=0.0)
    profit_loss = db.Column(db.Float, nullable=True)
    profit_loss_pct = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='open')  # 'open', 'closed', 'cancelled'
    
    # Technical indicators values at time of trade
    rsi_value = db.Column(db.Float, nullable=True)
    bb_upper = db.Column(db.Float, nullable=True)
    bb_middle = db.Column(db.Float, nullable=True)
    bb_lower = db.Column(db.Float, nullable=True)
    macd = db.Column(db.Float, nullable=True)
    macd_signal = db.Column(db.Float, nullable=True)
    macd_hist = db.Column(db.Float, nullable=True)
    stoch_k = db.Column(db.Float, nullable=True)
    stoch_d = db.Column(db.Float, nullable=True)
    
    ml_prediction = db.Column(db.Float, nullable=True)
    ml_confidence = db.Column(db.Float, nullable=True)
    
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime, nullable=True)
    
    stop_loss = db.Column(db.Float, nullable=True)
    take_profit = db.Column(db.Float, nullable=True)
    
    reason = db.Column(db.String(200))
    notes = db.Column(db.Text, nullable=True)
    paper_trade = db.Column(db.Boolean, default=True)

class MarketData(db.Model):
    """Historical market data for analysis"""
    id = db.Column(db.Integer, primary_key=True)
    trading_pair = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, index=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)
    
    # Technical indicators
    rsi = db.Column(db.Float, nullable=True)
    bb_upper = db.Column(db.Float, nullable=True)
    bb_middle = db.Column(db.Float, nullable=True)
    bb_lower = db.Column(db.Float, nullable=True)
    macd = db.Column(db.Float, nullable=True)
    macd_signal = db.Column(db.Float, nullable=True)
    macd_hist = db.Column(db.Float, nullable=True)
    stoch_k = db.Column(db.Float, nullable=True)
    stoch_d = db.Column(db.Float, nullable=True)
    
    # Moving Averages
    ema_9 = db.Column(db.Float, nullable=True)
    ema_21 = db.Column(db.Float, nullable=True)
    ema_50 = db.Column(db.Float, nullable=True)
    ema_200 = db.Column(db.Float, nullable=True)
    sma_50 = db.Column(db.Float, nullable=True)
    sma_200 = db.Column(db.Float, nullable=True)
    
    # Volume-based indicators
    vwap_14 = db.Column(db.Float, nullable=True)
    
    # Trend indicators
    adx = db.Column(db.Float, nullable=True)
    di_plus = db.Column(db.Float, nullable=True)
    di_minus = db.Column(db.Float, nullable=True)
    
    # Ichimoku Cloud indicators
    ichimoku_tenkan = db.Column(db.Float, nullable=True)
    ichimoku_kijun = db.Column(db.Float, nullable=True)
    ichimoku_senkou_a = db.Column(db.Float, nullable=True)
    ichimoku_senkou_b = db.Column(db.Float, nullable=True) 
    ichimoku_chikou = db.Column(db.Float, nullable=True)
    
    # Composite index for efficient querying
    __table_args__ = (
        db.Index('idx_market_data_pair_time', trading_pair, timestamp),
    )

class BacktestResult(db.Model):
    """Results from backtesting strategies"""
    id = db.Column(db.Integer, primary_key=True)
    trading_pair = db.Column(db.String(20))
    timeframe = db.Column(db.String(10))
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    initial_capital = db.Column(db.Float)
    final_capital = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    winning_trades = db.Column(db.Integer)
    losing_trades = db.Column(db.Integer)
    profit_loss = db.Column(db.Float)
    profit_loss_pct = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    max_drawdown_pct = db.Column(db.Float)
    config_snapshot = db.Column(db.Text)  # JSON string of config used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PerformanceMetrics(db.Model):
    """System performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, index=True)
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)
    profit_loss = db.Column(db.Float, default=0.0)
    profit_loss_pct = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
