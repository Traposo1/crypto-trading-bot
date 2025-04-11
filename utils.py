import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from models import Trade, MarketData, PerformanceMetrics
from app import db

# Configure logging
logger = logging.getLogger(__name__)

def format_timestamp(timestamp) -> str:
    """Format a timestamp for display"""
    if timestamp is None:
        return "N/A"
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    return str(timestamp)

def format_currency(value: float, currency: str = "USDT") -> str:
    """Format a currency value for display"""
    if value is None:
        return "N/A"
    
    return f"{value:.6f} {currency}"

def format_percentage(value: float) -> str:
    """Format a percentage value for display"""
    if value is None:
        return "N/A"
    
    return f"{value:.2f}%"

def calculate_performance_metrics(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calculate performance metrics for the bot
    
    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Build query for closed trades
        query = Trade.query.filter_by(status="closed")
        
        if start_date:
            query = query.filter(Trade.exit_time >= start_date)
        
        if end_date:
            query = query.filter(Trade.exit_time <= end_date)
        
        trades = query.all()
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit_loss is not None and t.profit_loss > 0)
        losing_trades = sum(1 for t in trades if t.profit_loss is not None and t.profit_loss <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.profit_loss for t in trades if t.profit_loss is not None)
        
        # Calculate average metrics
        avg_profit = sum(t.profit_loss for t in trades if t.profit_loss is not None and t.profit_loss > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.profit_loss for t in trades if t.profit_loss is not None and t.profit_loss <= 0) / losing_trades if losing_trades > 0 else 0
        avg_profit_pct = sum(t.profit_loss_pct for t in trades if t.profit_loss_pct is not None and t.profit_loss_pct > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss_pct = sum(t.profit_loss_pct for t in trades if t.profit_loss_pct is not None and t.profit_loss_pct <= 0) / losing_trades if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.profit_loss for t in trades if t.profit_loss is not None and t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in trades if t.profit_loss is not None and t.profit_loss < 0))
        # Use a large number (999) instead of infinity for JSON compatibility
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        # Calculate average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades if t.exit_time is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_profit_pct': avg_profit_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration
        }
    
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {
            'error': str(e),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0
        }

def update_daily_performance_metrics() -> None:
    """Update the daily performance metrics in the database"""
    try:
        # Get today's date
        today = datetime.utcnow().date()
        
        # Check if entry already exists for today
        existing = PerformanceMetrics.query.filter_by(date=today).first()
        
        if existing:
            # Update existing entry
            performance = existing
        else:
            # Create new entry
            performance = PerformanceMetrics(date=today)
        
        # Get trades closed today
        start_of_day = datetime.combine(today, datetime.min.time())
        end_of_day = datetime.combine(today, datetime.max.time())
        
        today_trades = Trade.query.filter(
            Trade.status == "closed",
            Trade.exit_time >= start_of_day,
            Trade.exit_time <= end_of_day
        ).all()
        
        # Update metrics
        performance.total_trades = len(today_trades)
        performance.winning_trades = sum(1 for t in today_trades if t.profit_loss is not None and t.profit_loss > 0)
        performance.losing_trades = sum(1 for t in today_trades if t.profit_loss is not None and t.profit_loss <= 0)
        
        # Calculate profit/loss
        total_profit = sum(t.profit_loss for t in today_trades if t.profit_loss is not None)
        performance.profit_loss = total_profit
        
        # Calculate profit/loss percentage (simplified approach)
        if performance.total_trades > 0:
            performance.profit_loss_pct = sum(t.profit_loss_pct for t in today_trades if t.profit_loss_pct is not None) / performance.total_trades
        
        # Save to database
        if not existing:
            db.session.add(performance)
        
        db.session.commit()
        
        logger.info(f"Updated daily performance metrics for {today}")
    
    except Exception as e:
        logger.error(f"Error updating daily performance metrics: {str(e)}")
        db.session.rollback()

def get_market_summary(symbol: str, timeframe: str = '1h', limit: int = 24) -> Dict[str, Any]:
    """
    Get a summary of market data for the given symbol
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for the data
        limit: Number of periods to include
        
    Returns:
        Dictionary with market summary data
    """
    try:
        # Query the latest market data
        data = MarketData.query.filter_by(trading_pair=symbol).order_by(MarketData.timestamp.desc()).limit(limit).all()
        
        if not data:
            logger.warning(f"No market data found for {symbol}")
            return {'error': f"No market data found for {symbol}"}
        
        # Reverse to get chronological order
        data = data[::-1]
        
        # Calculate summary metrics
        current_price = data[-1].close
        open_price = data[0].open
        high_price = max(d.high for d in data)
        low_price = min(d.low for d in data)
        
        price_change = current_price - open_price
        price_change_pct = (price_change / open_price) * 100 if open_price > 0 else 0
        
        # Calculate volatility (standard deviation of returns)
        prices = [d.close for d in data]
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns) * 100 if returns else 0
        
        # Calculate volume information
        total_volume = sum(d.volume for d in data)
        avg_volume = total_volume / len(data) if data else 0
        
        # Get latest indicator values
        latest = data[-1]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'timestamp': latest.timestamp,
            'indicators': {
                'rsi': latest.rsi,
                'bb_upper': latest.bb_upper,
                'bb_middle': latest.bb_middle,
                'bb_lower': latest.bb_lower,
                'macd': latest.macd,
                'macd_signal': latest.macd_signal,
                'macd_hist': latest.macd_hist,
                'stoch_k': latest.stoch_k,
                'stoch_d': latest.stoch_d
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        return {'error': str(e)}

def export_trade_history(format: str = 'json') -> str:
    """
    Export trade history to a file
    
    Args:
        format: Output format ('json' or 'csv')
        
    Returns:
        Formatted data string
    """
    try:
        # Query all trades
        trades = Trade.query.all()
        
        # Convert to list of dictionaries
        trade_list = []
        for trade in trades:
            trade_dict = {
                'id': trade.id,
                'trading_pair': trade.trading_pair,
                'order_id': trade.order_id,
                'trade_type': trade.trade_type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'amount': trade.amount,
                'fee': trade.fee,
                'profit_loss': trade.profit_loss,
                'profit_loss_pct': trade.profit_loss_pct,
                'status': trade.status,
                'entry_time': format_timestamp(trade.entry_time),
                'exit_time': format_timestamp(trade.exit_time),
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'reason': trade.reason,
                'paper_trade': trade.paper_trade
            }
            trade_list.append(trade_dict)
        
        if format == 'json':
            return json.dumps(trade_list, indent=2)
        elif format == 'csv':
            # Build CSV string
            if not trade_list:
                return "No trades found"
            
            headers = list(trade_list[0].keys())
            csv_data = ",".join(headers) + "\n"
            
            for trade in trade_list:
                row = [str(trade.get(h, "")) for h in headers]
                csv_data += ",".join(row) + "\n"
            
            return csv_data
        else:
            return f"Unsupported format: {format}"
    
    except Exception as e:
        logger.error(f"Error exporting trade history: {str(e)}")
        return f"Error: {str(e)}"
