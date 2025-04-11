import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from models import Trade
from app import db
from bot.exchange import execute_order, close_trade, get_account_balance

# Configure logging
logger = logging.getLogger(__name__)

def execute_trades(signal: Dict[str, Any], df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict]:
    """
    Execute trades based on trading signals and risk management rules
    
    Args:
        signal: Trading signal dictionary
        df: DataFrame with OHLCV data and indicators
        config: Configuration dictionary
        
    Returns:
        Dictionary with trade information if a trade was executed, None otherwise
    """
    try:
        # Get current price
        current_price = df.iloc[-1]['close']
        
        # Check if we have a valid buy or sell signal
        if not signal['buy'] and not signal['sell']:
            return None
        
        # Get trading pair - prefer from signal if available, otherwise fall back to config
        trading_pair = signal.get('trading_pair', config['trading_pair'])
        
        # Calculate position size based on capital and risk
        balance = get_account_balance()
        available_capital = 0
        
        # Determine available capital from balance
        quote_currency = trading_pair.split('/')[1]  # e.g., USDT
        if quote_currency in balance:
            available_capital = balance[quote_currency].get('free', 0)
        
        # If we don't have balance data (e.g., in paper trading mode), use the configured capital
        if available_capital <= 0:
            available_capital = config['capital_per_trade']
        
        # Calculate position size (amount to buy/sell)
        position_size = calculate_position_size(available_capital, current_price, config)
        
        if position_size <= 0:
            logger.warning("Position size calculation returned zero or negative value")
            return None
        
        # Execute buy order
        if signal['buy']:
            logger.info(f"Executing BUY order for {position_size} {trading_pair} at ~{current_price}")
            order = execute_order(trading_pair, 'market', 'buy', position_size)
            
            if order:
                logger.info(f"BUY order executed successfully: {order}")
                return {
                    'action': 'buy',
                    'trading_pair': trading_pair,
                    'amount': position_size,
                    'price': current_price,
                    'order_id': order.get('id', ''),
                    'timestamp': datetime.utcnow()
                }
            else:
                logger.error(f"Failed to execute BUY order for {trading_pair}")
        
        # Execute sell order (find matching open trade)
        elif signal['sell']:
            open_trades = get_open_trades(trading_pair)
            
            if not open_trades:
                logger.warning(f"SELL signal generated but no open trades found for {trading_pair}")
                return None
            
            # Close the oldest open trade
            trade = open_trades[0]
            logger.info(f"Executing SELL order for trade #{trade.id} at ~{current_price}")
            
            result = close_trade(trade.id, current_price, "Sell signal from strategy")
            
            if result:
                logger.info(f"SELL order executed successfully: {result}")
                return {
                    'action': 'sell',
                    'trading_pair': trading_pair,
                    'amount': trade.amount,
                    'price': current_price,
                    'profit_loss': result.get('profit_loss', 0),
                    'profit_loss_pct': result.get('profit_loss_pct', 0),
                    'timestamp': datetime.utcnow()
                }
            else:
                logger.error(f"Failed to execute SELL order for trade #{trade.id}")
        
        return None
    
    except Exception as e:
        logger.error(f"Error executing trades: {str(e)}")
        return None

def calculate_position_size(available_capital: float, current_price: float, config: Dict[str, Any]) -> float:
    """
    Calculate position size based on available capital and risk parameters
    
    Args:
        available_capital: Available capital for trading
        current_price: Current price of the asset
        config: Configuration dictionary
        
    Returns:
        Position size (amount to buy/sell)
    """
    try:
        # Calculate base position size
        position_size = min(config['capital_per_trade'], available_capital) / current_price
        
        # Round down to appropriate precision
        precision = 6  # Default precision for most cryptocurrencies
        
        # Adjust precision based on price (higher priced assets typically have lower precision)
        if current_price >= 1000:
            precision = 4
        elif current_price >= 100:
            precision = 5
        
        # Round down to ensure we don't exceed available capital
        position_size = int(position_size * 10**precision) / 10**precision
        
        return position_size
    
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0

def get_open_trades(symbol: Optional[str] = None) -> List[Trade]:
    """
    Get list of open trades
    
    Args:
        symbol: Optional trading pair symbol to filter by
        
    Returns:
        List of open Trade objects
    """
    try:
        query = Trade.query.filter_by(status='open')
        
        if symbol:
            query = query.filter_by(trading_pair=symbol)
        
        # Order by entry time (oldest first)
        open_trades = query.order_by(Trade.entry_time).all()
        
        return open_trades
    
    except Exception as e:
        logger.error(f"Error getting open trades: {str(e)}")
        return []

def check_stop_loss_take_profit(df: pd.DataFrame, config: Dict[str, Any], trading_pair: Optional[str] = None) -> None:
    """
    Check open trades for stop loss or take profit conditions
    
    Args:
        df: DataFrame with latest OHLCV data
        config: Configuration dictionary
        trading_pair: Optional trading pair to check (if None, uses config's trading_pair)
    """
    try:
        # Get the latest price
        current_price = df.iloc[-1]['close']
        
        # Get the trading pair - either from parameter or config
        symbol = trading_pair if trading_pair else config['trading_pair']
        
        # Get open trades for this pair
        open_trades = get_open_trades(symbol)
        
        for trade in open_trades:
            # Skip if not a buy trade
            if trade.trade_type != 'buy':
                continue
            
            # Check stop loss
            if trade.stop_loss and current_price <= trade.stop_loss:
                logger.info(f"Stop loss triggered for trade #{trade.id} ({symbol}) at {current_price}")
                close_trade(trade.id, current_price, "Stop loss triggered")
                continue
            
            # Check take profit
            if trade.take_profit and current_price >= trade.take_profit:
                logger.info(f"Take profit triggered for trade #{trade.id} ({symbol}) at {current_price}")
                close_trade(trade.id, current_price, "Take profit triggered")
                continue
    
    except Exception as e:
        logger.error(f"Error checking stop loss/take profit for {trading_pair}: {str(e)}")

def adjust_stop_loss(trade_id: int, new_stop_loss: float) -> bool:
    """
    Adjust stop loss for an existing trade
    
    Args:
        trade_id: ID of the trade
        new_stop_loss: New stop loss price
        
    Returns:
        True if successful, False otherwise
    """
    try:
        trade = Trade.query.get(trade_id)
        
        if not trade:
            logger.error(f"Trade with ID {trade_id} not found")
            return False
        
        if trade.status != 'open':
            logger.warning(f"Cannot adjust stop loss for non-open trade #{trade_id}")
            return False
        
        # Update stop loss
        trade.stop_loss = new_stop_loss
        db.session.commit()
        
        logger.info(f"Stop loss for trade #{trade_id} adjusted to {new_stop_loss}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error adjusting stop loss: {str(e)}")
        db.session.rollback()
        return False

def calculate_risk_metrics(trades: List[Trade]) -> Dict[str, float]:
    """
    Calculate risk metrics based on trade history
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Dictionary with risk metrics
    """
    if not trades:
        return {
            'win_rate': 0,
            'loss_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    try:
        # Filter closed trades with profit/loss data
        closed_trades = [t for t in trades if t.status == 'closed' and t.profit_loss is not None]
        
        if not closed_trades:
            return {
                'win_rate': 0,
                'loss_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Calculate win/loss metrics
        winning_trades = [t for t in closed_trades if t.profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_count = len(closed_trades)
        
        win_rate = win_count / total_count if total_count > 0 else 0
        loss_rate = loss_count / total_count if total_count > 0 else 0
        
        # Calculate profit/loss metrics
        avg_profit = sum(t.profit_loss for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        
        # Use a large number (999) instead of infinity for JSON compatibility
        profit_factor = total_profit / total_loss if total_loss > 0 else 999.0
        
        # Calculate drawdown
        equity_curve = []
        equity = 0
        peak = 0
        drawdowns = []
        
        # Sort trades by exit time
        sorted_trades = sorted(closed_trades, key=lambda t: t.exit_time or datetime.min)
        
        for trade in sorted_trades:
            equity += trade.profit_loss or 0
            equity_curve.append(equity)
            
            peak = max(peak, equity)
            drawdown = peak - equity
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.profit_loss_pct for t in closed_trades if t.profit_loss_pct is not None]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 1
        
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        return {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        return {
            'win_rate': 0,
            'loss_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'error': str(e)
        }
