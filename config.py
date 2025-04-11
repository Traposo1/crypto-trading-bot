import os
from typing import Dict, Any
from models import BotConfig
from app import db

# Default configuration
DEFAULT_CONFIG = {
    # Legacy API credentials (for backward compatibility)
    "api_key": "",
    "api_secret": "",
    
    # KuCoin API credentials (primary exchange)
    "kucoin_api_key": "",
    "kucoin_api_secret": "",
    "kucoin_passphrase": "",
    
    "trading_pair": "BTC/USDT",
    "timeframe": "5m",
    "capital_per_trade": 100.0,
    "max_open_trades": 3,
    "paper_trading": True,
    
    # Technical indicators
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    
    "bb_period": 20,
    "bb_std_dev": 2.0,
    
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_overbought": 80.0,
    "stoch_oversold": 20.0,
    
    # Risk management
    "stop_loss_pct": 2.0,
    "take_profit_pct": 3.0,
    
    # ML model settings
    "ml_enabled": True,
    "ml_confidence_threshold": 0.7,
    "ml_lookback_periods": 100,
}

def get_config() -> Dict[str, Any]:
    """
    Get the bot configuration, reading from the database.
    Falls back to default config if database is not available.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get API credentials from environment if available
    ENV_CONFIG = DEFAULT_CONFIG.copy()
    
    # KuCoin API credentials from environment (priority)
    if os.environ.get("KUCOIN_API_KEY"):
        ENV_CONFIG["kucoin_api_key"] = os.environ.get("KUCOIN_API_KEY")
    if os.environ.get("KUCOIN_API_SECRET"):
        ENV_CONFIG["kucoin_api_secret"] = os.environ.get("KUCOIN_API_SECRET")
    if os.environ.get("KUCOIN_PASSPHRASE"):
        ENV_CONFIG["kucoin_passphrase"] = os.environ.get("KUCOIN_PASSPHRASE")
    
    # Legacy Binance API credentials from environment (for backward compatibility)
    if os.environ.get("BINANCE_API_KEY"):
        ENV_CONFIG["api_key"] = os.environ.get("BINANCE_API_KEY")
    if os.environ.get("BINANCE_API_SECRET"):
        ENV_CONFIG["api_secret"] = os.environ.get("BINANCE_API_SECRET")
    
    try:
        # Try to get config from database
        config_entry = BotConfig.query.first()
        
        if config_entry:
            logger.info("Retrieved configuration from database")
            # Convert database model to dictionary
            db_config = {}
            
            # Get all attributes that are in our default config
            for key in DEFAULT_CONFIG.keys():
                if hasattr(config_entry, key):
                    value = getattr(config_entry, key)
                    # Handle special cases for comma-separated string fields
                    if key == 'active_pairs' and value and isinstance(value, str):
                        db_config[key] = value
                    else:
                        db_config[key] = value
            
            # Merge with default and env config, prioritizing db values
            merged_config = {**DEFAULT_CONFIG, **ENV_CONFIG, **db_config}
            return merged_config
        else:
            # No config in database, use default + environment values
            logger.warning("No configuration found in database, using default")
            # Add secrets from environment variables
            config = {**DEFAULT_CONFIG, **ENV_CONFIG}
            
            # Create a new config entry
            try:
                new_config = BotConfig()
                # Set attributes from default config
                for key, value in config.items():
                    if hasattr(new_config, key):
                        setattr(new_config, key, value)
                
                # Save to database
                db.session.add(new_config)
                db.session.commit()
                logger.info("Created new configuration in database")
            except Exception as db_error:
                logger.error(f"Failed to create new config: {str(db_error)}")
                db.session.rollback()
            
            return config
    except Exception as e:
        logger.error(f"Error in get_config: {str(e)}")
        # Return default + environment values as fallback
        return {**DEFAULT_CONFIG, **ENV_CONFIG}

def update_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the bot configuration in the database.
    Returns the updated configuration.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Try to get existing config from database
        config_entry = BotConfig.query.first()
        
        if not config_entry:
            # Create new config entry if it doesn't exist
            logger.info("Creating new config entry in database")
            config_entry = BotConfig()
            db.session.add(config_entry)
        
        # Update the config entry with new values
        for key, value in new_config.items():
            if hasattr(config_entry, key):
                setattr(config_entry, key, value)
        
        # Special handling for active_pairs (ensure it exists in the model)
        if 'active_pairs' in new_config and hasattr(config_entry, 'active_pairs'):
            config_entry.active_pairs = new_config['active_pairs']
        
        # Convert any list type data to comma-separated strings
        if 'active_pairs' in new_config and isinstance(new_config['active_pairs'], list):
            config_entry.active_pairs = ','.join(new_config['active_pairs'])
            
        # Commit changes to database
        db.session.commit()
        logger.info("Successfully updated configuration in database")
        
        # Return the full updated config
        return {**get_config(), **new_config}  # Merge updated values with full config
        
    except Exception as e:
        logger.error(f"Error in update_config: {str(e)}")
        db.session.rollback()  # Rollback on error
        # Return the original new_config as fallback + current config
        current_config = get_config()
        return {**current_config, **new_config}
