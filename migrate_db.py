from app import app, db
from models import BotConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_database():
    """Apply database migrations"""
    with app.app_context():
        try:
            # Migration 1: Add the active_pairs column to the bot_config table
            migrate_active_pairs_column()
            
            # Migration 2: Add new indicator columns to the market_data table
            migrate_market_data_indicators()
            
            # Migration 3: Add Ichimoku Cloud indicator columns to the market_data table
            migrate_ichimoku_columns()
            
            # Migration 4: Add KuCoin API credential columns to the bot_config table
            migrate_kucoin_api_fields()
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            db.session.rollback()
            raise

def migrate_active_pairs_column():
    """Add the active_pairs column to the bot_config table"""
    try:
        # Check if column exists
        result = db.session.execute(db.text("SELECT column_name FROM information_schema.columns WHERE table_name='bot_config' AND column_name='active_pairs'"))
        if not result.fetchone():
            logger.info("Adding active_pairs column to bot_config table")
            # Add the active_pairs column
            db.session.execute(db.text("ALTER TABLE bot_config ADD COLUMN active_pairs TEXT DEFAULT 'BTC/USDT'"))
            db.session.commit()
            logger.info("Active pairs column migration completed successfully")
        else:
            logger.info("Column active_pairs already exists")

        # Update existing rows
        configs = BotConfig.query.all()
        for config in configs:
            if not config.active_pairs:
                config.active_pairs = config.trading_pair
        db.session.commit()
        logger.info("Updated existing configs with trading pairs")
    except Exception as e:
        logger.error(f"Active pairs migration failed: {str(e)}")
        db.session.rollback()
        raise

def migrate_market_data_indicators():
    """Add new indicator columns to market_data table"""
    try:
        # Define the new columns to add
        new_columns = [
            # Moving Averages
            {"name": "ema_9", "type": "FLOAT"},
            {"name": "ema_21", "type": "FLOAT"},
            {"name": "ema_50", "type": "FLOAT"},
            {"name": "ema_200", "type": "FLOAT"},
            {"name": "sma_50", "type": "FLOAT"},
            {"name": "sma_200", "type": "FLOAT"},
            
            # Volume-based indicators
            {"name": "vwap_14", "type": "FLOAT"},
            
            # Trend indicators
            {"name": "adx", "type": "FLOAT"},
            {"name": "di_plus", "type": "FLOAT"},
            {"name": "di_minus", "type": "FLOAT"}
        ]
        
        # Check if columns exist and add them if they don't
        for column in new_columns:
            result = db.session.execute(db.text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='market_data' AND column_name='{column['name']}'"
            ))
            
            if not result.fetchone():
                logger.info(f"Adding {column['name']} column to market_data table")
                db.session.execute(db.text(
                    f"ALTER TABLE market_data ADD COLUMN {column['name']} {column['type']}"
                ))
                
        db.session.commit()
        logger.info("Market data indicators migration completed successfully")
    except Exception as e:
        logger.error(f"Market data indicators migration failed: {str(e)}")
        db.session.rollback()
        raise

def migrate_ichimoku_columns():
    """Add Ichimoku Cloud indicator columns to market_data table"""
    try:
        # Define the new columns to add
        new_columns = [
            # Ichimoku Cloud indicators
            {"name": "ichimoku_tenkan", "type": "FLOAT"},
            {"name": "ichimoku_kijun", "type": "FLOAT"},
            {"name": "ichimoku_senkou_a", "type": "FLOAT"},
            {"name": "ichimoku_senkou_b", "type": "FLOAT"},
            {"name": "ichimoku_chikou", "type": "FLOAT"}
        ]
        
        # Check if columns exist and add them if they don't
        for column in new_columns:
            result = db.session.execute(db.text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='market_data' AND column_name='{column['name']}'"
            ))
            
            if not result.fetchone():
                logger.info(f"Adding {column['name']} column to market_data table")
                db.session.execute(db.text(
                    f"ALTER TABLE market_data ADD COLUMN {column['name']} {column['type']}"
                ))
                
        db.session.commit()
        logger.info("Ichimoku Cloud indicators migration completed successfully")
    except Exception as e:
        logger.error(f"Ichimoku Cloud indicators migration failed: {str(e)}")
        db.session.rollback()
        raise

def migrate_kucoin_api_fields():
    """Add KuCoin API credential columns to the bot_config table"""
    try:
        # Define the new columns to add
        new_columns = [
            {"name": "kucoin_api_key", "type": "VARCHAR(256)"},
            {"name": "kucoin_api_secret", "type": "VARCHAR(256)"},
            {"name": "kucoin_passphrase", "type": "VARCHAR(256)"}
        ]
        
        # Check if columns exist and add them if they don't
        for column in new_columns:
            result = db.session.execute(db.text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='bot_config' AND column_name='{column['name']}'"
            ))
            
            if not result.fetchone():
                logger.info(f"Adding {column['name']} column to bot_config table")
                db.session.execute(db.text(
                    f"ALTER TABLE bot_config ADD COLUMN {column['name']} {column['type']}"
                ))
                
        # Handle existing API keys - migrate from Binance to KuCoin if available
        configs = BotConfig.query.all()
        for config in configs:
            # If KuCoin fields are empty but legacy fields exist, copy them over
            if config.api_key and not config.kucoin_api_key:
                config.kucoin_api_key = config.api_key
            if config.api_secret and not config.kucoin_api_secret:
                config.kucoin_api_secret = config.api_secret
                
        db.session.commit()
        logger.info("KuCoin API credentials migration completed successfully")
    except Exception as e:
        logger.error(f"KuCoin API credentials migration failed: {str(e)}")
        db.session.rollback()
        raise

if __name__ == "__main__":
    migrate_database()
    logger.info("Migration process completed")