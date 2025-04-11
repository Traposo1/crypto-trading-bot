import os
import logging
from app import app, db
from models import BotConfig
from config import get_config, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database_settings():
    """Reset the database settings and create a new configuration that includes environment variables."""
    try:
        # Delete current config from database
        logger.info("Deleting existing configuration from database...")
        BotConfig.query.delete()
        db.session.commit()
        
        # Load environment variables for API keys
        env_config = {}
        
        # KuCoin API keys (priority)
        if os.environ.get("KUCOIN_API_KEY"):
            env_config["kucoin_api_key"] = os.environ.get("KUCOIN_API_KEY")
            logger.info("Found KUCOIN_API_KEY in environment")
        if os.environ.get("KUCOIN_API_SECRET"):
            env_config["kucoin_api_secret"] = os.environ.get("KUCOIN_API_SECRET")
            logger.info("Found KUCOIN_API_SECRET in environment")
        if os.environ.get("KUCOIN_PASSPHRASE"):
            env_config["kucoin_passphrase"] = os.environ.get("KUCOIN_PASSPHRASE")
            logger.info("Found KUCOIN_PASSPHRASE in environment")
        
        # Legacy Binance API keys
        if os.environ.get("BINANCE_API_KEY"):
            env_config["api_key"] = os.environ.get("BINANCE_API_KEY")
            logger.info("Found BINANCE_API_KEY in environment")
        if os.environ.get("BINANCE_API_SECRET"):
            env_config["api_secret"] = os.environ.get("BINANCE_API_SECRET")
            logger.info("Found BINANCE_API_SECRET in environment")
        
        # Create new config with default values plus any environment variables
        config_data = {**DEFAULT_CONFIG, **env_config}
        
        # Create and save new config
        new_config = BotConfig()
        for key, value in config_data.items():
            if hasattr(new_config, key):
                logger.info(f"Setting {key} in new configuration")
                setattr(new_config, key, value)
        
        # Add to database
        db.session.add(new_config)
        db.session.commit()
        logger.info("New configuration created successfully")
        
        # Verify by loading the config
        current_config = get_config()
        logger.info("Configuration verified in database")
        
        # Print sanitized configuration (hiding actual API keys)
        safe_config = current_config.copy()
        api_fields = ['api_key', 'api_secret', 'kucoin_api_key', 'kucoin_api_secret', 'kucoin_passphrase']
        for field in api_fields:
            if field in safe_config and safe_config[field]:
                safe_config[field] = '******'
        
        logger.info(f"New config: {safe_config}")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting database settings: {str(e)}")
        db.session.rollback()
        return False

if __name__ == "__main__":
    logger.info("Starting database settings reset...")
    # Use Flask application context
    with app.app_context():
        success = reset_database_settings()
        if success:
            logger.info("Database settings reset completed successfully")
        else:
            logger.error("Database settings reset failed")