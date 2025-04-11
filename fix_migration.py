from app import app, db
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_migration():
    with app.app_context():
        try:
            # Add the KuCoin API fields
            logger.info("Adding KuCoin API fields to database...")
            
            # Create columns for KuCoin API credentials if they don't exist
            db.session.execute(text("ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS kucoin_api_key VARCHAR(256)"))
            db.session.execute(text("ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS kucoin_api_secret VARCHAR(256)"))
            db.session.execute(text("ALTER TABLE bot_config ADD COLUMN IF NOT EXISTS kucoin_passphrase VARCHAR(256)"))
            
            # Migrate existing API credentials
            db.session.execute(text("""
                UPDATE bot_config 
                SET kucoin_api_key = api_key 
                WHERE api_key IS NOT NULL AND kucoin_api_key IS NULL
            """))
            
            db.session.execute(text("""
                UPDATE bot_config 
                SET kucoin_api_secret = api_secret 
                WHERE api_secret IS NOT NULL AND kucoin_api_secret IS NULL
            """))
            
            db.session.commit()
            logger.info("KuCoin API fields migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    apply_migration()