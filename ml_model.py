import os
import logging
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from bot.indicators import get_indicator_features

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = None
MODEL_PATH = "model"
MODEL_FILE = os.path.join(MODEL_PATH, "model.pkl")
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.pkl")

def initialize_model() -> None:
    """Initialize the machine learning model, creating directory if needed"""
    global model, scaler
    
    try:
        # Create model directory if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        # Check if model and scaler files exist
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            # Load existing model and scaler
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            logger.info("Loaded existing ML model and scaler")
        else:
            # Create ensemble model and scaler
            # Define base models for ensemble
            rf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=6, 
                min_samples_split=5,
                random_state=42
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            ada = AdaBoostClassifier(
                n_estimators=50,
                learning_rate=0.1,
                random_state=42
            )
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                activation='relu',
                solver='adam', 
                alpha=0.0001,
                max_iter=200,
                random_state=42
            )
            
            # Create ensemble voting classifier
            model = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb),
                    ('ada', ada),
                    ('mlp', mlp)
                ],
                voting='soft'
            )
            
            scaler = StandardScaler()
            logger.info("Created new ensemble ML model and scaler")
    
    except Exception as e:
        logger.error(f"Error initializing ML model: {str(e)}")
        model = None
        scaler = None

def prepare_training_data(df: pd.DataFrame, target_lookahead: int = 12, threshold_pct: float = 0.01, 
                       include_multiclass: bool = True) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Prepare data for model training with advanced target calculation
    
    Args:
        df: DataFrame with OHLCV data and indicators
        target_lookahead: Number of periods to look ahead for target calculation
        threshold_pct: Percentage threshold for determining significant price movements
        include_multiclass: Whether to include a multiclass target variable
        
    Returns:
        Tuple of (X, y_binary, y_multiclass) where:
          - X is feature DataFrame
          - y_binary is binary target (1 for up, 0 for down/no change)
          - y_multiclass is multiclass target (2=strong up, 1=moderate up, 0=no change, -1=moderate down, -2=strong down)
          - Note: y_multiclass will be None if include_multiclass is False
    """
    try:
        # Make sure we have enough data
        if df is None or len(df) < 30:
            logger.warning("Not enough data for ML training")
            return pd.DataFrame(), pd.Series(), None
            
        # Get features from indicators
        features = get_indicator_features(df)
        if not features:
            logger.warning("Feature extraction failed")
            return pd.DataFrame(), pd.Series(), None
        
        # Calculate future returns
        future_returns = df['close'].shift(-target_lookahead) / df['close'] - 1
        
        # Create binary target variable (1 if price goes up by threshold_pct+ in next periods, 0 otherwise)
        y_binary = (future_returns > threshold_pct).astype(int)
        
        # Create multiclass target for more granular predictions if requested
        y_multiclass = None
        if include_multiclass:
            # Define thresholds for different classes
            strong_up_threshold = threshold_pct * 2.0  # e.g., 2% if threshold_pct=1%
            moderate_up_threshold = threshold_pct  # e.g., 1%
            no_change_threshold = threshold_pct * 0.2  # e.g., 0.2%
            moderate_down_threshold = -threshold_pct  # e.g., -1%
            strong_down_threshold = -threshold_pct * 2.0  # e.g., -2%
            
            # Create multiclass labels
            y_multiclass = pd.Series(0, index=future_returns.index)  # Default to 'no change'
            y_multiclass = y_multiclass.mask(future_returns > strong_up_threshold, 2)  # Strong up
            y_multiclass = y_multiclass.mask((future_returns > moderate_up_threshold) & (future_returns <= strong_up_threshold), 1)  # Moderate up
            y_multiclass = y_multiclass.mask((future_returns < -no_change_threshold) & (future_returns >= moderate_down_threshold), -1)  # Moderate down
            y_multiclass = y_multiclass.mask(future_returns < strong_down_threshold, -2)  # Strong down
        
        # Align the data and remove rows with NaN values
        X = features.iloc[:-target_lookahead]  # Remove last rows where future data is not available
        y_binary = y_binary.iloc[:-target_lookahead]
        
        if include_multiclass:
            y_multiclass = y_multiclass.iloc[:-target_lookahead]
        
        # Remove any remaining NaN values 
        valid_indices = ~(X.isna().any(axis=1) | y_binary.isna())
        X = X[valid_indices]
        y_binary = y_binary[valid_indices]
        
        if include_multiclass:
            y_multiclass = y_multiclass[valid_indices]
        
        # Add some engineered features based on Ichimoku trends
        # Feature importance calculation
        feature_correlations = X.corrwith(y_binary)
        X['top_feature_mean'] = X[feature_correlations.abs().nlargest(5).index].mean(axis=1)
        
        # Add volatility-weighted features for better predictions in volatile markets
        if 'volatility' in X.columns:
            volatility_mask = X['volatility'] > X['volatility'].quantile(0.75)
            for col in ['rsi', 'macd', 'cloud_distance', 'price_above_cloud']:
                if col in X.columns:
                    X[f'{col}_vol_weighted'] = X[col] * X['volatility'].fillna(0)
        
        return X, y_binary, y_multiclass
    
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        return pd.DataFrame(), pd.Series(), None

def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train the machine learning model with enhanced features and ensemble approach
    
    Args:
        df: DataFrame with OHLCV data and indicators
        
    Returns:
        Dictionary with training metrics
    """
    global model, scaler
    
    if model is None or scaler is None:
        initialize_model()
    
    if model is None or scaler is None:
        logger.error("Failed to initialize ML model")
        return {"error": "Failed to initialize model"}
    
    try:
        # Prepare training data optimized for scalping with shorter time horizons and smaller thresholds
        X, y_binary, y_multiclass = prepare_training_data(df, 
                                                         target_lookahead=6,  # Reduced from 12 to 6 for shorter-term predictions
                                                         threshold_pct=0.005,  # Reduced from 0.01 to 0.005 (0.5%) for smaller price moves
                                                         include_multiclass=True)
        
        if len(X) < 100:
            logger.warning(f"Not enough data for training, only {len(X)} samples available")
            return {"error": "Not enough data for training"}
        
        # Split data into training and testing sets using time series split
        tscv = TimeSeriesSplit(n_splits=5)
        split = list(tscv.split(X))
        train_idx, test_idx = split[-1]  # Use the last split
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        
        # Apply feature selection if we have enough data
        if len(X_train) > 20:
            # Find most important features based on correlation with target
            feature_corr = X_train.corrwith(y_train)
            top_features = feature_corr.abs().sort_values(ascending=False).head(min(20, len(X_train.columns))).index.tolist()
            
            logger.info(f"Selected {len(top_features)} top features for training")
            
            # Keep only the most important features
            X_train = X_train[top_features]
            X_test = X_test[top_features]
        
        # Normalize features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate comprehensive metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.5,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_balance": y_train.mean(),
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
            
            # Specific trading metrics
            metrics["profit_potential"] = (tp / max(1, tp + fn)) * 100  # % of actual ups correctly predicted
            metrics["loss_avoidance"] = (tn / max(1, tn + fp)) * 100   # % of actual downs correctly predicted
        
        # Save the model and scaler
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        
        # Save feature list for consistent prediction
        feature_list = list(X_train.columns)
        with open(os.path.join(MODEL_PATH, "feature_list.pkl"), "wb") as f:
            pickle.dump(feature_list, f)
        
        logger.info(f"Model trained successfully with accuracy {metrics['accuracy']:.4f}, F1 {metrics['f1']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {"error": str(e)}

def predict_market(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[int, float]:
    """
    Generate market prediction using trained ML model with Ichimoku Cloud enhancements
    
    Args:
        df: DataFrame with OHLCV data and indicators
        config: Configuration dictionary
        
    Returns:
        Tuple of (prediction, confidence) where prediction is 1 for up, 0 for down/no change
    """
    global model, scaler
    
    if model is None or scaler is None:
        initialize_model()
    
    if model is None or scaler is None:
        logger.warning("ML model not initialized")
        return 0, 0.0
    
    try:
        # Prepare features for the latest data point
        features = get_indicator_features(df)
        
        # Check if features is valid and not empty
        if features is None or features.empty:
            logger.warning("Failed to extract features or empty feature set returned")
            return 0, 0.5  # Return neutral prediction with moderate confidence
            
        # Handle features DataFrame properly
        try:
            X = features.iloc[-1:] if len(features) > 0 else features
        except Exception as e:
            logger.warning(f"Error accessing features: {e}")
            X = features  # Use all features if there's an indexing error
        
        # Check for NaN values
        if X.isna().any().any():
            logger.warning("NaN values in features, replacing with zeros")
            X = X.fillna(0)
        
        # Add additional volatility-weighted features if needed
        if 'volatility' in X.columns:
            for col in ['rsi', 'macd', 'cloud_distance', 'price_above_cloud']:
                if col in X.columns:
                    X[f'{col}_vol_weighted'] = X[col] * X['volatility'].fillna(0)
        
        # Check if we have saved feature list to ensure consistent feature order
        feature_list_path = os.path.join(MODEL_PATH, "feature_list.pkl")
        if os.path.exists(feature_list_path):
            with open(feature_list_path, 'rb') as f:
                feature_list = pickle.load(f)
                
            # Only use features that were used during training
            common_features = [col for col in feature_list if col in X.columns]
            if common_features:
                X = X[common_features]
            else:
                logger.warning("No common features found between model and current data")
        
        # Scale features
        try:
            X_scaled = scaler.transform(X)
        except ValueError as ve:
            # Handle feature mismatch by filling missing columns with zeros
            logger.warning(f"Feature mismatch during prediction: {ve}")
            X_scaled = scaler.transform(X.fillna(0))
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        confidence = max(model.predict_proba(X_scaled)[0])
        
        # Add additional context about Ichimoku signals for more robust prediction
        ichimoku_bullish = False
        ichimoku_bearish = False
        
        # Check for specific Ichimoku Cloud patterns
        if all(c in df.columns for c in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b', 'close']):
            last_row = df.iloc[-1]
            price_above_cloud = (last_row['close'] > last_row['ichimoku_senkou_a'] and 
                               last_row['close'] > last_row['ichimoku_senkou_b'])
            price_below_cloud = (last_row['close'] < last_row['ichimoku_senkou_a'] and 
                               last_row['close'] < last_row['ichimoku_senkou_b'])
            tk_cross_bullish = last_row['ichimoku_tenkan'] > last_row['ichimoku_kijun']
            cloud_bullish = last_row['ichimoku_senkou_a'] > last_row['ichimoku_senkou_b']
            
            # Strong bullish signal
            ichimoku_bullish = (price_above_cloud and tk_cross_bullish and cloud_bullish)
            
            # Strong bearish signal
            ichimoku_bearish = (price_below_cloud and not tk_cross_bullish and not cloud_bullish)
            
            # Adjust confidence if Ichimoku signals conflict with ML prediction
            if (prediction == 1 and ichimoku_bearish) or (prediction == 0 and ichimoku_bullish):
                confidence = max(0.51, min(confidence, 0.65))  # Lower confidence on conflicting signals
            elif (prediction == 1 and ichimoku_bullish) or (prediction == 0 and ichimoku_bearish):
                confidence = min(0.99, confidence * 1.15)  # Increase confidence on confirming signals
        
        logger.debug(f"ML prediction: {prediction} with confidence {confidence:.4f}, " +
                    f"Ichimoku confirms: {'bullish' if ichimoku_bullish else 'bearish' if ichimoku_bearish else 'neutral'}")
        
        return prediction, confidence
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return 0, 0.0

def evaluate_model_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate enhanced ML model performance on historical data
    
    Args:
        df: DataFrame with OHLCV data and indicators
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    global model, scaler
    
    if model is None or scaler is None:
        initialize_model()
    
    if model is None or scaler is None:
        logger.error("ML model not initialized")
        return {"error": "ML model not initialized"}
    
    try:
        # Prepare data with scalping-optimized target calculation
        X, y_binary, y_multiclass = prepare_training_data(df, 
                                                         target_lookahead=6,  # Reduced from 12 to 6 for shorter-term predictions
                                                         threshold_pct=0.005,  # Reduced from 0.01 to 0.005 (0.5%) for smaller price moves
                                                         include_multiclass=True)
        
        if len(X) < 50:
            logger.warning(f"Not enough data for evaluation, only {len(X)} samples available")
            return {"error": "Not enough data for evaluation"}
        
        # Apply feature selection if we have a saved feature list
        feature_list_path = os.path.join(MODEL_PATH, "feature_list.pkl")
        if os.path.exists(feature_list_path):
            with open(feature_list_path, 'rb') as f:
                feature_list = pickle.load(f)
                
            # Only use features that were used during training
            common_features = [col for col in feature_list if col in X.columns]
            if common_features:
                X = X[common_features]
            else:
                logger.warning("No common features found between model and current data")
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        # Calculate comprehensive metrics
        metrics = {
            "accuracy": accuracy_score(y_binary, y_pred),
            "precision": precision_score(y_binary, y_pred, zero_division=0),
            "recall": recall_score(y_binary, y_pred, zero_division=0),
            "f1": f1_score(y_binary, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_binary, y_proba[:, 1]) if len(np.unique(y_binary)) > 1 else 0.5,
            "samples": len(X),
            "positive_ratio": y_binary.mean(),
            "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_binary, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
            
            # Trading specific metrics
            metrics["profit_potential"] = (tp / max(1, tp + fn)) * 100  # % of actual ups correctly predicted
            metrics["loss_avoidance"] = (tn / max(1, tn + fp)) * 100   # % of actual downs correctly predicted
        
        # Calculate trading performance metrics
        confidence_threshold = 0.7  # Only consider high-confidence predictions
        predicted_up = (y_proba[:, 1] > confidence_threshold)  # Confident predictions of upward movement
        correct_up = predicted_up & (y_binary == 1)
        incorrect_up = predicted_up & (y_binary == 0)
        
        # Calculate ROI with different profit/loss assumptions
        std_return = df['close'].pct_change().std() * 100  # Standard deviation of returns as percentage
        avg_up_return = max(1.0, std_return)  # Average return when correctly predicting up move
        avg_down_loss = max(1.0, std_return * 0.8)  # Average loss when incorrectly predicting up move
        
        roi = np.sum(correct_up) * avg_up_return/100 - np.sum(incorrect_up) * avg_down_loss/100
        metrics["roi"] = roi
        metrics["win_ratio"] = np.sum(correct_up) / max(1, np.sum(predicted_up))
        metrics["trade_count"] = int(np.sum(predicted_up))
        
        # Calculate effectiveness of Ichimoku-based features if available
        ichimoku_features = [col for col in X.columns if any(x in col for x in 
                                                          ['ichimoku', 'cloud', 'tk_cross', 'tenkan', 'kijun'])]
        if ichimoku_features:
            # Calculate correlation of Ichimoku features with target
            corr = X[ichimoku_features].corrwith(y_binary).abs().mean()
            metrics["ichimoku_feature_correlation"] = float(corr)
            
            # Identify top Ichimoku features by importance
            top_ichimoku = sorted([(col, abs(X[col].corr(y_binary))) 
                                  for col in ichimoku_features if not pd.isna(X[col].corr(y_binary))],
                                 key=lambda x: x[1], reverse=True)[:3]
            metrics["top_ichimoku_features"] = [f"{name} ({corr:.3f})" for name, corr in top_ichimoku]
        
        logger.info(f"Model evaluation: accuracy={metrics['accuracy']:.4f}, ROI={metrics['roi']:.4f}, " +
                    f"F1={metrics['f1']:.4f}, Win ratio={metrics['win_ratio']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {"error": str(e)}
