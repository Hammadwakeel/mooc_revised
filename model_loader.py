import os
import logging
import joblib
import pickle
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default artifact paths (override with env vars if you want)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/my_model_balanced_keras.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "model/scaler.pkl")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "model/label_encoder.pkl")

# Globals holding loaded artifacts
MODEL = None
SCALER = None
LABEL_ENCODER = None

def init_artifacts():
    """Load model, scaler and label encoder into global vars."""
    global MODEL, SCALER, LABEL_ENCODER

    # Load Keras model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded.")
    except Exception as e:
        logger.exception(f"Failed to load Keras model: {e}")
        raise

    # Load scaler
    try:
        if os.path.exists(SCALER_PATH):
            logger.info(f"Loading scaler from {SCALER_PATH}")
            SCALER = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded.")
        else:
            logger.warning(f"Scaler path {SCALER_PATH} not found. Predictions may fail if inputs are not pre-scaled.")
            SCALER = None
    except Exception as e:
        logger.exception(f"Failed to load scaler: {e}")
        SCALER = None

    # Load LabelEncoder
    try:
        if os.path.exists(ENCODER_PATH):
            logger.info(f"Loading label encoder from {ENCODER_PATH}")
            try:
                LABEL_ENCODER = joblib.load(ENCODER_PATH)
            except Exception:
                with open(ENCODER_PATH, "rb") as f:
                    LABEL_ENCODER = pickle.load(f)
            logger.info("Label encoder loaded.")
        else:
            logger.warning(f"Label encoder path {ENCODER_PATH} not found. Returning numeric classes instead.")
            LABEL_ENCODER = None
    except Exception as e:
        logger.exception(f"Failed to load label encoder: {e}")
        LABEL_ENCODER = None

def get_model():
    return MODEL

def get_scaler():
    return SCALER

def get_label_encoder():
    return LABEL_ENCODER
