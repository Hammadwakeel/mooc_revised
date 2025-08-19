import numpy as np
from typing import List, Dict, Any

def preprocess_features(raw: np.ndarray, scaler) -> np.ndarray:
    """
    Accepts (n_samples, n_features) and returns shaped array for model: (n_samples, 1, n_features)
    """
    if raw.ndim != 2:
        raise ValueError("Input must be 2D array shape (n_samples, n_features)")

    if scaler is not None:
        try:
            scaled = scaler.transform(raw)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")
    else:
        scaled = raw

    reshaped = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
    return reshaped

def predict_from_array(X: np.ndarray, model, scaler, label_encoder) -> List[Dict[str, Any]]:
    X_proc = preprocess_features(X, scaler)
    probs = model.predict(X_proc)

    results = []
    for i in range(probs.shape[0]):
        prob = probs[i].tolist()
        class_idx = int(np.argmax(prob))
        if label_encoder is not None:
            try:
                label = str(label_encoder.inverse_transform([class_idx])[0])
            except Exception:
                label = str(class_idx)
        else:
            label = str(class_idx)

        results.append({
            "predicted_label": label,
            "predicted_class_index": class_idx,
            "probabilities": prob,
        })
    return results
