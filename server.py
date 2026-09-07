"""
FastAPI server for the LSTM dead-reckoning model.
Confirmed against the actual model file:
    Input shape  : (50, 6)   -> 50 timesteps, 6 features
    Output shape : (2,)      -> [dx, dy] displacement in meters

Run with:
    pip install fastapi uvicorn tensorflow numpy --break-system-packages
    python server.py

Then it's live at http://<your-laptop-ip>:8000
The app developer hits http://<your-laptop-ip>:8000/predict from the phone
(same wifi network as this laptop).

-----------------------------------------------------------------------------
REQUEST FORMAT (what the app sends):
{
  "samples": [
    [ax, ay, az, gx, gy, gz],   <- timestep 1
    [ax, ay, az, gx, gy, gz],   <- timestep 2
    ...                          (exactly 50 rows total)
  ]
}
Feature order in each row MUST match training order exactly.

RESPONSE FORMAT (already inverse-transformed — real meters, ready to use):
{
  "dx": 1.23,   <- meters, east-west displacement over this 5s window
  "dy": -0.45   <- meters, north-south displacement over this 5s window
}
-----------------------------------------------------------------------------
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf

# ---------------------------------------------------------------------------
# CONFIG — confirmed from model.h5 / model.keras
# ---------------------------------------------------------------------------
MODEL_PATH = "dead_reckoning_model.keras"   # use the .keras file (h5 also provided as backup)
WINDOW_SIZE = 50
NUM_FEATURES = 6

# ---------------------------------------------------------------------------
# StandardScaler values from training — confirmed from notebook output.
# Input (X) is scaled before feeding the model, and output (y) was ALSO
# scaled during training — so predictions must be inverse-transformed back
# to real meters before returning them to the app.
# ---------------------------------------------------------------------------
FEATURE_MEAN = np.array([
    -0.008648344617715778, 0.010714699036443351, 0.04133052272808403,
    0.0013294360453619755, -0.007740909551146749, 0.001673381899857959,
], dtype=np.float32)

FEATURE_SCALE = np.array([
    1.072885398394555, 1.0661582151076423, 0.5658776963943067,
    0.1300291179803206, 0.1447678929875961, 0.06441153353753133,
], dtype=np.float32)

# Output (y) scaler — used to INVERSE-transform the model's raw prediction
# back into real meters: real_value = (scaled_value * scale) + mean
TARGET_MEAN = np.array([0.010158394219499253, 0.05602831172330583], dtype=np.float32)
TARGET_SCALE = np.array([8.100725349061884, 6.522277938505198], dtype=np.float32)

# ---------------------------------------------------------------------------
app = FastAPI(title="Dead Reckoning Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Input shape:", model.input_shape, "| Output shape:", model.output_shape)

if list(model.input_shape) != [None, WINDOW_SIZE, NUM_FEATURES]:
    print(f"WARNING: model.input_shape {model.input_shape} does not match "
          f"expected (None, {WINDOW_SIZE}, {NUM_FEATURES}) — check CONFIG above.")


class IMUWindow(BaseModel):
    samples: list[list[float]] = Field(
        ..., description=f"Shape ({WINDOW_SIZE}, {NUM_FEATURES}) — 50 timesteps x 6 features"
    )


class PredictionResponse(BaseModel):
    dx: float  # meters, east-west displacement over the 5s window
    dy: float  # meters, north-south displacement over the 5s window


def preprocess(samples: np.ndarray) -> np.ndarray:
    """Apply the SAME StandardScaler normalization used during training."""
    x = (samples - FEATURE_MEAN) / FEATURE_SCALE
    return x[np.newaxis, ...]  # (1, 50, 6)


@app.get("/")
def health():
    return {
        "status": "ok",
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "model_input_shape": str(model.input_shape),
        "model_output_shape": str(model.output_shape),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(window: IMUWindow):
    samples = np.array(window.samples, dtype=np.float32)

    if samples.shape != (WINDOW_SIZE, NUM_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape ({WINDOW_SIZE}, {NUM_FEATURES}), got {samples.shape}",
        )

    x = preprocess(samples)
    pred_scaled = model.predict(x, verbose=0)[0]  # shape (2,) -> scaled [dx, dy]

    # Inverse-transform back to real meters (model was trained on scaled y)
    pred_meters = (pred_scaled * TARGET_SCALE) + TARGET_MEAN

    return PredictionResponse(dx=float(pred_meters[0]), dy=float(pred_meters[1]))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)