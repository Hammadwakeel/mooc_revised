from fastapi import FastAPI
from routes import router
from model_loader import init_artifacts

app = FastAPI(title="Keras Inference API (multi-file, single-folder)")
app.include_router(router)

@app.on_event("startup")
def startup_event():
    # Load model/scaler/encoder into memory at startup
    init_artifacts()
