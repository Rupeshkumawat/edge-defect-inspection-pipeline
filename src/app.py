from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from src.pipeline import preprocess
from src.batcher import DynamicMicroBatcher

batcher: DynamicMicroBatcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    # Using models/model_fp32.onnx for now
    batcher = DynamicMicroBatcher("models/model_fp32.onnx", max_batch_size=4, timeout_ms=15.0)
    await batcher.start()
    yield

app = FastAPI(title="Edge Defect Inspection Microservice", lifespan=lifespan)

class Detection(BaseModel):
    class_id: int
    confidence: float
    bbox: List[int]

class InspectionResponse(BaseModel):
    status: str
    defects_found: int
    detections: List[Detection]

@app.post("/v1/inspect", response_model=InspectionResponse)
async def inspect_frame(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image format.")

    try:
        content = await file.read()
        tensor, ratio, pad, orig_shape = preprocess(content)
        detections = await batcher.submit(tensor, ratio, pad, orig_shape)
        
        status = "DEFECT_DETECTED" if len(detections) > 0 else "PASS"
        return {
            "status": status,
            "defects_found": len(detections),
            "detections": detections
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")
