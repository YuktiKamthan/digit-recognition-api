# api.py - Enhanced with Monitoring
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from datetime import datetime
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Digit Recognition API with Monitoring",
    description="CNN-based digit recognition with production monitoring",
    version="2.0"
)

# Global metrics storage
metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_errors": 0,
    "predictions_by_digit": {str(i): 0 for i in range(10)},
    "start_time": datetime.now().isoformat(),
    "model_version": "v1.0",
    "model_accuracy": 0.9899
}

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model('digit_recognition_model.h5')
        logger.info("✅ Model loaded successfully at startup")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array / 255.0
        
        # Reshape for model (add batch and channel dimensions)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# ============================================
# NEW: Health Check Endpoint
# ============================================
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    Returns: Service status and model availability
    """
    try:
        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "model_loaded": False,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": metrics["model_version"],
            "uptime": (datetime.now() - datetime.fromisoformat(metrics["start_time"])).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)}
        )

# ============================================
# NEW: Metrics Endpoint
# ============================================
@app.get("/metrics")
async def get_metrics():
    """
    Metrics endpoint for monitoring
    Returns: API usage statistics and model performance
    """
    try:
        uptime_seconds = (datetime.now() - datetime.fromisoformat(metrics["start_time"])).total_seconds()
        
        return {
            "service_metrics": {
                "total_requests": metrics["total_requests"],
                "successful_predictions": metrics["successful_predictions"],
                "failed_predictions": metrics["failed_predictions"],
                "total_errors": metrics["total_errors"],
                "success_rate": (
                    metrics["successful_predictions"] / metrics["total_requests"] * 100 
                    if metrics["total_requests"] > 0 else 0
                ),
                "uptime_seconds": uptime_seconds,
                "uptime_hours": round(uptime_seconds / 3600, 2)
            },
            "model_metrics": {
                "model_version": metrics["model_version"],
                "model_accuracy": metrics["model_accuracy"],
                "predictions_by_digit": metrics["predictions_by_digit"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ============================================
# Root Endpoint
# ============================================
@app.get("/")
async def root():
    return {
        "message": "Digit Recognition API with Monitoring",
        "version": "2.0",
        "endpoints": {
            "predict": "POST /predict - Upload image for digit prediction",
            "health": "GET /health - Check service health",
            "metrics": "GET /metrics - Get API metrics",
            "docs": "GET /docs - Interactive API documentation"
        },
        "model_accuracy": metrics["model_accuracy"],
        "status": "running"
    }

# ============================================
# ENHANCED: Prediction Endpoint with Monitoring
# ============================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    Enhanced with logging and metrics tracking
    """
    start_time = datetime.now()
    
    # Update request counter
    metrics["total_requests"] += 1
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            metrics["failed_predictions"] += 1
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        
        # Update success metrics
        metrics["successful_predictions"] += 1
        metrics["predictions_by_digit"][str(predicted_digit)] += 1
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Log successful prediction
        logger.info(
            f"✅ Prediction: {predicted_digit} | "
            f"Confidence: {confidence:.4f} | "
            f"Time: {inference_time:.3f}s"
        )
        
        # Prepare response
        response = {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "all_probabilities": {
                str(i): float(predictions[0][i]) 
                for i in range(10)
            },
            "inference_time_seconds": inference_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        metrics["failed_predictions"] += 1
        metrics["total_errors"] += 1
        
        logger.error(f"❌ Prediction failed: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================
# NEW: Model Info Endpoint
# ============================================
@app.get("/model_info")
async def model_info():
    """Get detailed model information"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "model_version": metrics["model_version"],
            "model_accuracy": metrics["model_accuracy"],
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_parameters": model.count_params(),
            "layers": len(model.layers),
            "architecture": [layer.name for layer in model.layers]
        }
    except Exception as e:
        logger.error(f"Model info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Digit Recognition API with Monitoring")
    print("=" * 60)
    print("📊 New endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /metrics - API metrics")
    print("   • GET  /model_info - Model details")
    print("   • POST /predict - Digit prediction")
    print("   • GET  /docs - Swagger UI")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)