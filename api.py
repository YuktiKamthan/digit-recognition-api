# api.py - FastAPI service for digit recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Digit Recognition API",
    description="Upload a handwritten digit image and get prediction",
    version="1.0.0"
)

# Load model at startup
print("Loading model...")
try:
    model = tf.keras.models.load_model('digit_recognition_model.h5')
    print("✓ Model loaded successfully!")
    
    # Load model info
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print(f"✓ Model accuracy: {model_info['accuracy'] * 100:.2f}%")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Helper function to preprocess image
def preprocess_image(image_data):
    """Convert uploaded image to format expected by model"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize (0-255 to 0-1)
        image_array = image_array.astype('float32') / 255.0
        
        # Reshape for model
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Digit Recognition API!",
        "status": "running",
        "model_accuracy": f"{model_info['accuracy'] * 100:.2f}%",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict (POST)",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API is working"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# Main prediction endpoint
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Upload an image and get digit prediction
    """
    try:
        # Read uploaded file
        image_data = await file.read()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        
        # All probabilities
        all_probabilities = {
            str(i): float(predictions[0][i]) 
            for i in range(10)
        }
        
        # Response
        response = {
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_score": round(confidence, 4),
            "all_probabilities": all_probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Model info endpoint
@app.get("/model_info")
def get_model_info():
    """Get information about the model"""
    return {
        "model_type": "Convolutional Neural Network (CNN)",
        "framework": "TensorFlow/Keras",
        "accuracy": f"{model_info['accuracy'] * 100:.2f}%",
        "trained_date": model_info.get('trained_date', 'N/A'),
        "training_platform": model_info.get('platform', 'Google Colab')
    }

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Digit Recognition API")
    print("="*50)
    print("\nAPI will be available at:")
    print("  - Main: http://localhost:8000/")
    print("  - Docs: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)