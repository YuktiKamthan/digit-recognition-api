# 🔢 Digit Recognition API

A production-ready REST API for handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.99%25-success.svg)

## 🎯 Features

- **High Accuracy**: CNN model achieving 98.99% accuracy on MNIST test dataset
- **REST API**: FastAPI-based service with automatic interactive documentation
- **Real-time Predictions**: Sub-second inference time for digit recognition
- **Easy to Use**: Upload any handwritten digit image and get instant predictions
- **Production Ready**: Includes error handling, input validation, and comprehensive logging

## 🏗️ Architecture

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Training**: Google Colab with T4 GPU
- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Accuracy**: 98.99% on test set

### CNN Layers
```
Input (28x28x1) 
→ Conv2D (32 filters) + ReLU + MaxPooling + Dropout
→ Conv2D (64 filters) + ReLU + MaxPooling + Dropout
→ Flatten
→ Dense (128) + ReLU + Dropout
→ Dense (10) + Softmax
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/digit_recognition_api.git
cd digit_recognition_api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the API**
```bash
python3 api.py
```

4. **Open your browser**
```
http://localhost:8000/docs
```

## 📖 API Documentation

### Endpoints

#### `GET /`
Welcome endpoint with API information

#### `GET /health`
Health check endpoint

#### `POST /predict`
Upload an image and get digit prediction

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (PNG, JPG, etc.)

**Response:**
```json
{
  "success": true,
  "predicted_digit": 5,
  "confidence": "98.45%",
  "confidence_score": 0.9845,
  "all_probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    ...
    "5": 0.9845,
    ...
  },
  "timestamp": "2024-12-12T13:15:00"
}
```

#### `GET /model_info`
Get information about the trained model

## 🧪 Testing the API

### Using the Interactive Docs
1. Go to `http://localhost:8000/docs`
2. Click on `POST /predict`
3. Click "Try it out"
4. Upload a digit image
5. Click "Execute"

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_digit.png"
```

### Using Python
```python
import requests

with open('test_digit.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

## 📁 Project Structure
```
digit_recognition_api/
├── api.py                          # FastAPI application
├── digit_recognition_model.h5      # Trained CNN model
├── model_info.pkl                  # Model metadata
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Technologies Used

- **Machine Learning**: TensorFlow, Keras, NumPy
- **API Framework**: FastAPI, Uvicorn
- **Image Processing**: Pillow (PIL)
- **Training Platform**: Google Colab (T4 GPU)
- **Development**: Python 3.12, Cursor IDE

## 📊 Model Performance

- **Training Accuracy**: ~99.5%
- **Test Accuracy**: 98.99%
- **Training Time**: ~2-3 minutes on Google Colab T4 GPU
- **Inference Time**: <100ms per image

## 🎓 What I Learned

- Building and training CNNs from scratch using TensorFlow/Keras
- Implementing production-ready REST APIs with FastAPI
- Model deployment and serving best practices
- Image preprocessing and data pipeline optimization
- Working with Google Colab for GPU-accelerated training

## 🔮 Future Improvements

- [ ] Add Docker containerization
- [ ] Deploy to cloud (Heroku/Railway/Render)
- [ ] Create web UI for easier testing
- [ ] Add model versioning and A/B testing
- [ ] Implement batch prediction endpoint
- [ ] Add monitoring and logging dashboard
- [ ] Support for multiple digit recognition

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Yukti Kamthan**
- LinkedIn: [linkedin.com/in/yuktikamthan](https://linkedin.com/in/yuktikamthan)
- Email: yuktikamthan@gmail.com

## 🙏 Acknowledgments

- MNIST Dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- FastAPI Documentation
- TensorFlow/Keras Documentation
- Google Colab for free GPU access

---

⭐ If you found this project helpful, please consider giving it a star!