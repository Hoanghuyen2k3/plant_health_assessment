# Plant Health Assessment - Demo App

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Requirements

- Python 3.8+
- PyTorch
- Streamlit

## 🎯 Features

### Interactive Demo Features:
- ✅ **Image Upload**: Upload plant images via drag-and-drop or file picker
- ✅ **Real-time Prediction**: Instant health assessment (Healthy/Diseased)
- ✅ **Confidence Scores**: See prediction probabilities
- ✅ **Visualizations**: Probability distribution charts
- ✅ **Model Information**: Display model architecture and device
- ✅ **User-Friendly UI**: Clean, modern interface

### What the Demo Shows:
1. **End-to-End Pipeline**: 
   - Image upload → Preprocessing → Model inference → Results display
   
2. **Robustness**:
   - Handles different image formats (PNG, JPG, JPEG)
   - Auto-detects model architecture
   - Works on CPU or GPU
   - Clear error handling

3. **Interactive Elements**:
   - Real-time predictions
   - Visual feedback (confetti for healthy plants!)
   - Detailed probability breakdowns
   - Model status indicators