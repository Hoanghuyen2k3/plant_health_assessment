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
- Model file: `models/best_health_model.pth`

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

## 📁 Project Structure

```
Project/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── models/
│   └── best_health_model.pth   # Trained model (required)
├── ML_Project_Cleaned (1).ipynb  # Training notebook
└── README_DEMO.md             # This file
```

## 🔧 Configuration

### Model Path
If your model is saved in a different location, update the path in the sidebar:
- Default: `models/best_health_model.pth`
- Or specify full path: `/path/to/your/model.pth`

### Model Type
The app can auto-detect the model architecture, or you can manually specify:
- `auto` (default): Tries SimpleCNN first, then ResNet18
- `SimpleCNN`: Force SimpleCNN architecture
- `ResNet18`: Force ResNet18 architecture

## 🎓 Demonstration Checklist

For your project demonstration, make sure to:

- [ ] **Load the model successfully** (show model type and device)
- [ ] **Upload a healthy plant image** (show prediction and confidence)
- [ ] **Upload a diseased plant image** (show different prediction)
- [ ] **Explain the pipeline**: Upload → Preprocess → Inference → Display
- [ ] **Show confidence scores** and explain what they mean
- [ ] **Demonstrate robustness**: Try different images, formats
- [ ] **Highlight key features**: Real-time, interactive, user-friendly

## 💡 Tips for Demo

1. **Prepare Test Images**: Have a few example images ready (healthy and diseased)
2. **Explain the Pipeline**: Walk through each step of the inference process
3. **Show Confidence**: Explain how confidence scores indicate model certainty
4. **Discuss Robustness**: Mention error handling, different formats, etc.
5. **Highlight Features**: Point out the interactive elements and visualizations

## 🐛 Troubleshooting

### Model Not Found
- Check that `models/best_health_model.pth` exists
- Update the model path in the sidebar if saved elsewhere

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### CUDA Errors
- The app works on both CPU and GPU
- If CUDA is not available, it will automatically use CPU

## 📊 Model Performance

- **Test Accuracy**: ~99.5%
- **Classes**: Healthy, Diseased
- **Input**: 224×224 RGB images
- **Architecture**: CNN-based (SimpleCNN or ResNet18)

## 🎯 Next Steps

1. Run the app: `streamlit run app.py`
2. Load your model using the sidebar
3. Upload test images
4. Practice your demonstration!

---

**Good luck with your project demonstration! 🌱**

