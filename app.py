"""
Plant Health Assessment - Streamlit Demo App
============================================
Interactive demonstration of the plant health classification model.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import platform
import urllib.request
import os

# Page configuration
st.set_page_config(
    page_title="Plant Health Assessment",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #ecf0f1;
        margin: 1rem 0;
    }
    .healthy {
        color: #2ecc71;
        font-weight: bold;
    }
    .diseased {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Download model from GitHub Release if not exists
@st.cache_resource
def download_model_if_needed(model_name="best_health_model.pth"):
    """Download model from GitHub release if it doesn't exist locally"""
    model_path = Path(model_name)
    
    if not model_path.exists():
        st.info("📥 Downloading model from GitHub Release... This may take a moment.")
        try:
            url = "https://github.com/Hoanghuyen2k3/plant_health_assessment/releases/download/v1.0/best_health_model.pth"
            urllib.request.urlretrieve(url, model_path)
            st.success(f"✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            st.info("💡 Please ensure you have internet connection and the release URL is correct.")
            return None
    return model_path

# Model Architecture Definitions
class SimpleCNN(nn.Module):
    """Simple CNN baseline model"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class BaselineResNet18(nn.Module):
    """ResNet18 baseline model"""
    def __init__(self, num_classes=2, pretrained=True):
        super(BaselineResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model(model_path, model_type='auto'):
    """
    Load the trained model
    
    Args:
        model_path: Path to the saved model
        model_type: 'SimpleCNN', 'ResNet18', or 'auto' (tries both)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try to load as SimpleCNN first
    if model_type == 'auto' or model_type == 'SimpleCNN':
        try:
            model = SimpleCNN(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            return model, device, 'SimpleCNN'
        except Exception as e:
            if model_type == 'SimpleCNN':
                raise e
            # If auto and SimpleCNN fails, try ResNet18
            pass
    
    # Try ResNet18
    if model_type == 'auto' or model_type == 'ResNet18':
        try:
            model = BaselineResNet18(num_classes=2, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            return model, device, 'ResNet18'
        except Exception as e:
            if model_type == 'ResNet18':
                raise e
            raise RuntimeError(f"Could not load model. Tried both SimpleCNN and ResNet18. Error: {e}")
    
    raise ValueError(f"Unknown model type: {model_type}")

def get_clipboard_image():
    """
    Get image from system clipboard
    Supports Windows, macOS, and Linux
    """
    try:
        # Try using PIL's ImageGrab (works on Windows and macOS)
        if platform.system() in ['Windows', 'Darwin']:  # Darwin is macOS
            image = ImageGrab.grabclipboard()
            if image is not None and isinstance(image, Image.Image):
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image
        else:
            # Linux support using xclip
            import subprocess
            result = subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o'],
                                  capture_output=True, check=False)
            if result.returncode == 0 and result.stdout:
                image = Image.open(io.BytesIO(result.stdout))
                return image.convert('RGB')
        return None
    except Exception as e:
        print(f"Clipboard error: {e}")
        return None

def get_transforms():
    """Get image transformation pipeline (same as validation/test)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, device, transform):
    """
    Predict plant health from image
    
    Returns:
        prediction: 'healthy' or 'diseased'
        confidence: confidence score (0-1)
        probabilities: dict with class probabilities
    """
    # Preprocess image
    if isinstance(image, Image.Image):
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        # If numpy array, convert to PIL first
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_names = ['healthy', 'diseased']
    prediction = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    prob_dict = {
        'healthy': probabilities[0][0].item(),
        'diseased': probabilities[0][1].item()
    }
    
    return prediction, confidence_score, prob_dict

def create_confidence_chart(probabilities):
    """Create a bar chart showing prediction probabilities"""
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#2ecc71' if c == 'healthy' else '#e74c3c' for c in classes]
    
    bars = ax.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Health Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Auto-download model from GitHub
        model_path = download_model_if_needed("best_health_model.pth")
        
        if model_path is None:
            st.error("❌ Cannot proceed without model file")
            return
        
        model_type = st.selectbox(
            "Model Type",
            options=['auto', 'SimpleCNN', 'ResNet18'],
            help="Auto-detect or manually specify model architecture"
        )
        
        # Load model button
        load_model_btn = st.button("🔄 Load Model", type="primary")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        # Model status
        if 'model_loaded' in st.session_state and st.session_state.model_loaded:
            st.success(f"✅ Model Loaded: {st.session_state.model_type}")
            st.info(f"Device: {st.session_state.device.type.upper()}")
        else:
            st.warning("⚠️ Model not loaded yet")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. **Load Model**: Click the "Load Model" button
        2. **Upload Image**: Use the file uploader below
        3. **View Results**: See prediction and confidence scores
        4. **Explore**: Try different plant images!
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Plant Image</h2>', unsafe_allow_html=True)
        
        # Tabs for different input methods
        tab1, tab2 = st.tabs(["📁 Upload File", "📋 Paste from Clipboard"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a plant image to assess its health"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.session_state.uploaded_image = image
        
        with tab2:
            paste_btn = st.button("📋 Paste Image from Clipboard", use_container_width=True, key="paste_btn")
            if paste_btn:
                with st.spinner("Getting image from clipboard..."):
                    clipboard_image = get_clipboard_image()
                if clipboard_image is not None:
                    st.session_state.uploaded_image = clipboard_image
                    st.session_state.paste_active = True
                    st.success("✅ Image pasted successfully!")
                else:
                    st.error("❌ No image found in clipboard.")
                    st.info("💡 Try copying an image from your browser or screenshot tool first.")
            
            # Show pasted image if it exists
            if 'uploaded_image' in st.session_state and st.session_state.get('paste_active', False):
                st.image(st.session_state.uploaded_image, caption="Pasted Image", use_container_width=True)
        
        # Load model
        if load_model_btn:
            try:
                with st.spinner("Loading model..."):
                    model, device, detected_type = load_model(model_path, model_type)
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.model_type = detected_type
                    st.session_state.model_loaded = True
                    st.success(f"✅ Model loaded successfully! Type: {detected_type}")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.session_state.model_loaded = False
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Check if model is loaded
        if 'model_loaded' not in st.session_state or not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first using the sidebar!")
        elif 'uploaded_image' not in st.session_state:
            st.info("📤 Please upload or paste an image first")
        else:
            # Make prediction
            try:
                transform = get_transforms()
                prediction, confidence, probabilities = predict_image(
                    st.session_state.model,
                    st.session_state.uploaded_image,
                    st.session_state.device,
                    transform
                )
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Prediction result
                if prediction == 'healthy':
                    st.markdown(f'<h2 class="healthy">✅ Prediction: HEALTHY</h2>', unsafe_allow_html=True)
                    st.balloons()  # Celebration for healthy plants!
                else:
                    st.markdown(f'<h2 class="diseased">⚠️ Prediction: DISEASED</h2>', unsafe_allow_html=True)
                
                # Confidence score
                confidence_percent = confidence * 100
                st.metric("Confidence", f"{confidence_percent:.2f}%")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability chart
                st.markdown("### 📊 Probability Distribution")
                fig = create_confidence_chart(probabilities)
                st.pyplot(fig)
                
                # Detailed probabilities
                with st.expander("📋 Detailed Probabilities"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Healthy", f"{probabilities['healthy']*100:.2f}%")
                    with col_b:
                        st.metric("Diseased", f"{probabilities['diseased']*100:.2f}%")
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    # Additional Information Section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("### 🎯 Purpose")
        st.markdown("""
        This system uses deep learning to automatically 
        assess plant health by analyzing visual indicators 
        such as color changes, dryness, or disease symptoms.
        """)
    
    with info_col2:
        st.markdown("### 🔬 Model Details")
        if 'model_type' in st.session_state:
            st.markdown(f"**Architecture**: {st.session_state.model_type}")
        else:
            st.markdown("**Architecture**: CNN-based")
        st.markdown("""
        - **Classes**: Healthy, Diseased
        - **Input Size**: 224×224 pixels
        - **Accuracy**: ~99.5%
        """)
    
    with info_col3:
        st.markdown("### 💡 Usage Tips")
        st.markdown("""
        - Use clear, well-lit images
        - Focus on the plant leaves
        - Ensure the plant is centered
        - Works best with close-up shots
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>"
        "🌱 Plant Health Assessment System | Machine Learning Project"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

