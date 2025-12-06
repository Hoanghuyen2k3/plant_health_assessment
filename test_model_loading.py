"""
Quick test script to verify model can be loaded
Run this before the demo to ensure everything works
"""

import torch
from pathlib import Path
from app import load_model, SimpleCNN, BaselineResNet18

def test_model_loading():
    """Test if the model can be loaded successfully"""
    model_path = "models/best_health_model.pth"
    
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    # Check if file exists
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model file not found at: {model_path}")
        print(f"   Please ensure the model is saved at this location.")
        return False
    
    print(f"✓ Model file found: {model_path}")
    
    # Try loading
    try:
        print("\nAttempting to load model (auto-detect)...")
        model, device, model_type = load_model(model_path, model_type='auto')
        print(f"✓ Model loaded successfully!")
        print(f"  - Type: {model_type}")
        print(f"  - Device: {device}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print("\nTrying manual detection...")
        
        # Try SimpleCNN
        try:
            print("  Trying SimpleCNN...")
            model, device, model_type = load_model(model_path, model_type='SimpleCNN')
            print(f"✓ Loaded as SimpleCNN!")
            return True
        except Exception as e1:
            print(f"  ✗ SimpleCNN failed: {e1}")
        
        # Try ResNet18
        try:
            print("  Trying ResNet18...")
            model, device, model_type = load_model(model_path, model_type='ResNet18')
            print(f"✓ Loaded as ResNet18!")
            return True
        except Exception as e2:
            print(f"  ✗ ResNet18 failed: {e2}")
        
        print("\n❌ Could not load model with any architecture.")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print("\n" + "=" * 60)
    if success:
        print("✅ Model loading test PASSED!")
        print("   You're ready to run the Streamlit app!")
    else:
        print("❌ Model loading test FAILED!")
        print("   Please check the model file and try again.")
    print("=" * 60)

