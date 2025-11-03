"""
Streamlit App for Satellite Image Classification
ViT + Gaussian Process (Linear Kernel)

Deploy on Streamlit Cloud or run locally:
streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import pickle
from transformers import ViTImageProcessor, ViTForImageClassification
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide"
)

# Class names for EuroSAT
CLASS_NAMES = [
    'Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway',
    'Industrial', 'Pasture', 'Permanent Crop', 'Residential',
    'River', 'Sea Lake'
]

@st.cache_resource
def load_models():
    """Load all models (cached for performance)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HF_REPO_ID = "AdityaK1729/VIT" 
    # Load ViT for direct classification
    vit_model = ViTForImageClassification.from_pretrained(HF_REPO_ID)
    vit_model.to(device)
    vit_model.eval()
    
    # Load processor
    processor = ViTImageProcessor.from_pretrained(HF_REPO_ID)
    
    # Load embedding extractor (ViT without head)
    embed_model = ViTForImageClassification.from_pretrained(HF_REPO_ID)
    embed_model.classifier = torch.nn.Identity()
    embed_model.to(device)
    embed_model.eval()
    
    # Load GP (Linear kernel)
    with open('deployment_models/gp_classifier.pkl', 'rb') as f:
        gp_model = pickle.load(f)
    
    # Load scaler
    with open('deployment_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    import json
    with open('deployment_models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return vit_model, embed_model, processor, gp_model, scaler, device, metadata

def predict_vit(image, model, processor, device):
    """Direct ViT prediction"""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    
    # Top 3
    top3_probs, top3_indices = torch.topk(probs, k=3)
    top3 = {CLASS_NAMES[idx]: float(prob) for idx, prob in zip(top3_indices, top3_probs)}
    
    return CLASS_NAMES[pred_idx], confidence, top3, probs.cpu().numpy()

def predict_gp(image, embed_model, processor, gp_model, scaler, device):
    """GP prediction with uncertainty"""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Extract 768-dim embedding
        embedding = embed_model(**inputs).logits.cpu().numpy()
    
    # Scale embedding (CRITICAL - must use same scaler as training)
    embedding_scaled = scaler.transform(embedding)
    
    # GP prediction
    probs = gp_model.predict_proba(embedding_scaled)[0]
    
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    # Uncertainty (entropy)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    uncertainty = entropy / np.log(len(CLASS_NAMES))
    
    # Top 3
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3 = {CLASS_NAMES[idx]: float(probs[idx]) for idx in top3_indices}
    
    return CLASS_NAMES[pred_idx], confidence, uncertainty, top3, probs

# Main app
def main():
    # Header
    st.title("🛰️ Few-Shot Satellite Image Classification")
    st.markdown("### Vision Transformer + Gaussian Process (Linear Kernel)")
    
    # Info box
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown("""
        **Problem**: Adapt a pre-trained Vision Transformer to satellite imagery with only 20 samples per class.
        
        **Solution**: 
        - Extract embeddings from fine-tuned ViT (frozen backbone)
        - Train Gaussian Process with **Linear kernel** (DotProduct)
        - Achieve ~83-85% accuracy vs 71.6% direct ViT
        
        **Key Innovation**: 
        - GP provides calibrated uncertainty estimates
        - Linear kernel works best for pretrained embeddings (RBF fails)
        - Frozen backbone superior to Deep Kernel Learning in few-shot regime
        
        **Classes**: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, 
        Pasture, Permanent Crop, Residential, River, Sea Lake
        """)
    
    # Load models
    with st.spinner("Loading models..."):
        vit_model, embed_model, processor, gp_model, scaler, device, metadata = load_models()
    
    # Sidebar
    st.sidebar.header("📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a satellite image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a 224x224 RGB satellite image from EuroSAT"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **Performance Summary**:
    - Direct ViT: 71.59% accuracy
    - ViT + GP (Linear): **{metadata.get('test_accuracy', 0.835)*100:.2f}% accuracy** ✅
    - Training: {metadata.get('num_train_samples', 200)} samples (20-shot)
    - Kernel: {metadata.get('kernel', 'DotProduct (Linear)')}
    - Test samples: {metadata.get('num_test_samples', 2700)}
    """)
    
    # Main content
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("Classifying..."):
                # Get predictions
                vit_pred, vit_conf, vit_top3, vit_probs = predict_vit(
                    image, vit_model, processor, device
                )
                gp_pred, gp_conf, gp_uncertainty, gp_top3, gp_probs = predict_gp(
                    image, embed_model, processor, gp_model, scaler, device
                )
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["🤖 Direct ViT", "🎯 ViT + GP (Proposed)", "🔄 Comparison"])
            
            with tab1:
                st.markdown(f"### Prediction: **{vit_pred}**")
                st.markdown(f"**Confidence:** {vit_conf:.2%}")
                
                st.markdown("#### Top-3 Predictions:")
                for cls, prob in vit_top3.items():
                    st.progress(prob, text=f"{cls}: {prob:.2%}")
            
            with tab2:
                st.markdown(f"### Prediction: **{gp_pred}**")
                st.markdown(f"**Confidence:** {gp_conf:.2%}")
                st.markdown(f"**Uncertainty:** {gp_uncertainty:.2%}")
                
                # Uncertainty interpretation
                if gp_uncertainty > 0.7:
                    st.error("⚠️ **HIGH UNCERTAINTY** - Manual review recommended")
                elif gp_uncertainty > 0.4:
                    st.warning("⚡ **MODERATE UNCERTAINTY** - Borderline case")
                else:
                    st.success("✅ **LOW UNCERTAINTY** - Model is confident")
                
                st.markdown("#### Top-3 Predictions:")
                for cls, prob in gp_top3.items():
                    st.progress(prob, text=f"{cls}: {prob:.2%}")
            
            with tab3:
                # Agreement
                agree = vit_pred == gp_pred
                if agree:
                    st.success(f"✅ **Models AGREE** on {vit_pred}")
                else:
                    st.error(f"❌ **Models DISAGREE**")
                    st.markdown(f"- ViT predicts: **{vit_pred}**")
                    st.markdown(f"- GP predicts: **{gp_pred}**")
                
                # Metrics
                improvement = (gp_conf - vit_conf) * 100
                st.metric(
                    "Confidence Change",
                    f"{improvement:+.2f}%",
                    delta=f"{improvement:+.2f}%"
                )
                
                # Chart comparison
                st.markdown("#### Prediction Probabilities")
                import pandas as pd
                
                comparison_df = pd.DataFrame({
                    'Class': CLASS_NAMES,
                    'ViT': vit_probs,
                    'GP': gp_probs
                })
                
                st.bar_chart(comparison_df.set_index('Class'))
                
                st.markdown("---")
                st.markdown("""
                **Why GP (Linear Kernel) is Better:**
                1. **Higher Accuracy**: ~83-85% vs 71.6% (+12-14%)
                2. **Uncertainty Estimates**: Flags unreliable predictions
                3. **Few-Shot Learning**: Works with only 20 samples/class
                4. **Linear Kernel**: Optimal for pretrained embeddings (RBF fails)
                """)
    
    else:
        st.info("👈 Upload a satellite image to begin classification")
        
        # Show example
        st.markdown("### 📚 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Frozen Backbone**
            
            Fine-tuned ViT extracts 768-dim embeddings
            - Preserves pretrained knowledge
            - Avoids overfitting
            - Fast inference
            """)
        
        with col2:
            st.markdown("""
            **2. GP Head (Linear Kernel)**
            
            GP on scaled embeddings
            - DotProduct (Linear) kernel
            - Non-parametric learning
            - Uncertainty quantification
            - 200 samples → 83-85% accuracy
            """)
        
        with col3:
            st.markdown("""
            **3. Uncertainty-Aware**
            
            Flags unreliable predictions
            - High uncertainty → Human review
            - Low uncertainty → Trust model
            - Critical for remote sensing
            """)
        
        st.markdown("---")
        st.markdown("### 🔑 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ What Worked:**
            - Linear kernel GP: 83-85% accuracy
            - Frozen backbone approach
            - StandardScaler normalization
            - 20-shot learning regime
            """)
        
        with col2:
            st.markdown("""
            **❌ What Failed:**
            - RBF kernel: 11% (matrix issues)
            - Deep Kernel Learning: 4-14%
            - Direct ViT: 71.6% (overfitting)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎓 IE643 Project | Few-Shot Satellite Image Classification</p>
        <p><strong>Key Finding</strong>: Frozen backbone + Linear GP outperforms Deep Kernel Learning</p>
        <p><em>Linear kernel optimal for pretrained embeddings (RBF causes numerical instability)</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
