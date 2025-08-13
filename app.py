import os
import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports to reduce startup time
@st.cache_resource
def load_dependencies():
    """Load heavy dependencies only when needed"""
    try:
        from ultralytics import YOLO
        from fpdf import FPDF
        logger.info("Heavy dependencies loaded successfully")
        return YOLO, FPDF
    except Exception as e:
        logger.error(f"Error loading dependencies: {e}")
        st.error(f"Error loading ML dependencies: {e}")
        return None, None

# HuggingFace integration (optional)
def setup_huggingface():
    """Setup HuggingFace integration if token is available"""
    try:
        from huggingface_hub import HfApi, hf_hub_download, login
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(hf_token)
            return True, os.getenv("HF_REPO", "foduucom/stockmarket-pattern-detection-yolov8")
        return False, None
    except Exception as e:
        logger.warning(f"HuggingFace setup failed: {e}")
        return False, None

# Load pre-trained models with better error handling
@st.cache_resource
def load_models():
    """Load YOLO model with fallbacks"""
    YOLO, _ = load_dependencies()
    if YOLO is None:
        return None, {}
    
    try:
        # Try HuggingFace first
        hf_available, hf_repo = setup_huggingface()
        if hf_available and hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=hf_repo, filename="best.pt")
                model = YOLO(model_path)
                logger.info(f"Loaded model from HuggingFace: {hf_repo}")
            except Exception as e:
                logger.warning(f"HF model loading failed: {e}, using default")
                model = YOLO('yolov8n.pt')  # Smallest model
        else:
            model = YOLO('yolov8n.pt')  # Default to nano model for free tier
            logger.info("Loaded default YOLOv8n model")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        st.error("Failed to load YOLO model")
        return None, {}
    
    # Pattern mapping
    pattern_map = {
        0: "Head and Shoulders",
        1: "Double Top", 
        2: "Double Bottom",
        3: "Triangle",
        4: "Flag",
        5: "Cup and Handle",
        6: "Wedge",
        7: "Channel"
    }
    return model, pattern_map

# Optimized analysis function
def analyze_chart(image, model, pattern_map, strategy="conservative"):
    """Analyze chart with memory optimization"""
    if model is None:
        return [], image
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            # Resize large images to save memory
            max_size = 640
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            image_array = np.array(image)
        else:
            image_array = image
        
        # Run inference
        results = model(image_array, verbose=False)  # Suppress verbose output
        
        predictions = []
        annotated_image = image_array.copy()
        
        if len(results[0].boxes) > 0:
            confidence_threshold = 0.3 if strategy == "aggressive" else 0.5
            
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > confidence_threshold:
                        pattern_name = pattern_map.get(class_id, "Unknown Pattern")
                        predictions.append({
                            "pattern": pattern_name,
                            "confidence": confidence,
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })
                        
                        # Draw annotations
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f"{pattern_name}: {confidence:.2f}", 
                                   (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    logger.warning(f"Error processing detection: {e}")
                    continue
        
        return predictions, annotated_image
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        st.error(f"Analysis failed: {e}")
        return [], image

# Simple PDF generation
def generate_pdf_report(predictions):
    """Generate simple text report instead of PDF for memory efficiency"""
    report = "Stock Chart Analysis Report\n"
    report += "=" * 30 + "\n\n"
    
    if predictions:
        report += "Detected Patterns:\n\n"
        for i, pred in enumerate(predictions, 1):
            report += f"{i}. {pred['pattern']}\n"
            report += f"   Confidence: {pred['confidence']:.2f}\n"
            report += f"   Location: {pred['bbox']}\n\n"
    else:
        report += "No patterns detected\n"
    
    return report

# Main application
def main():
    # Streamlit configuration
    st.set_page_config(
        page_title="AI Stock Chart Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 AI Stock Chart Analysis Tool")
    st.markdown("Upload a stock chart to detect technical patterns using YOLOv8")
    st.markdown("---")
    
    # Health check endpoint for Render
    if st.sidebar.button("🔍 Health Check"):
        st.sidebar.success("Application is running!")
        logger.info("Health check requested")
    
    # Load models with UI feedback
    model_status = st.empty()
    model_status.info("🔄 Loading AI models... Please wait.")
    
    try:
        model, pattern_map = load_models()
        
        if model is None:
            model_status.error("⚠️ Could not load the AI model. Please try again later.")
            st.info("This might be due to memory constraints on the free tier.")
            st.stop()
        else:
            model_status.success("✅ AI models loaded successfully!")
            
    except Exception as e:
        model_status.error(f"❌ Error loading models: {e}")
        st.stop()
    
    # Settings
    st.sidebar.header("Settings")
    strategy = st.sidebar.selectbox(
        "Analysis Strategy",
        ["conservative", "aggressive"],
        help="Conservative: Higher confidence, Aggressive: Lower confidence"
    )
    
    # Force UI elements to show
    st.markdown("### 📊 Upload Your Chart")
    st.write("Choose a stock chart image to analyze for technical patterns.")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload a stock chart screenshot (max 5MB)"
        )
        
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
                st.error("File too large. Please upload an image smaller than 5MB.")
            else:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Chart", use_column_width=True)
                    
                    if st.button("🔍 Analyze Chart", type="primary", use_container_width=True):
                        with st.spinner("Analyzing patterns..."):
                            predictions, annotated_image = analyze_chart(image, model, pattern_map, strategy)
                        
                        # Store in session state
                        st.session_state.predictions = predictions
                        st.session_state.annotated_image = annotated_image
                        st.session_state.original_image = image
                        st.rerun()  # Force refresh to show results
                        
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        else:
            st.info("👆 Please upload a chart image to get started")
    with col2:
        st.markdown("### 📈 Analysis Results")
        
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            st.success(f"✅ Found {len(st.session_state.predictions)} pattern(s)")
            
            # Show results
            for i, pred in enumerate(st.session_state.predictions, 1):
                with st.expander(f"Pattern {i}: {pred['pattern']}", expanded=True):
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
                    st.write(f"**Location:** {pred['bbox']}")
            
            # Show annotated image
            if hasattr(st.session_state, 'annotated_image'):
                st.image(st.session_state.annotated_image, caption="Detected Patterns", use_column_width=True)
            
            # Text report download
            if st.button("📄 Generate Report", use_container_width=True):
                report = generate_pdf_report(st.session_state.predictions)
                st.download_button(
                    label="⬇️ Download Text Report",
                    data=report,
                    file_name="chart_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        elif hasattr(st.session_state, 'predictions') and len(st.session_state.predictions) == 0:
            st.info("ℹ️ No patterns detected. Try adjusting the strategy or uploading a different chart.")
        else:
            st.info("📤 Upload and analyze a chart to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use clear, high-contrast chart images for best results")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("Application encountered an error. Please refresh the page.")
