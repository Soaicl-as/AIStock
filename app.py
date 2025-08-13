import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple caching for models
@st.cache_resource
def load_yolo_model():
    """Load YOLO model with simple fallback"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Smallest model
        logger.info("YOLOv8n model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO: {e}")
        return None

def get_pattern_map():
    """Get pattern mapping"""
    return {
        0: "Head and Shoulders",
        1: "Double Top", 
        2: "Double Bottom",
        3: "Triangle",
        4: "Flag",
        5: "Cup and Handle",
        6: "Wedge",
        7: "Channel"
    }

def analyze_chart_simple(image, model, pattern_map, confidence_threshold=0.5):
    """Simple chart analysis"""
    if model is None:
        return [], np.array(image)
    
    try:
        # Resize if too large
        if isinstance(image, Image.Image):
            if image.width > 640 or image.height > 640:
                image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            img_array = image
        
        # Run detection
        results = model(img_array, verbose=False)
        
        predictions = []
        annotated = img_array.copy()
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                try:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    if conf > confidence_threshold:
                        x1, y1, x2, y2 = coords
                        pattern_name = pattern_map.get(cls_id, "Unknown")
                        
                        predictions.append({
                            "pattern": pattern_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2)
                        })
                        
                        # Draw box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"{pattern_name}: {conf:.2f}", 
                                   (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                except:
                    continue
        
        return predictions, annotated
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return [], np.array(image)

def main():
    # Page config
    st.set_page_config(
        page_title="Stock Chart Analyzer",
        page_icon="📈",
        layout="centered"
    )
    
    # Title
    st.title("📈 Stock Chart Analyzer")
    st.write("Upload a stock chart to detect technical patterns")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_yolo_model()
        pattern_map = get_pattern_map()
    
    if model is None:
        st.error("Failed to load model")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Settings
    st.subheader("Settings")
    strategy = st.selectbox(
        "Detection Strategy", 
        ["Conservative (0.5)", "Aggressive (0.3)"],
        help="Higher threshold = fewer but more confident detections"
    )
    threshold = 0.5 if "Conservative" in strategy else 0.3
    
    # File upload
    st.subheader("Upload Chart")
    uploaded_file = st.file_uploader(
        "Choose image", 
        type=["png", "jpg", "jpeg"],
        help="Max 5MB"
    )
    
    if uploaded_file is not None:
        # File size check
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File too large (max 5MB)")
            st.stop()
        
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chart", width=400)
            
            # Analyze button
            if st.button("🔍 Analyze Chart", type="primary"):
                with st.spinner("Analyzing..."):
                    predictions, annotated = analyze_chart_simple(image, model, pattern_map, threshold)
                
                # Results
                st.subheader("Results")
                
                if predictions:
                    st.success(f"Found {len(predictions)} patterns:")
                    
                    # Show predictions
                    for i, pred in enumerate(predictions, 1):
                        st.write(f"**{i}. {pred['pattern']}** - Confidence: {pred['confidence']:.1%}")
                    
                    # Show annotated image
                    st.image(annotated, caption="Detected Patterns", width=400)
                    
                    # Simple text report
                    report_text = "Stock Chart Analysis Report\n\n"
                    for i, pred in enumerate(predictions, 1):
                        report_text += f"{i}. {pred['pattern']} ({pred['confidence']:.1%})\n"
                    
                    st.download_button(
                        "📄 Download Report",
                        data=report_text,
                        file_name="analysis_report.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No patterns detected. Try a different image or lower threshold.")
                    
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a chart image to get started")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by YOLOv8")

if __name__ == "__main__":
    main()
