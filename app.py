import os
import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fpdf import FPDF
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login

# Login to HF (using env var for security)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
else:
    st.warning("HF_TOKEN not set. Persistence disabled.")

hf_repo = os.getenv("HF_REPO", "foduucom/stockmarket-pattern-detection-yolov8")  # Fallback to original

# Load pre-trained models (now from HF Hub)
@st.cache_resource
def load_models():
    try:
        # Download latest weights from HF repo
        model_path = hf_hub_download(repo_id=hf_repo, filename="best.pt")  # Assume fine-tuned saves as best.pt
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model from HF Hub: {e}")
        # Fallback to default YOLOv8 model
        model = YOLO('yolov8n.pt')
    
    # Pattern mapping for stock chart patterns
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

# Analyze chart function
def analyze_chart(image, model, pattern_map, strategy="conservative"):
    # Convert PIL image to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run YOLO inference
    results = model(image)
    
    predictions = []
    annotated_image = image.copy()
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Apply strategy filtering
            confidence_threshold = 0.3 if strategy == "aggressive" else 0.5
            
            if confidence > confidence_threshold:
                pattern_name = pattern_map.get(class_id, "Unknown Pattern")
                predictions.append({
                    "pattern": pattern_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
                
                # Draw bounding box on image
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{pattern_name}: {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return predictions, annotated_image

# Self-Finetuning System (updated for HF push)
def finetune_model(uploaded_file, model):
    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract dataset
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Look for data.yaml file
            data_yaml_path = None
            for root, dirs, files in os.walk(tmpdir):
                if "data.yaml" in files:
                    data_yaml_path = os.path.join(root, "data.yaml")
                    break
            
            if not data_yaml_path:
                st.error("No data.yaml file found in the uploaded dataset")
                return
            
            # Fine-tune
            try:
                results = model.train(data=data_yaml_path, epochs=5, imgsz=640)
                
                # Save best weights
                best_model_path = "best.pt"
                model.save(best_model_path)
                
                # Push to HF Hub
                if hf_token and hf_repo:
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj=best_model_path,
                        path_in_repo="best.pt",
                        repo_id=hf_repo,
                        repo_type="model",
                        commit_message="Upload fine-tuned YOLOv8 weights"
                    )
                    st.success("Model fine-tuned and pushed to HF Hub! Reload to use updated version.")
                else:
                    st.warning("HF_TOKEN or HF_REPO not set. Fine-tuning not persisted.")
            except Exception as e:
                st.error(f"Error during fine-tuning: {e}")

# Generate PDF report
def generate_pdf_report(predictions, image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Stock Chart Analysis Report", ln=1, align="C")
    pdf.ln(10)
    
    if predictions:
        pdf.cell(200, 10, txt="Detected Patterns:", ln=1)
        pdf.ln(5)
        
        for i, pred in enumerate(predictions, 1):
            pdf.cell(200, 10, txt=f"{i}. {pred['pattern']} (Confidence: {pred['confidence']:.2f})", ln=1)
    else:
        pdf.cell(200, 10, txt="No patterns detected", ln=1)
    
    return pdf.output(dest='S').encode('latin1')

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Stock Chart Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Stock Chart Analysis Tool")
    st.markdown("Upload a stock chart screenshot to detect technical patterns using YOLOv8")
    
    # Load models
    model, pattern_map = load_models()
    
    # Sidebar
    st.sidebar.header("Settings")
    strategy = st.sidebar.selectbox(
        "Analysis Strategy",
        ["conservative", "aggressive"],
        help="Conservative: Higher confidence threshold, Aggressive: Lower confidence threshold"
    )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Upload Chart")
        uploaded_file = st.file_uploader(
            "Choose a chart image...",
            type=["png", "jpg", "jpeg"],
            help="Upload a stock chart screenshot"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chart", use_column_width=True)
            
            if st.button("🔍 Analyze Chart", type="primary"):
                with st.spinner("Analyzing chart patterns..."):
                    predictions, annotated_image = analyze_chart(image, model, pattern_map, strategy)
                
                # Store results in session state
                st.session_state.predictions = predictions
                st.session_state.annotated_image = annotated_image
    
    with col2:
        st.header("📈 Analysis Results")
        
        if hasattr(st.session_state, 'predictions'):
            if st.session_state.predictions:
                st.success(f"Found {len(st.session_state.predictions)} pattern(s)")
                
                # Display predictions
                for i, pred in enumerate(st.session_state.predictions, 1):
                    with st.expander(f"Pattern {i}: {pred['pattern']}", expanded=True):
                        st.write(f"**Confidence:** {pred['confidence']:.2f}")
                        st.write(f"**Location:** {pred['bbox']}")
                
                # Show annotated image
                st.image(st.session_state.annotated_image, caption="Detected Patterns", use_column_width=True)
                
                # PDF download
                if st.button("📄 Generate PDF Report"):
                    pdf_data = generate_pdf_report(st.session_state.predictions, image)
                    st.download_button(
                        label="⬇️ Download Report",
                        data=pdf_data,
                        file_name="chart_analysis_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("No patterns detected in the uploaded chart")
    
    # Fine-tuning section
    st.header("🔧 Model Fine-tuning")
    st.markdown("Upload your own labeled dataset to improve the model")
    
    with st.expander("Upload Training Dataset"):
        st.markdown("""
        **Dataset Format:**
        - Upload a ZIP file containing:
          - `data.yaml` configuration file
          - `images/` folder with training images
          - `labels/` folder with YOLO format annotations
        """)
        
        dataset_file = st.file_uploader(
            "Choose dataset ZIP file...",
            type=["zip"],
            help="Upload a YOLO format dataset"
        )
        
        if dataset_file is not None:
            if st.button("🚀 Start Fine-tuning"):
                with st.spinner("Fine-tuning model... This may take several minutes."):
                    finetune_model(dataset_file, model)

if __name__ == "__main__":
    main()
