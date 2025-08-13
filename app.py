import os
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login

# ... (rest of imports unchanged)

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
    # Download latest weights from HF repo
    model_path = hf_hub_download(repo_id=hf_repo, filename="best.pt")  # Assume fine-tuned saves as best.pt
    model = YOLO(model_path)
    # ... (pattern_map unchanged)
    return model, pattern_map

# ... (analyze_chart unchanged)

# Self-Finetuning System (updated for HF push)
def finetune_model(uploaded_file):
    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            os.system(f"unzip {zip_path} -d {tmpdir}")
            # Fine-tune
            results = model.train(data=os.path.join(tmpdir, "data.yaml"), epochs=5, imgsz=640)
            # Save best weights
            model.save("best.pt")  # Or use results.best for path
            # Push to HF Hub
            if hf_token and hf_repo:
                api = HfApi()
                api.upload_file(
                    path_or_fileobj="best.pt",
                    path_in_repo="best.pt",
                    repo_id=hf_repo,
                    repo_type="model",
                    commit_message="Upload fine-tuned YOLOv8 weights"
                )
                st.success("Model fine-tuned and pushed to HF Hub! Reload to use updated version.")
            else:
                st.warning("HF_TOKEN or HF_REPO not set. Fine-tuning not persisted.")

# ... (rest of UI unchanged)
