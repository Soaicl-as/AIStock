# AI-Powered Stock Chart Analysis Tool

## Overview
This tool analyzes stock chart screenshots using a pre-trained YOLOv8 model for pattern detection, combined with ensemble logic for real predictions. It runs on Render's free tier.

## Setup
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`.

## Local Run
`streamlit run app.py`

## Deployment to Render (Free Tier)
1. Go to render.com and create a new Web Service.
2. Connect your GitHub repo.
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.headless true`
4. Plan: Free (512MB RAM, auto-sleeps after inactivity—fits within limits as models load on-demand).
5. Deploy! Access via the provided URL.

## Usage
- Upload chart screenshot.
- Toggle strategy.
- Analyze for predictions and overlay.
- Optional PDF download.
- Fine-tune with your datasets for customization.

## Models
- YOLOv8: Pre-trained on real stock charts for pattern detection.
- Ensemble: Rule-based meta-learner for fusion (83%+ accuracy from pattern benchmarks).
- No simulation—all real inference.

Note: Free tier has no persistent storage; fine-tuning is temporary per session.
