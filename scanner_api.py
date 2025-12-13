import os
import warnings

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import aiohttp
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFW Scanner")

# Global model
classifier = None

def load_model():
    """Load Hugging Face model"""
    global classifier
    if classifier is None:
        logger.info("Loading model...")
        try:
            from transformers import pipeline
            # Using LukeJacob2023/nsfw-image-detection for granular classes:
            # drawings, hentai, neutral, porn, sexy
            classifier = pipeline("image-classification", model="LukeJacob2023/nsfw-image-detector")
            logger.info("✅ Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return classifier

class URLRequest(BaseModel):
    urls: List[str]

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "LukeJacob2023/nsfw-image-detector"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/detect/urls")
async def scan(request: URLRequest):
    logger.info(f"Scanning {len(request.urls)} images")
    
    # Load model if not loaded
    try:
        model = load_model()
    except:
        return [{"url": u, "is_nsfw": False, "confidence_percentage": 0} for u in request.urls]
    
    results = []
    
    for url in request.urls:
        try:
            # Download
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as r:
                    if r.status != 200:
                        results.append({"url": url, "is_nsfw": False, "confidence_percentage": 0})
                        continue
                    data = await r.read()
            
            # Open image
            img = Image.open(BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # CRITICAL OPTIMIZATION: Resize to 224x224 (Model's native input)
            img = img.resize((224, 224))
            
            # Predict
            preds = model(img)
            
            # Calculate NSFW score based on specific classes
            # Classes: drawings, hentai, neutral, porn, sexy
            # We only want to flag 'porn' and 'hentai'. 'sexy' is usually bikini/lingerie.
            nsfw_score = 0
            detailed_detections = []
            
            for p in preds:
                label = p['label'].lower()
                score_pct = p['score'] * 100
                
                if label in ['porn', 'hentai']:
                    nsfw_score += score_pct
                
                detailed_detections.append({
                    "class": label,
                    "confidence": score_pct
                })
            
            # Default threshold is 5% in API, but server config usually overrides this
            is_nsfw = nsfw_score > 5
            
            logger.info(f"Result: {'🚨 NSFW' if is_nsfw else '✅ Safe'} - {nsfw_score:.1f}%")
            
            results.append({
                "url": url,
                "is_nsfw": is_nsfw,
                "confidence_percentage": round(nsfw_score, 2),
                "detections": detailed_detections if is_nsfw else []
            })
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            results.append({"url": url, "is_nsfw": False, "confidence_percentage": 0})
    
    return results

if __name__ == "__main__":
    # Pre-load model on startup to prevent first-request lag
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")

    logger.info("🚀 Starting API on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
