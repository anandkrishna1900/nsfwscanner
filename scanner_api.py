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

# Global models
classifier_nsfw = None
classifier_gore = None

def load_models():
    """Load Hugging Face models"""
    global classifier_nsfw, classifier_gore
    
    # 1. Load NSFW Model (Porn/Hentai vs Neutral/Sexy)
    if classifier_nsfw is None:
        logger.info("Loading NSFW model...")
        try:
            from transformers import pipeline
            classifier_nsfw = pipeline("image-classification", model="LukeJacob2023/nsfw-image-detector")
            logger.info("✅ NSFW Model loaded")
        except Exception as e:
            logger.error(f"Failed to load NSFW model: {e}")
            raise

    # 2. Load Gore/Violence Model
    if classifier_gore is None:
        logger.info("Loading Gore model...")
        try:
            from transformers import pipeline
            # Using jaranohaal/vit-base-violence-detection for violence/gore
            classifier_gore = pipeline("image-classification", model="jaranohaal/vit-base-violence-detection")
            logger.info("✅ Gore Model loaded")
        except Exception as e:
            logger.error(f"Failed to load Gore model: {e}")
            # Don't raise here, allow running with just NSFW if Gore fails? 
            # Better to fail so we know it's broken.
            raise
            
    return classifier_nsfw, classifier_gore

class URLRequest(BaseModel):
    urls: List[str]

@app.get("/")
def home():
    return {
        "status": "running",
        "models": [
            "LukeJacob2023/nsfw-image-detector",
            "jaranohaal/vit-base-violence-detection"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/detect/urls")
async def scan(request: URLRequest):
    logger.info(f"Scanning {len(request.urls)} images")
    
    # Load models if not loaded
    try:
        model_nsfw, model_gore = load_models()
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
            
            # --- PREDICTION 1: NSFW (Porn/Hentai) ---
            preds_nsfw = model_nsfw(img)
            
            nsfw_score = 0
            detailed_detections = []
            
            for p in preds_nsfw:
                label = p['label'].lower()
                score_pct = p['score'] * 100
                
                # Only aggregate Porn and Hentai. Ignore 'sexy', 'neutral', 'drawings'
                if label in ['porn', 'hentai']:
                    nsfw_score += score_pct
                
                detailed_detections.append({
                    "class": label,
                    "confidence": score_pct
                })

            # --- PREDICTION 2: GORE (Violence) ---
            preds_gore = model_gore(img)
            gore_score = 0
            
            for p in preds_gore:
                label = p['label'].lower()
                score_pct = p['score'] * 100
                
                # Labels usually 'violence' vs 'non_violence' or 'violent'
                if 'violence' in label or 'violent' in label:
                    if 'non' not in label: # Exclude 'non_violence'
                        gore_score = score_pct
                        detailed_detections.append({
                            "class": "gore",
                            "confidence": score_pct
                        })
            
            # --- FINAL DECISION ---
            # Max score determines the "confidence" we report
            final_score = max(nsfw_score, gore_score)
            
            # Default threshold check
            is_nsfw = final_score > 5
            
            status_emoji = "✅ Safe"
            if is_nsfw:
                if gore_score > nsfw_score:
                    status_emoji = "🩸 GORE"
                else:
                    status_emoji = "🔞 NSFW"
            
            logger.info(f"Result: {status_emoji} - Max Score: {final_score:.1f}%")
            
            results.append({
                "url": url,
                "is_nsfw": is_nsfw,
                "confidence_percentage": round(final_score, 2),
                "detections": detailed_detections if is_nsfw else []
            })
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            results.append({"url": url, "is_nsfw": False, "confidence_percentage": 0})
    
    return results

if __name__ == "__main__":
    # Pre-load model on startup to prevent first-request lag
    try:
        load_models()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")

    logger.info("🚀 Starting API on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
