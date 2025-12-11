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
            classifier = pipeline("image-classification", model="Marqo/nsfw-image-detection-384")
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
        "model": "Marqo NSFW Detection"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/detect/urls")
async def scan(request: URLRequest):
    logger.info(f"Scanning {len(request.urls)} images")
    
    # Load model
    try:
        model = load_model()
    except:
        return [{"url": u, "is_nsfw": False, "confidence_percentage": 0} for u in request.urls]
    
    results = []
    
    for url in request.urls:
        try:
            # Download
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as r:
                    if r.status != 200:
                        results.append({"url": url, "is_nsfw": False, "confidence_percentage": 0})
                        continue
                    data = await r.read()
            
            # Open image
            img = Image.open(BytesIO(data))
            
            # Predict
            preds = model(img)
            
            # Find NSFW score
            nsfw_score = 0
            for p in preds:
                if p['label'].lower() == 'nsfw':
                    nsfw_score = p['score'] * 100
            
            is_nsfw = nsfw_score > 50
            
            logger.info(f"Result: {'🚨 NSFW' if is_nsfw else '✅ Safe'} - {nsfw_score:.1f}%")
            
            results.append({
                "url": url,
                "is_nsfw": is_nsfw,
                "confidence_percentage": round(nsfw_score, 2),
                "detections": [{"class": "NSFW", "confidence": nsfw_score}] if is_nsfw else []
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({"url": url, "is_nsfw": False, "confidence_percentage": 0})
    
    return results

if __name__ == "__main__":
    logger.info("🚀 Starting API on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
