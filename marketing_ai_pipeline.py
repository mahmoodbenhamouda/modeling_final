import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    TrainingArguments,
    Trainer
)
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from tqdm import tqdm
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {str(e)}")

# Define model storage paths
MODEL_STORAGE_PATH = "./models"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

class EmotionClassifier:
    """Emotion classification model for marketing text"""
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        
    def initialize(self):
        """Initialize the emotion classification model"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base",
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "emotion"),
                num_labels=len(self.emotions)
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base",
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "emotion")
            )
            
            logger.info("Emotion classification model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize emotion classification model: {str(e)}")
            # Fallback to using pipeline
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Fallback emotion pipeline initialized successfully")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback emotion pipeline: {str(e2)}")
    
    def classify(self, text: str) -> Dict[str, float]:
        """Classify emotion in text"""
        if self.model is None or self.tokenizer is None:
            # Use fallback if available
            if hasattr(self, 'emotion_pipeline'):
                results = self.emotion_pipeline(text)
                return {results[0]['label']: results[0]['score']}
            return {"neutral": 1.0}  # Default fallback
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_dict = {self.emotions[i]: float(probs[0][i]) for i in range(len(self.emotions))}
        return probs_dict

class NamedEntityRecognizer:
    """Named Entity Recognition for marketing content"""
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.nlp = None
        
    def initialize(self):
        """Initialize the NER model"""
        try:
            # Load SpaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("NER model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NER model: {str(e)}")
            # Try to download the model
            try:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("NER model downloaded and initialized successfully")
            except Exception as e2:
                logger.error(f"Failed to download and initialize NER model: {str(e2)}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if self.nlp is None:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def get_entity_summary(self, text: str) -> Dict: 
        """Get summary of entities for marketing analysis""" 
        entities = self.extract_entities(text) 
        
        # Count entity types 
        entity_counts = {key: len(values) for key, values in entities.items()} 
        
        # Marketing-focused entity categories 
        marketing_categories = { 
            "PRODUCT": [],  # Product mentions  
            "BRAND": [],    # Brand mentions 
            "AUDIENCE": [], # Target audience mentions 
            "BENEFIT": [],  # Benefit mentions 
        } 
        
        # Map SpaCy entities to marketing categories 
        if "PRODUCT" in entities or "ORG" in entities: 
            marketing_categories["PRODUCT"] = entities.get("PRODUCT", []) + entities.get("ORG", []) 
        if "ORG" in entities: 
            marketing_categories["BRAND"] = entities.get("ORG", []) 
        if "PERSON" in entities or "NORP" in entities: 
            marketing_categories["AUDIENCE"] = entities.get("PERSON", []) + entities.get("NORP", []) 
            
        return {
            "detailed_entities": entities,
            "entity_counts": entity_counts,
            "marketing_categories": marketing_categories
        }

class MarketingAIPipeline:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.text_model = None
        self.image_model = None
        self.sentiment_model = None
        self.trend_model = None
        self.tokenizer = None
        self.emotion_classifier = EmotionClassifier(device)
        self.ner = NamedEntityRecognizer(device)
        
    def initialize_models(self):
        """Initialize all models in the pipeline"""
        logger.info("Initializing models...")
        
        # Initialize text generation model (GPT-2 for fine-tuning)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "gpt2")
            ) 
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "gpt2")
            ).to(self.device)
            logger.info("Text generation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text generation model: {str(e)}")
        
        # Initialize sentiment analysis model
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == "cuda" else -1,
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "sentiment")
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment analysis model: {str(e)}")
            # Fallback with a simpler model
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Fallback sentiment model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback sentiment model: {str(e2)}")
        
        # Initialize image generation model (using Stable Diffusion)
        try:
            model_id = "stabilityai/stable-diffusion-2-1-base"
            self.image_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=os.path.join(MODEL_STORAGE_PATH, "stable_diffusion")
            )
            self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(
                self.image_model.scheduler.config
            )
            self.image_model = self.image_model.to(self.device)
            logger.info("Image generation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image generation model: {str(e)}")
        
        # Initialize trend prediction model
        self.trend_model = self._create_trend_model()
        
        # Save trend model path
        self.trend_model_path = os.path.join(MODEL_STORAGE_PATH, "trend_model.pt")
        
        # Initialize emotion classifier
        self.emotion_classifier.initialize()
        
        # Initialize NER
        self.ner.initialize()
        
    def _create_trend_model(self) -> nn.Module:
        """Create a more sophisticated trend prediction model with LSTM layers"""
        model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # For probability output
        ).to(self.device)
        return model
    
    def save_models(self):
        """Save all models to disk"""
        logger.info("Saving models...")
        if self.trend_model is not None:
            torch.save(self.trend_model.state_dict(), self.trend_model_path)
            logger.info(f"Trend model saved to {self.trend_model_path}")
    
    def load_models(self):
        """Load models from disk"""
        logger.info("Loading models...")
        if os.path.exists(self.trend_model_path):
            self.trend_model.load_state_dict(torch.load(self.trend_model_path))
            logger.info(f"Trend model loaded from {self.trend_model_path}")
    
    def fine_tune_text_model(self, training_data: List[str], epochs: int = 3):
        """Fine-tune the text generation model on custom data"""
        logger.info("Fine-tuning text model...")
        
        # Prepare dataset
        encoded_data = []
        for text in tqdm(training_data, desc="Encoding training data"):
            # Add special tokens for marketing context
            marketing_text = f"<marketing>{text}</marketing>"
            encoded = self.tokenizer(marketing_text, truncation=True, max_length=512, padding="max_length")
            encoded_data.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"].copy()
            })
        
        class MarketingDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
                
            def __len__(self):
                return len(self.encodings)
                
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.encodings[idx]["input_ids"]), 
                    "attention_mask": torch.tensor(self.encodings[idx]["attention_mask"]), 
                    "labels": torch.tensor(self.encodings[idx]["labels"])
                }
        
        dataset = MarketingDataset(encoded_data)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_STORAGE_PATH, "marketing_text_model"),
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_dir=os.path.join(MODEL_STORAGE_PATH, "logs"),
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.text_model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Fine-tune
        trainer.train()
        
        # Save fine-tuned model
        trainer.save_model(os.path.join(MODEL_STORAGE_PATH, "marketing_text_model"))
        logger.info("Text model fine-tuning completed and saved")
    
    def generate_marketing_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate marketing text based on input prompt"""
        if self.text_model is None or self.tokenizer is None:
            return "Model not loaded. Please initialize models first."
        
        marketing_prompt = f"Create a marketing content for: {prompt}\n\n"
        
        inputs = self.tokenizer(marketing_prompt, return_tensors="pt").to(self.device)
        
        try:
            outputs = self.text_model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.92,
                do_sample=True,
                no_repeat_ngram_size=2,
                top_k=50
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            if generated_text.startswith(marketing_prompt):
                generated_text = generated_text[len(marketing_prompt):]
                
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating marketing text: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def generate_marketing_image(self, prompt: str) -> Image.Image:
        """Generate marketing image based on text prompt"""
        if self.image_model is None:
            dummy_image = Image.new('RGB', (512, 512), color=(200, 200, 200))
            return dummy_image
            
        marketing_prompt = f"professional marketing image of {prompt}, high quality, photorealistic, detailed, commercial photography style"
        
        try:
            with torch.autocast(device_type='cuda' if self.device == "cuda" else 'cpu'):
                image = self.image_model(
                    marketing_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    negative_prompt="low quality, blurry, distorted, ugly, bad proportions, watermark, text, logo"
                ).images[0]
                
            return image
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}")
            # Return a blank image in case of failure
            dummy_image = Image.new('RGB', (512, 512), color=(200, 200, 200))
            return dummy_image
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if self.sentiment_model is None:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            return self.sentiment_model(text)[0]
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "ERROR", "score": 0.0}
    
    def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotion in text"""
        return self.emotion_classifier.classify(text)
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        return self.ner.get_entity_summary(text)
    
    def predict_trends(self, historical_data: np.ndarray) -> float:
        """Predict future trends based on historical data"""
        if self.trend_model is None:
            return 0.5  # Default neutral prediction
            
        self.trend_model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(historical_data).to(self.device)
            prediction = self.trend_model(input_tensor)
        return prediction.item()
    
    def train_trend_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100):
        """Train the trend prediction model with more comprehensive training"""
        logger.info("Training trend prediction model...")
        
        criterion = nn.BCELoss() if self.trend_model[-1].__class__.__name__ == "Sigmoid" else nn.MSELoss()
        optimizer = optim.Adam(self.trend_model.parameters(), lr=0.001)
        
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Training loop with progress tracking
        losses = []
        for epoch in range(epochs):
            self.trend_model.train()
            optimizer.zero_grad()
            
            outputs = self.trend_model(X_train)
            loss = criterion(outputs, y_train.view(-1, 1))
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Save the trained model
        self.save_models()
        
        # Return loss history for analysis
        return losses
    
    def comprehensive_marketing_analysis(self, text: str) -> Dict:
        """Perform a comprehensive analysis of marketing text"""
        results = {}
        
        # Basic text stats
        words = word_tokenize(text)
        results["text_length"] = len(text)
        results["word_count"] = len(words)
        
        # Sentiment analysis
        results["sentiment"] = self.analyze_sentiment(text)
        
        # Emotion analysis
        results["emotion"] = self.analyze_emotion(text)
        
        # Named entity recognition
        results["entities"] = self.extract_entities(text)
        
        return results
    
    def create_marketing_visualization(self, analysis_results: Dict) -> Optional[str]:
        """Create visualization of marketing analysis results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create a 2x2 subplot grid
            plt.subplot(2, 2, 1)
            
            # Plot emotion distribution if available
            if "emotion" in analysis_results:
                emotions = analysis_results["emotion"]
                plt.bar(emotions.keys(), emotions.values())
                plt.title("Emotion Distribution")
                plt.xticks(rotation=45)
                
            # Plot entity counts if available
            plt.subplot(2, 2, 2)
            if "entities" in analysis_results and "entity_counts" in analysis_results["entities"]:
                entity_counts = analysis_results["entities"]["entity_counts"]
                if entity_counts:
                    plt.bar(entity_counts.keys(), entity_counts.values())
                    plt.title("Named Entity Types")
                    plt.xticks(rotation=45)
                
            # Plot sentiment score if available
            plt.subplot(2, 2, 3)
            if "sentiment" in analysis_results:
                sentiment = analysis_results["sentiment"]
                plt.bar(["Sentiment Score"], [sentiment["score"]])
                plt.title(f"Sentiment: {sentiment['label']}")
                plt.ylim(0, 1)
                
            # Text stats
            plt.subplot(2, 2, 4)
            plt.bar(["Text Length", "Word Count"], 
                   [analysis_results.get("text_length", 0), 
                    analysis_results.get("word_count", 0)])
            plt.title("Text Statistics")
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def generate_complete_marketing_package(self, prompt: str) -> Dict:
        """Generate a complete marketing package including text, image, and analysis"""
        results = {}
        
        # Generate marketing text
        results["text"] = self.generate_marketing_text(prompt)
        
        # Generate marketing image
        image = self.generate_marketing_image(prompt)
        
        # Save image to buffer
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        results["image_base64"] = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        # Analyze generated text
        results["analysis"] = self.comprehensive_marketing_analysis(results["text"])
        
        # Create visualization
        results["visualization"] = self.create_marketing_visualization(results["analysis"])
        
        return results

def main():
    # Initialize the pipeline
    pipeline = MarketingAIPipeline()
    pipeline.initialize_models()
    
    # Example usage
    prompt = "Create an engaging social media post for a new eco-friendly product"
    
    # Generate marketing text
    generated_text = pipeline.generate_marketing_text(prompt)
    print(f"Generated Text: {generated_text}")
    
    # Generate marketing image
    generated_image = pipeline.generate_marketing_image(prompt)
    generated_image.save("generated_marketing_image.png")
    
    # Analyze sentiment
    sentiment = pipeline.analyze_sentiment(generated_text)
    print(f"Sentiment Analysis: {sentiment}")
    
    # Analyze emotion
    emotion = pipeline.analyze_emotion(generated_text)
    print(f"Emotion Analysis: {emotion}")
    
    # Extract entities
    entities = pipeline.extract_entities(generated_text)
    print(f"Named Entities: {entities}")
    
    # Example trend prediction (you'll need to provide actual historical data)
    historical_data = np.random.randn(100)  # Replace with actual historical data
    trend_prediction = pipeline.predict_trends(historical_data)
    print(f"Trend Prediction: {trend_prediction}")

if __name__ == "__main__":
    main() 