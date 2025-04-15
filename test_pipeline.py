import pandas as pd
import numpy as np
from marketing_ai_pipeline import MarketingAIPipeline
import logging
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data():
    """Load and prepare training data from the datasets"""
    logger.info("Loading training data...")
    
    # Load reviews data for text generation and sentiment analysis
    reviews_df = pd.read_csv('data/Reviews.csv')
    reviews_texts = reviews_df['Text'].tolist()[:1000]  # Use first 1000 reviews for testing
    
    # Load social media advertising data for trend prediction
    social_df = pd.read_csv('data/Social_Media_Advertising.csv')
    
    # Load digital marketing campaigns data
    campaigns_df = pd.read_csv('data/digital_marketing_campaigns_smes.csv')
    
    return reviews_texts, social_df, campaigns_df

def test_text_generation(pipeline):
    """Test text generation functionality"""
    logger.info("\n=== Testing Text Generation ===")
    
    # Test prompts for different industries
    test_prompts = [
        "Create an engaging social media post for a new eco-friendly product",
        "Write a compelling advertisement for a tech startup's new app",
        "Generate a marketing message for a fashion brand's summer collection"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nGenerating text for prompt: {prompt}")
        generated_text = pipeline.generate_marketing_text(prompt)
        logger.info(f"Generated text: {generated_text}")
        
        # Test sentiment analysis on generated text
        sentiment = pipeline.analyze_sentiment(generated_text)
        logger.info(f"Sentiment analysis: {sentiment}")

def test_image_generation(pipeline):
    """Test image generation functionality"""
    logger.info("\n=== Testing Image Generation ===")
    
    # Create output directory if it doesn't exist
    os.makedirs("generated_images", exist_ok=True)
    
    # Test prompts for different types of marketing images
    test_prompts = [
        "Professional product photography of eco-friendly items",
        "Modern tech gadget advertisement with clean background",
        "Fashion collection showcase with lifestyle setting"
    ]
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nGenerating image for prompt: {prompt}")
        try:
            image = pipeline.generate_marketing_image(prompt)
            image.save(f"generated_images/test_image_{i+1}.png")
            logger.info(f"Image saved as generated_images/test_image_{i+1}.png")
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")

def test_trend_prediction(pipeline, social_df):
    """Test trend prediction functionality"""
    logger.info("\n=== Testing Trend Prediction ===")
    
    # Prepare sample data for trend prediction
    # Using 'Age' and 'EstimatedSalary' as features (you can modify based on your actual columns)
    X = social_df[['Age', 'EstimatedSalary']].values[:100]  # Use first 100 samples
    y = social_df['Purchased'].values[:100]  # Assuming 'Purchased' is the target
    
    # Train the trend model
    logger.info("Training trend prediction model...")
    pipeline.train_trend_model(X, y, epochs=50)
    
    # Test prediction
    test_data = np.array([[25, 50000], [35, 75000], [45, 100000]])
    for i, data in enumerate(test_data):
        prediction = pipeline.predict_trends(data)
        logger.info(f"Prediction for sample {i+1}: {prediction:.4f}")

def main():
    try:
        # Load training data
        reviews_texts, social_df, campaigns_df = load_training_data()
        
        # Initialize pipeline
        logger.info("Initializing Marketing AI Pipeline...")
        pipeline = MarketingAIPipeline()
        pipeline.initialize_models()
        
        # Fine-tune text model with reviews
        logger.info("\nFine-tuning text model with reviews...")
        pipeline.fine_tune_text_model(reviews_texts, epochs=2)
        
        # Test all functionalities
        test_text_generation(pipeline)
        test_image_generation(pipeline)
        test_trend_prediction(pipeline, social_df)
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 