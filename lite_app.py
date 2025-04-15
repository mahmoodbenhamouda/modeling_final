import streamlit as st
import os
import random
import datetime
import numpy as np
from PIL import Image
import io
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Set StableDiffusion as not available by default to avoid CUDA errors
STABLE_DIFFUSION_AVAILABLE = False

# Set API-based image generation flag to True
INFERENCE_API_AVAILABLE = True

# Add DALL-E Mini API option as a backup
DALLE_MINI_AVAILABLE = True

# Set page configuration
st.set_page_config(
    page_title="Digital Marketing AI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .info-text {
        font-size: 0.9rem;
        color: #616161;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>🚀 Digital Marketing AI Platform (Lite)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate marketing content for your campaigns</p>", unsafe_allow_html=True)

# Simple marketing text generator function
def generate_marketing_text(prompt, use_real_data=False):
    """Generate marketing text based on prompt, optionally using real review data"""
    industry = ""
    audience = ""
    goal = ""
    product = ""
    tone = ""
    
    # Extract information from prompt
    if "technology" in prompt.lower():
        industry = "technology"
    elif "fashion" in prompt.lower():
        industry = "fashion"
    elif "food" in prompt.lower() or "beverage" in prompt.lower():
        industry = "food and beverage"
    elif "health" in prompt.lower() or "wellness" in prompt.lower():
        industry = "health and wellness"
    elif "finance" in prompt.lower():
        industry = "finance"
    elif "travel" in prompt.lower():
        industry = "travel"
    elif "entertainment" in prompt.lower():
        industry = "entertainment"
    elif "education" in prompt.lower():
        industry = "education"
    elif "retail" in prompt.lower():
        industry = "retail"
    else:
        industry = "product"
    
    if "young" in prompt.lower() or "18-24" in prompt.lower():
        audience = "young adults"
    elif "25-34" in prompt.lower():
        audience = "young professionals"
    elif "35-44" in prompt.lower():
        audience = "established professionals"
    elif "45-60" in prompt.lower():
        audience = "experienced consumers"
    elif "60+" in prompt.lower() or "seniors" in prompt.lower():
        audience = "seniors"
    elif "professionals" in prompt.lower():
        audience = "busy professionals"
    elif "students" in prompt.lower():
        audience = "students"
    elif "parents" in prompt.lower():
        audience = "parents"
    else:
        audience = "customers"
        
    if "brand awareness" in prompt.lower():
        goal = "brand awareness"
    elif "product launch" in prompt.lower():
        goal = "product launch"
    elif "promotion" in prompt.lower():
        goal = "promotion"
    elif "lead generation" in prompt.lower():
        goal = "lead generation"
    elif "customer retention" in prompt.lower():
        goal = "customer loyalty"
    elif "event marketing" in prompt.lower():
        goal = "upcoming event"
    else:
        goal = "marketing"
    
    # Extract tone information
    if "professional" in prompt.lower():
        tone = "professional"
    elif "casual" in prompt.lower():
        tone = "casual"
    elif "exciting" in prompt.lower():
        tone = "exciting"
    elif "informative" in prompt.lower():
        tone = "informative"
    elif "persuasive" in prompt.lower():
        tone = "persuasive"
    elif "humorous" in prompt.lower():
        tone = "fun"
    else:
        tone = "engaging"
        
    # Extract product description
    if "product details:" in prompt.lower():
        product = prompt.lower().split("product details:")[1].strip()
    else:
        product = f"{industry} solution"
    
    # Use review data if available
    if use_real_data and "reviews_data" in st.session_state:
        reviews = st.session_state["reviews_data"]
        
        # Filter reviews by industry (if relevant keywords appear in the review)
        industry_keywords = {
            "technology": ["tech", "electronic", "digital", "device", "gadget", "computer", "phone", "app"],
            "fashion": ["fashion", "clothing", "wear", "style", "outfit", "apparel", "dress", "shoes"],
            "food and beverage": ["food", "beverage", "drink", "taste", "flavor", "meal", "restaurant", "eat"],
            "health and wellness": ["health", "wellness", "fitness", "healthy", "workout", "supplement", "vitamin"],
            "finance": ["finance", "money", "payment", "bank", "cost", "price", "invest", "account"]
        }
        
        # Get keywords for our industry
        keywords = industry_keywords.get(industry, [industry])
        
        # Try to find relevant reviews
        filtered_reviews = []
        for keyword in keywords:
            if "Text" in reviews.columns:
                relevant = reviews[reviews["Text"].str.contains(keyword, case=False, na=False)]
                if not relevant.empty:
                    filtered_reviews.append(relevant)
        
        # Use review data to create marketing content
        if filtered_reviews:
            # Combine all relevant reviews
            combined_reviews = pd.concat(filtered_reviews).drop_duplicates()
            
            # Get positive reviews (rating >= 4 if Rating column exists)
            if "Rating" in combined_reviews.columns:
                positive_reviews = combined_reviews[combined_reviews["Rating"] >= 4]
            else:
                positive_reviews = combined_reviews
                
            if not positive_reviews.empty:
                # Extract positive phrases from reviews
                phrases = []
                
                for review in positive_reviews["Text"].sample(min(5, len(positive_reviews))):
                    # Simple sentence splitting
                    sentences = review.split(".")
                    for sentence in sentences:
                        if len(sentence) > 10 and any(pos in sentence.lower() for pos in ["great", "love", "best", "amazing", "excellent", "awesome", "good"]):
                            phrases.append(sentence.strip())
                
                if phrases:
                    # Create marketing copy using real customer phrases
                    testimonial = f'"{random.choice(phrases)}"'
                    
                    marketing_text = f"""Introducing our exceptional {industry} solution designed for {audience}!

Our {product} has been winning rave reviews from customers:

{testimonial}

Perfect for your {goal} needs. Don't miss this opportunity to join our satisfied customers!
"""
                    return marketing_text
    
    # Create templates for different tones
    professional_templates = [
        f"Introducing our premium {industry} collection tailored for {audience}. {product.capitalize()} delivers exceptional quality and performance for your {goal} needs. Elevate your experience with our meticulously designed solutions.",
        
        f"Meet the new standard in {industry}: our latest product designed specifically for {audience}. {product.capitalize()} combines innovation with reliability, making it the ideal choice for your {goal} requirements.",
        
        f"For discerning {audience} who demand excellence in {industry}, we present our newest offering. {product.capitalize()} represents our commitment to quality and performance, perfect for achieving your {goal} objectives."
    ]
    
    casual_templates = [
        f"Hey {audience}! Check out our awesome new {industry} product! {product.capitalize()} is exactly what you need for your {goal} - and it's super easy to use!",
        
        f"Looking for a great {industry} solution? We've got you covered! Our {product} is perfect for {audience} like you who want to rock their {goal} without any hassle.",
        
        f"We made something special just for {audience}! Our new {industry} product makes {goal} so much easier and more fun. {product.capitalize()} - because you deserve the best!"
    ]
    
    exciting_templates = [
        f"🔥 BREAKTHROUGH IN {industry.upper()}! 🔥 Unveiling our revolutionary product for {audience}! {product.capitalize()} will transform how you approach {goal}! Limited release - act now!",
        
        f"GAME-CHANGER ALERT! Our newest {industry} innovation is here to revolutionize {goal} for {audience}! {product.capitalize()} - prepare to be amazed!",
        
        f"INCREDIBLE NEWS for {audience}! We've just unleashed the most powerful {industry} solution for {goal}! {product.capitalize()} - redefining what's possible!"
    ]
    
    informative_templates = [
        f"Research shows that {audience} face unique challenges with {goal}. Our {industry} solution addresses these concerns through innovative design. {product.capitalize()} offers specific benefits including improved efficiency and reliability.",
        
        f"Our latest {industry} development brings three key advantages to {audience}: enhanced performance, streamlined experience, and long-term reliability. {product.capitalize()} was developed specifically to optimize your {goal} outcomes.",
        
        f"Understanding the needs of {audience}, we've engineered a {industry} solution with measurable benefits. {product.capitalize()} delivers quantifiable improvements for your {goal}, with attention to detail and quality assurance."
    ]
    
    persuasive_templates = [
        f"Why settle for ordinary {industry} solutions when you can experience extraordinary results? {product.capitalize()} was designed exclusively for {audience} who truly value excellence in {goal}. Make the smart choice today.",
        
        f"Thousands of {audience} have already discovered the difference our {industry} solution makes. {product.capitalize()} consistently outperforms alternatives for {goal}. Isn't it time you joined them?",
        
        f"You deserve better than average {industry} products. That's why we created a premium solution specifically for discerning {audience}. {product.capitalize()} - because your {goal} deserves nothing less than excellence."
    ]
    
    humorous_templates = [
        f"Attention {audience}! Are your current {industry} products making you say 'meh'? Our {product} is like that, but actually good! Perfect for {goal} or just showing off to your less cool friends!",
        
        f"Breaking news: {audience} no longer need to suffer through terrible {industry} experiences! Our {product} is here to save the day! Warning: may cause excessive happiness and success with {goal}!",
        
        f"Let's be honest, most {industry} products for {audience} are about as exciting as watching paint dry. But our {product}? It's like watching paint dry... ON THE MOON! Your {goal} will never be the same!"
    ]
    
    # Select template based on tone
    if tone == "professional":
        templates = professional_templates
    elif tone == "casual":
        templates = casual_templates
    elif tone == "exciting":
        templates = exciting_templates
    elif tone == "informative":
        templates = informative_templates
    elif tone == "persuasive":
        templates = persuasive_templates
    elif tone == "fun":
        templates = humorous_templates
    else:
        # Default to professional if tone not recognized
        templates = professional_templates
    
    # Add a slight delay to simulate processing time
    import time
    time.sleep(1)
    
    # Return a random template from the appropriate tone category
    return random.choice(templates)

# Simple sentiment analyzer
def analyze_sentiment(text):
    """A simple placeholder function for sentiment analysis"""
    # Count positive and negative words
    positive_words = ["revolutionary", "premium", "exceptional", "elevate", "transform", 
                      "innovation", "cutting-edge", "unparalleled", "perfect", "smart", 
                      "special", "outstanding", "innovative", "reliable"]
    
    negative_words = ["tired", "ordinary", "miss out"]
    
    # Count occurrences
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    
    # Calculate sentiment score (between 0 and 1)
    total_words = len(text.split())
    sentiment_score = (positive_count - negative_count * 0.5) / max(1, total_words) * 10
    sentiment_score = max(0, min(1, sentiment_score))  # Clip between 0 and 1
    
    # Determine sentiment label
    if sentiment_score >= 0.7:
        label = "VERY POSITIVE"
    elif sentiment_score >= 0.5:
        label = "POSITIVE"
    elif sentiment_score >= 0.3: 
        label = "NEUTRAL" 
    else: 
        label = "NEGATIVE" 
        
    return {
        "label": label,
        "score": sentiment_score
    }

# Simple emotion classifier
def analyze_emotion(text):
    """A simple placeholder function for emotion analysis"""
    # Define emotion keywords
    emotion_keywords = {
        "joy": ["revolutionary", "exceptional", "transform", "elevate", "perfect", "smart", "special"],
        "surprise": ["game-changing", "cutting-edge", "unparalleled", "innovation"],
        "neutral": ["product", "solution", "designed", "features", "customers"],
        "fear": ["miss out", "don't miss", "tired of"],
        "anger": [],
        "sadness": []
    }
    
    # Count occurrences
    emotion_counts = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for keyword in keywords if keyword in text.lower())
        emotion_counts[emotion] = count
    
    # Calculate probabilities
    total = sum(emotion_counts.values()) or 1  # Avoid division by zero
    emotion_probs = {emotion: count/total for emotion, count in emotion_counts.items()}
    
    # Normalize to ensure all add up to 1
    total_prob = sum(emotion_probs.values())
    emotion_probs = {emotion: prob/total_prob for emotion, prob in emotion_probs.items()}
    
    # Add random variation to make it more realistic
    for emotion in emotion_probs:
        variation = random.uniform(-0.1, 0.1)
        emotion_probs[emotion] = max(0, min(1, emotion_probs[emotion] + variation))
    
    # Renormalize
    total_prob = sum(emotion_probs.values())
    emotion_probs = {emotion: prob/total_prob for emotion, prob in emotion_probs.items()}
    
    return emotion_probs

# Generate a placeholder image
def generate_placeholder_image(width=512, height=512, color=(200, 220, 240)):
    """Generate a simple placeholder image with text"""
    img = Image.new('RGB', (width, height), color=color)
    
    # Add a slight gradient
    pixels = img.load()
    for i in range(width):
        for j in range(height):
            r = int(color[0] * (1 - j/height * 0.2))
            g = int(color[1] * (1 - j/height * 0.1))
            b = int(color[2] * (1 - j/height * 0.3))
            pixels[i, j] = (r, g, b)
    
    return img

# Generate AI image with multiple API options
def generate_ai_image(prompt, width=512, height=512, use_alternative=False):
    """Generate an image using Hugging Face Inference API or alternative APIs"""
    if not INFERENCE_API_AVAILABLE:
        st.warning("AI image generation is not available. Using simple image generator instead.")
        return generate_placeholder_image(width, height)
    
    # Try alternative API first if specified
    if use_alternative and DALLE_MINI_AVAILABLE:
        try:
            st.info("Using DALL-E Mini API (no authentication needed)")
            # Use the free DALL-E Mini clone API
            API_URL = "https://bf.dallemini.ai/generate"
            
            payload = {"prompt": prompt}
            
            with st.spinner("Generating image via DALL-E Mini API (this may take up to 45 seconds)..."):
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    # The response is a base64 encoded image
                    try:
                        image_data = response.json()["images"][0]
                        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                        return image
                    except Exception as e:
                        st.error(f"Error decoding image: {str(e)}")
                else:
                    st.error(f"DALL-E Mini API failed: {response.status_code}")
        except Exception as e:
            st.error(f"Error with DALL-E Mini API: {str(e)}")
            st.info("Trying Hugging Face API instead...")
    
    # If we get here, either alternative wasn't used or it failed
    try:
        # Using a more accessible model that works with the public API
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        
        # Check if there's a Hugging Face API token set (not required but recommended)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        headers = {}
        
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            st.success("Using your Hugging Face API token for image generation.")
        else:
            st.warning("""
            No Hugging Face API token found. To use AI image generation reliably:
            1. Create a free account at huggingface.co
            2. Create a token at https://huggingface.co/settings/tokens
            3. Add it as an environment variable HUGGINGFACE_TOKEN
            
            For now, using the public API which has severe limitations.
            """)
        
        with st.spinner("Generating image via Hugging Face API (this may take up to 30 seconds)..."):
            # Set negative prompt to help avoid text in image
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "text, watermark, signature, blurry, distorted, low quality",
                    "width": width,
                    "height": height
                }
            }
            
            # Make the API request
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                st.error(f"Error with Hugging Face API: {response.status_code}")
                error_text = response.text
                
                # Provide more helpful error messages
                if response.status_code == 401:
                    st.error("""
                    Authentication error (401): You need to create a Hugging Face account and token.
                    
                    To fix this:
                    1. Go to https://huggingface.co/join to create a free account
                    2. Get a token at https://huggingface.co/settings/tokens
                    3. Set it as an environment variable named HUGGINGFACE_TOKEN
                    """)
                elif response.status_code == 503:
                    st.error("The model is currently overloaded with requests. Please try again later.")
                else:
                    st.error(f"Response: {error_text}")
                
                # Try alternative API if Hugging Face fails
                if DALLE_MINI_AVAILABLE and not use_alternative:
                    st.info("Trying alternative image generation API...")
                    return generate_ai_image(prompt, width, height, use_alternative=True)
                
                st.info("Falling back to basic image generator...")
                return generate_placeholder_image(width, height)
            
            # Convert the image from binary response
            image = Image.open(io.BytesIO(response.content))
            return image
        
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        
        # Try alternative API if Hugging Face fails
        if DALLE_MINI_AVAILABLE and not use_alternative:
            st.info("Trying alternative image generation API...")
            return generate_ai_image(prompt, width, height, use_alternative=True)
            
        st.info("Using basic image generator instead.")
        return generate_placeholder_image(width, height)

# Function to load real marketing data from CSV files
def load_marketing_data():
    """Load marketing data from CSV files in the data folder"""
    data = {}
    try:
        # Load digital marketing campaigns data
        campaigns_path = os.path.join("data", "digital_marketing_campaigns_smes.csv")
        if os.path.exists(campaigns_path):
            data["campaigns"] = pd.read_csv(campaigns_path)
            st.session_state["data_loaded"] = True
        
        # Load social media advertising data
        social_media_path = os.path.join("data", "Social_Media_Advertising.csv")
        if os.path.exists(social_media_path):
            data["social_media"] = pd.read_csv(social_media_path)
        
        # Load reviews data (only if needed - this file is large)
        reviews_path = os.path.join("data", "Reviews.csv")
        if os.path.exists(reviews_path) and "load_reviews" in st.session_state and st.session_state["load_reviews"]:
            data["reviews"] = pd.read_csv(reviews_path)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        data["error"] = str(e)
    
    return data

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Tools & Settings")
    app_mode = st.radio(
        "Choose a Mode",
        ["Content Generator", "Image Generator", "Trend Analyzer"]
    )
    
    # Display information based on the selected mode
    if app_mode == "Content Generator":
        st.info("Generate engaging marketing text with sentiment analysis")
    elif app_mode == "Image Generator":
        st.info("Create simple marketing images with text overlay")
    elif app_mode == "Trend Analyzer":
        st.info("Analyze marketing trends with simulated data")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a lite version of the Digital Marketing AI Platform.
    It uses simplified algorithms to demonstrate the capabilities.
    
    Features:
    - Text generation
    - Sentiment analysis
    - Trend visualization
    """)

# Content Generator
if app_mode == "Content Generator":
    st.markdown("<h2 class='sub-header'>📝 Marketing Content Generator</h2>", unsafe_allow_html=True)
    
    # Add option to use real review data
    use_review_data = st.checkbox("Use real customer reviews data for content generation", value=True)
    
    # Check if we need to load review data
    if use_review_data and "reviews_data" not in st.session_state:
        reviews_path = os.path.join("data", "Reviews.csv")
        
        if os.path.exists(reviews_path):
            with st.spinner("Loading reviews data (this may take a moment)..."):
                try:
                    # Load just the necessary columns to save memory
                    st.session_state["reviews_data"] = pd.read_csv(reviews_path, usecols=["Text", "Rating"] if "Rating" in pd.read_csv(reviews_path, nrows=1).columns else ["Text"])
                    st.success(f"Loaded {len(st.session_state['reviews_data'])} customer reviews!")
                except Exception as e:
                    st.error(f"Error loading reviews data: {str(e)}")
                    st.session_state["reviews_data"] = None
                    use_review_data = False
        else:
            st.warning("Reviews.csv not found in the data directory. Using template-based generation instead.")
            use_review_data = False
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # User inputs
        industry = st.selectbox(
            "Select Industry",
            ["Technology", "Fashion", "Food & Beverage", "Health & Wellness", "Finance", "Travel", "Entertainment", "Education", "Retail"]
        )
        
        audience = st.selectbox(
            "Target Audience",
            ["Young Adults (18-24)", "Adults (25-34)", "Middle Age (35-44)", "Older Adults (45-60)", "Seniors (60+)", "Professionals", "Students", "Parents"]
        )
        
        goal = st.selectbox(
            "Marketing Goal",
            ["Brand Awareness", "Product Launch", "Promotion", "Lead Generation", "Customer Retention", "Event Marketing"]
        )
        
        tone = st.select_slider(
            "Content Tone",
            options=["Professional", "Casual", "Exciting", "Informative", "Persuasive", "Humorous"]
        )
        
        product_description = st.text_area("Product or Service Description (optional)", height=100)
        
        # Combine inputs to create a prompt
        prompt = f"Create a {tone.lower()} {goal.lower()} marketing content for a {industry.lower()} product targeting {audience.lower()}"
        if product_description:
            prompt += f". Product details: {product_description}"
        
        # Generate button
        if st.button("Generate Marketing Content"):
            with st.spinner("Generating content..." + (" Using real customer reviews data" if use_review_data else "")):
                # Generate text with real review data if selected
                generated_text = generate_marketing_text(prompt, use_real_data=use_review_data)
                
                # Analyze sentiment
                sentiment = analyze_sentiment(generated_text)
                
                # Analyze emotion
                emotion = analyze_emotion(generated_text)
            
            # Display results
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.subheader("Generated Content")
            st.write(generated_text)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show data source
            if use_review_data and "reviews_data" in st.session_state and st.session_state["reviews_data"] is not None:
                st.info("Content generated using insights from real customer reviews! 🌟")
            else:
                st.info("Content generated using template-based system.")
    
    with col2:
        st.markdown("<h3>Content Analysis</h3>", unsafe_allow_html=True)
        
        # Only show after generation
        if 'generated_text' in locals():
            # Sentiment Analysis
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.subheader("Sentiment Analysis")
            
            sentiment_score = sentiment["score"]
            sentiment_label = sentiment["label"]
            
            st.progress(sentiment_score)
            st.markdown(f"<p class='highlight'>Sentiment: {sentiment_label} ({sentiment_score:.2f})</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Emotion Analysis
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.subheader("Emotion Analysis")
            
            # Display emotions as a horizontal bar
            if isinstance(emotion, dict):
                for emotion_name, score in sorted(emotion.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"**{emotion_name.capitalize()}**: {score:.2f}")
                    st.progress(score)
                
                # Top emotion
                top_emotion = max(emotion.items(), key=lambda x: x[1])
                st.markdown(f"<p class='highlight'>Primary Emotion: {top_emotion[0].capitalize()} ({top_emotion[1]:.2f})</p>", unsafe_allow_html=True)
            else:
                st.error("Emotion analysis failed. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Generated content will be analyzed here.")

# Image Generator
elif app_mode == "Image Generator":
    st.markdown("<h2 class='sub-header'>🖼️ Marketing Image Generator</h2>", unsafe_allow_html=True)
    
    st.markdown("Create marketing images with AI or custom text overlay.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        use_ai = st.checkbox("Use AI Image Generation", value=True)
        
        if use_ai:
            api_option = st.radio(
                "Image Generation API",
                ["Hugging Face (requires token for best results)", "DALL-E Mini (no token required)"]
            )
            use_alternative = api_option.startswith("DALL-E")
            
            # Add token input option for Hugging Face
            if not use_alternative:
                with st.expander("🔑 Set a Hugging Face Token"):
                    st.markdown("""
                    To use AI image generation properly, you need a Hugging Face token.
                    
                    1. Create a free account at [huggingface.co](https://huggingface.co/join)
                    2. Get a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
                    3. Paste it below (it will only be used for this session):
                    """)
                    user_token = st.text_input("Hugging Face Token", type="password", key="hf_token")
                    if user_token:
                        os.environ["HUGGINGFACE_TOKEN"] = user_token
                        st.success("Token has been set for this session!")
        
        image_type = st.selectbox(
            "Image Type",
            ["Product Banner", "Social Media Post", "Advertisement", "Promotional Banner"]
        )
        
        if use_ai:
            product_category = st.selectbox(
                "Product Category",
                ["Technology", "Fashion", "Food", "Cosmetics", "Furniture", "Sports", "Travel"]
            )
            
            style = st.selectbox(
                "Visual Style",
                ["Photorealistic", "Minimalist", "Vibrant", "Professional", "Artistic", "Futuristic"]
            )
            
            # Custom prompt
            ai_prompt = st.text_area(
                "AI Image Prompt",
                f"A professional marketing image for {product_category.lower()} product, {style.lower()} style, no text, high quality"
            )
            
            st.info("Note: The Hugging Face API has usage limitations. If it's not working, you can use the basic generator instead.")
        else:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Blue", "Green", "Purple", "Orange", "Red"]
            )
            
            # Color mappings
            color_map = {
                "Blue": (100, 150, 220),
                "Green": (100, 200, 150),
                "Purple": (150, 100, 200),
                "Orange": (240, 150, 80),
                "Red": (220, 100, 100)
            }
        
        # Text inputs
        headline = st.text_input("Headline Text", "New Product Launch")
        subtext = st.text_area("Sub Text", "Discover our amazing new product with innovative features!")
        
        # Image size - limit to what the API supports
        width_options = [512, 768]
        height_options = [512, 768]
        
        width = st.select_slider("Width", options=width_options, value=512)
        height = st.select_slider("Height", options=height_options, value=512)
        
        # Generate button
        if st.button("Generate Marketing Image"):
            with st.spinner("Generating image..."):
                # Generate base image - either AI or gradient
                if use_ai:
                    img = generate_ai_image(ai_prompt, width, height, use_alternative=use_alternative)
                else:
                    base_color = color_map[color_scheme]
                    img = generate_placeholder_image(width, height, base_color)
                
                # Create a drawing context
                draw = ImageDraw.Draw(img)
                
                # Try to use a nice font, fallback to default
                try:
                    # Try to find a system font that exists
                    system_fonts = ["Arial", "Helvetica", "Verdana", "Tahoma", "Segoe UI"]
                    headline_font = None
                    subtext_font = None
                    
                    for font_name in system_fonts:
                        try:
                            headline_font = ImageFont.truetype(font_name, 48)
                            subtext_font = ImageFont.truetype(font_name, 24)
                            break
                        except:
                            continue
                            
                    if headline_font is None:
                        # If none of the system fonts work, use default font
                        headline_font = ImageFont.load_default()
                        subtext_font = ImageFont.load_default()
                except:
                    headline_font = ImageFont.load_default()
                    subtext_font = ImageFont.load_default()
                
                # Calculate text position
                headline_width = draw.textlength(headline, font=headline_font)
                headline_x = (width - headline_width) / 2
                headline_y = height / 3
                
                # Draw headline
                draw.text((headline_x, headline_y), headline, font=headline_font, fill=(255, 255, 255))
                
                # Draw subtext with wrapping
                subtext_lines = []
                words = subtext.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    test_width = draw.textlength(test_line, font=subtext_font)
                    
                    if test_width < width * 0.8:
                        current_line = test_line
                    else:
                        subtext_lines.append(current_line)
                        current_line = word
                
                if current_line:
                    subtext_lines.append(current_line)
                
                # Draw each line of subtext
                line_height = 30
                start_y = headline_y + 80
                
                for i, line in enumerate(subtext_lines):
                    line_width = draw.textlength(line, font=subtext_font)
                    line_x = (width - line_width) / 2
                    draw.text((line_x, start_y + i * line_height), line, font=subtext_font, fill=(255, 255, 255))
                
                # Add a simple overlay based on the image type
                if image_type == "Product Banner":
                    draw.rectangle([(50, 50), (width - 50, height - 50)], outline=(255, 255, 255), width=5)
                elif image_type == "Social Media Post":
                    draw.ellipse([(width - 150, height - 150), (width - 50, height - 50)], fill=(255, 255, 255, 128))
                elif image_type == "Advertisement":
                    draw.rectangle([(50, height - 100), (width - 50, height - 50)], fill=(0, 0, 0, 128))
                    draw.text(((width - draw.textlength("LEARN MORE", font=subtext_font)) / 2, height - 85), 
                             "LEARN MORE", font=subtext_font, fill=(255, 255, 255))
                
                # Display the image
                st.image(img, caption=f"{image_type}: {headline}")
                
                # Add download button
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Encode image to base64
                img_base64 = base64.b64encode(img_byte_arr.read()).decode()
                
                # Create download link
                href = f'<a href="data:image/png;base64,{img_base64}" download="{headline.replace(" ", "_")}.png">Download Image</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Image Preview")
        st.markdown("Generated image will appear here.")
        st.markdown("### Tips for Good Marketing Images")
        st.markdown("""
        - Keep headlines short and impactful
        - Use contrasting colors for text visibility
        - Balance text and visual elements
        - Ensure the message is clear and focused
        - Match image style to your brand's personality
        """)

# Trend Analyzer
elif app_mode == "Trend Analyzer":
    st.markdown("<h2 class='sub-header'>📊 Marketing Trend Analyzer</h2>", unsafe_allow_html=True)
    
    # Check if we should use real data or synthetic data
    use_real_data = st.checkbox("Use real marketing data from CSV files", value=True)
    
    # Add user region input - place this high in the UI
    user_region = st.text_input("Enter Your Region/Market", placeholder="e.g., North America, Europe, Asia")
    
    if use_real_data:
        # Load real marketing data
        with st.spinner("Loading marketing data from CSV files..."):
            data = load_marketing_data()
        
        if "campaigns" in data:
            st.success(f"Successfully loaded real marketing campaign data ({len(data['campaigns'])} records)")
            
            # Data filtering options
            st.subheader("Filter Data")
            
            col1, col2, col3 = st.columns(3)
            
            # Get unique values for filters, but check if columns exist first
            # For industry column
            if "industry" in data["campaigns"].columns:
                industries = ['All'] + sorted(data["campaigns"]["industry"].unique().tolist())
            else:
                # Try alternatives or use default
                for possible_column in ["business_type", "category", "product_category"]:
                    if possible_column in data["campaigns"].columns:
                        industries = ['All'] + sorted(data["campaigns"][possible_column].unique().tolist())
                        st.info(f"Using '{possible_column}' as industry filter")
                        break
                else:
                    industries = ['All']
                    st.warning("No industry column found in the data")
            
            # For channel column
            if "channel" in data["campaigns"].columns:
                channels = ['All'] + sorted(data["campaigns"]["channel"].unique().tolist())
            else:
                # Try alternatives or use default
                for possible_column in ["marketing_channel", "platform", "ad_platform"]:
                    if possible_column in data["campaigns"].columns:
                        channels = ['All'] + sorted(data["campaigns"][possible_column].unique().tolist())
                        st.info(f"Using '{possible_column}' as channel filter")
                        break
                else:
                    channels = ['All']
                    st.warning("No channel column found in the data")
            
            # For region column (if available)
            region_column = None
            for col in ["region", "location", "market", "geography", "country"]:
                if col in data["campaigns"].columns:
                    region_column = col
                    regions = ['All'] + sorted(data["campaigns"][col].unique().tolist())
                    break
            else:
                regions = ['All']
                if user_region:
                    regions.append(user_region)  # Add user's custom region
            
            with col1:
                selected_industry = st.selectbox("Select Industry", industries)
                
            with col2:
                selected_channel = st.selectbox("Select Marketing Channel", channels)
                
            with col3:
                if region_column:
                    selected_region = st.selectbox("Select Region", regions)
                    if selected_region == user_region and user_region not in data["campaigns"][region_column].unique():
                        st.info(f"'{user_region}' is not in your historical data. Showing predictions for a new market.")
            
            # Filter the data
            filtered_data = data["campaigns"].copy()
            
            # Apply industry filter
            if selected_industry != 'All':
                if "industry" in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data["industry"] == selected_industry]
                else:
                    # Try alternatives
                    for possible_column in ["business_type", "category", "product_category"]:
                        if possible_column in filtered_data.columns:
                            filtered_data = filtered_data[filtered_data[possible_column] == selected_industry]
                            break
            
            # Apply channel filter
            if selected_channel != 'All':
                if "channel" in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data["channel"] == selected_channel]
                else:
                    # Try alternatives
                    for possible_column in ["marketing_channel", "platform", "ad_platform"]:
                        if possible_column in filtered_data.columns:
                            filtered_data = filtered_data[filtered_data[possible_column] == selected_channel]
                            break
            
            # Apply region filter if selected and available
            if region_column and selected_region != 'All' and selected_region != user_region:
                filtered_data = filtered_data[filtered_data[region_column] == selected_region]
            
            # Display filtered statistics based on what's actually available in the dataset
            st.subheader("Campaign Performance Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            # Display engagement metric
            with metrics_col1:
                if "engagement_metric" in filtered_data.columns:
                    avg_engagement = filtered_data["engagement_metric"].mean()
                    st.metric("Avg. Engagement", f"{avg_engagement:.3f}")
                else:
                    st.metric("Engagement", "N/A")
            
            # Display conversion rate
            with metrics_col2:
                if "conversion_rate" in filtered_data.columns:
                    avg_conversion = filtered_data["conversion_rate"].mean() * 100
                    st.metric("Conversion Rate", f"{avg_conversion:.2f}%")
                else:
                    st.metric("Conversion Rate", "N/A")
            
            # Display budget allocation
            with metrics_col3:
                if "budget_allocation" in filtered_data.columns:
                    avg_budget = filtered_data["budget_allocation"].mean()
                    st.metric("Avg. Budget", f"${avg_budget:.2f}")
                else:
                    st.metric("Budget", "N/A")
            
            # Display audience reach
            with metrics_col4:
                if "audience_reach" in filtered_data.columns:
                    total_reach = filtered_data["audience_reach"].sum()
                    st.metric("Total Reach", f"{total_reach:,.0f}")
                else:
                    st.metric("Audience Reach", "N/A")
            
            # Create visualizations based on available data
            st.subheader("Data Visualization")
            
            # Create new section for conversion rates by platform (device, OS, browser)
            if any(col in filtered_data.columns for col in ["device_conversion_rate", "os_conversion_rate", "browser_conversion_rate"]):
                st.write("Conversion Rates by Platform")
                
                platform_data = {}
                
                # Check and add each platform's conversion rate
                if "device_conversion_rate" in filtered_data.columns:
                    platform_data["Device"] = filtered_data["device_conversion_rate"].mean() * 100
                    
                if "os_conversion_rate" in filtered_data.columns:
                    platform_data["OS"] = filtered_data["os_conversion_rate"].mean() * 100
                    
                if "browser_conversion_rate" in filtered_data.columns:
                    platform_data["Browser"] = filtered_data["browser_conversion_rate"].mean() * 100
                
                # Create ASCII chart
                if platform_data:
                    chart_data = ""
                    max_value = max(platform_data.values())
                    
                    for platform, value in platform_data.items():
                        bars = "█" * int((value / max_value) * 40)
                        chart_data += f"{platform:<10} | {bars} {value:.2f}%\n"
                    
                    st.code(chart_data)
            
            # Create comparison by device type if device column exists
            if "device" in filtered_data.columns and "conversion_rate" in filtered_data.columns:
                device_data = filtered_data.groupby("device")["conversion_rate"].mean().reset_index()
                
                # Format the data for display
                chart_data = ""
                for _, row in device_data.iterrows():
                    device = row["device"]
                    rate = row["conversion_rate"] * 100
                    normalized = int((rate / device_data["conversion_rate"].max()) * 40)
                    bars = "█" * normalized
                    chart_data += f"{device:<15} | {bars} {rate:.2f}%\n"
                
                st.text("Conversion Rate by Device Type:")
                st.code(chart_data)
            
            # Create comparison by region if region column exists
            if region_column and "conversion_rate" in filtered_data.columns:
                region_data = filtered_data.groupby(region_column)["conversion_rate"].mean().reset_index()
                
                # Format the data for display
                chart_data = ""
                for _, row in region_data.iterrows():
                    region = row[region_column]
                    rate = row["conversion_rate"] * 100
                    normalized = int((rate / region_data["conversion_rate"].max()) * 40)
                    bars = "█" * normalized
                    chart_data += f"{region:<15} | {bars} {rate:.2f}%\n"
                
                st.text(f"Conversion Rate by {region_column.capitalize()}:")
                st.code(chart_data)
            
            # Advanced Trend Analysis and Prediction
            st.subheader("🔮 Advanced Trend Analysis & Prediction")
            
            # Check if we have campaign_id and conversion_rate for trend analysis
            if "campaign_id" in filtered_data.columns and "conversion_rate" in filtered_data.columns:
                # Sort by campaign_id to analyze trend over time
                trend_data = filtered_data.sort_values("campaign_id")
                
                # If we have enough data points, show trend
                if len(trend_data) >= 3:
                    # Create simple chart showing conversion rate trend over campaigns
                    chart_data = ""
                    for _, row in trend_data.iterrows():
                        campaign = f"Campaign {row['campaign_id']}"
                        rate = row["conversion_rate"] * 100
                        normalized = int((rate / trend_data["conversion_rate"].max()) * 40)
                        bars = "█" * normalized
                        chart_data += f"{campaign:<15} | {bars} {rate:.2f}%\n"
                    
                    st.text("Conversion Rate Trend Across Campaigns:")
                    st.code(chart_data)
                    
                    # Advanced prediction models
                    try:
                        import numpy as np
                        from sklearn.linear_model import LinearRegression
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.ensemble import RandomForestRegressor
                        
                        # Create tabs for different prediction models
                        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Linear Trend", "Polynomial Trend", "Random Forest"])
                        
                        # Prepare data for prediction
                        X = trend_data["campaign_id"].values.reshape(-1, 1)
                        y = trend_data["conversion_rate"].values
                        
                        # Calculate next campaign ID
                        next_campaign = trend_data["campaign_id"].max() + 1
                        
                        # 1. Linear Regression
                        with pred_tab1:
                            # Create and fit model
                            linear_model = LinearRegression()
                            linear_model.fit(X, y)
                            
                            # Predict
                            linear_pred = linear_model.predict([[next_campaign]])[0]
                            
                            # Show prediction
                            st.write(f"### Linear Trend Prediction")
                            st.write(f"Based on linear trend analysis, the predicted conversion rate for Campaign {next_campaign} is:")
                            prediction_color = "green" if linear_pred >= trend_data["conversion_rate"].mean() else "red"
                            st.markdown(f"<h2 style='color: {prediction_color};'>{linear_pred*100:.2f}%</h2>", unsafe_allow_html=True)
                            
                            # Add prediction context
                            last_rate = trend_data["conversion_rate"].iloc[-1]
                            change = ((linear_pred - last_rate) / last_rate) * 100
                            
                            if change > 0:
                                st.success(f"This represents a {abs(change):.2f}% **increase** from the last campaign.")
                            else:
                                st.error(f"This represents a {abs(change):.2f}% **decrease** from the last campaign.")
                            
                            # Add confidence score
                            r2_score = linear_model.score(X, y)
                            st.progress(r2_score)
                            st.write(f"Confidence score: {r2_score:.2f}")
                        
                        # 2. Polynomial Regression
                        with pred_tab2:
                            # Create polynomial features
                            degree = min(3, len(X) - 1)  # Avoid overfitting
                            poly = PolynomialFeatures(degree=degree)
                            X_poly = poly.fit_transform(X)
                            
                            # Create and fit model
                            poly_model = LinearRegression()
                            poly_model.fit(X_poly, y)
                            
                            # Predict
                            X_next_poly = poly.transform([[next_campaign]])
                            poly_pred = poly_model.predict(X_next_poly)[0]
                            
                            # Show prediction
                            st.write(f"### Polynomial Trend Prediction")
                            st.write(f"Based on non-linear trend analysis (degree {degree}), the predicted conversion rate for Campaign {next_campaign} is:")
                            prediction_color = "green" if poly_pred >= trend_data["conversion_rate"].mean() else "red"
                            st.markdown(f"<h2 style='color: {prediction_color};'>{poly_pred*100:.2f}%</h2>", unsafe_allow_html=True)
                            
                            # Add prediction context
                            last_rate = trend_data["conversion_rate"].iloc[-1]
                            change = ((poly_pred - last_rate) / last_rate) * 100
                            
                            if change > 0:
                                st.success(f"This represents a {abs(change):.2f}% **increase** from the last campaign.")
                            else:
                                st.error(f"This represents a {abs(change):.2f}% **decrease** from the last campaign.")
                            
                            # Add confidence score
                            r2_score_poly = poly_model.score(X_poly, y)
                            st.progress(r2_score_poly)
                            st.write(f"Confidence score: {r2_score_poly:.2f}")
                        
                        # 3. Random Forest (more robust to outliers)
                        with pred_tab3:
                            # Only use if we have enough data
                            if len(X) >= 5:
                                # Create and fit model
                                rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
                                rf_model.fit(X, y)
                                
                                # Predict
                                rf_pred = rf_model.predict([[next_campaign]])[0]
                                
                                # Show prediction
                                st.write(f"### Random Forest Prediction")
                                st.write(f"Based on ensemble learning, the predicted conversion rate for Campaign {next_campaign} is:")
                                prediction_color = "green" if rf_pred >= trend_data["conversion_rate"].mean() else "red"
                                st.markdown(f"<h2 style='color: {prediction_color};'>{rf_pred*100:.2f}%</h2>", unsafe_allow_html=True)
                                
                                # Add prediction context
                                last_rate = trend_data["conversion_rate"].iloc[-1]
                                change = ((rf_pred - last_rate) / last_rate) * 100
                                
                                if change > 0:
                                    st.success(f"This represents a {abs(change):.2f}% **increase** from the last campaign.")
                                else:
                                    st.error(f"This represents a {abs(change):.2f}% **decrease** from the last campaign.")
                                
                                # Feature importance
                                st.write("### Feature Importance")
                                st.write("Campaign sequence importance:")
                                st.progress(1.0)  # Only one feature, so 100% important
                            else:
                                st.warning("Not enough data points for Random Forest prediction (need at least 5 campaigns).")
                        
                        # Prediction for user's region if specified
                        if user_region and user_region not in regions:
                            st.subheader(f"🌎 Prediction for Your Region: {user_region}")
                            
                            # Get average region effect if we have region data
                            region_effect = 1.0  # Default: no effect
                            if region_column:
                                # Calculate how much each region deviates from the average
                                global_avg = filtered_data["conversion_rate"].mean()
                                region_effects = {}
                                
                                for region in filtered_data[region_column].unique():
                                    region_data = filtered_data[filtered_data[region_column] == region]
                                    region_avg = region_data["conversion_rate"].mean()
                                    region_effects[region] = region_avg / global_avg
                                
                                # Use the average regional effect
                                region_effect = sum(region_effects.values()) / len(region_effects)
                            
                            # Apply the regional effect to our best model prediction
                            if r2_score_poly > r2_score and r2_score_poly > 0.5:
                                best_pred = poly_pred
                                model_name = "polynomial"
                            elif len(X) >= 5 and r2_score > 0.5:
                                best_pred = rf_pred
                                model_name = "random forest"
                            else:
                                best_pred = linear_pred
                                model_name = "linear"
                            
                            # Apply region effect
                            region_pred = best_pred * region_effect
                            
                            # Show prediction
                            st.write(f"Based on {model_name} model and regional patterns, the predicted conversion rate for Campaign {next_campaign} in {user_region} is:")
                            prediction_color = "green" if region_pred >= trend_data["conversion_rate"].mean() else "red"
                            st.markdown(f"<h2 style='color: {prediction_color};'>{region_pred*100:.2f}%</h2>", unsafe_allow_html=True)
                            
                            # Regional impact
                            if region_effect > 1:
                                st.success(f"Your region factor: {region_effect:.2f}x (positive impact)")
                            elif region_effect < 1:
                                st.warning(f"Your region factor: {region_effect:.2f}x (negative impact)")
                            else:
                                st.info("Your region uses the baseline prediction (no specific regional data)")
                        
                    except Exception as e:
                        st.error(f"Could not perform trend prediction: {str(e)}")
        else:
            if "error" in data:
                st.error(f"Could not load real data: {data['error']}")
            else:
                st.warning("No campaign data found. Using synthetic data instead.")
                use_real_data = False
    
    if not use_real_data:
        st.markdown("This module generates synthetic marketing campaign data to demonstrate trend analysis.")
        
        # Create some demo parameters
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_duration = st.slider("Campaign Duration (days)", 10, 60, 30)
            ad_budget = st.slider("Daily Ad Budget ($)", 50, 500, 100)
            
        with col2:
            platform = st.selectbox(
                "Marketing Platform",
                ["Facebook", "Instagram", "Twitter", "LinkedIn", "TikTok", "Google Ads"]
            )
            
            target_demographic = st.multiselect(
                "Target Demographics",
                ["18-24", "25-34", "35-44", "45-60", "60+"],
                default=["25-34"]
            )
        
        if st.button("Generate Campaign Trends"):
            # Generate synthetic data
            dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(campaign_duration)]
            
            # Create baseline metrics with some randomness
            impressions = []
            clicks = []
            conversions = []
            
            base_impressions = ad_budget * 10
            base_ctr = 0.02  # 2% CTR
            base_conversion = 0.05  # 5% conversion
            
            # Add trends and seasonality
            for i in range(campaign_duration):
                # Add weekly seasonality
                day_of_week = (dates[i].weekday() + 1) % 7  # 0 = Monday, 6 = Sunday
                weekday_factor = 1.0 + 0.2 * (day_of_week > 4)  # Weekend boost
                
                # Add random noise
                noise = random.uniform(0.8, 1.2)
                
                # Add upward trend as campaign progresses
                trend_factor = 1.0 + (i / campaign_duration) * 0.5
                
                # Calculate metrics
                day_impressions = int(base_impressions * weekday_factor * noise * trend_factor)
                day_ctr = base_ctr * weekday_factor * random.uniform(0.9, 1.1)
                day_clicks = int(day_impressions * day_ctr)
                day_conversion = base_conversion * random.uniform(0.8, 1.2)
                day_conversions = int(day_clicks * day_conversion)
                
                impressions.append(day_impressions)
                clicks.append(day_clicks)
                conversions.append(day_conversions)
            
            # Display campaign metrics
            st.subheader("Campaign Performance Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            total_impressions = sum(impressions)
            total_clicks = sum(clicks)
            total_conversions = sum(conversions)
            
            with metrics_col1:
                st.metric("Total Impressions", f"{total_impressions:,}")
            
            with metrics_col2:
                st.metric("Total Clicks", f"{total_clicks:,}")
            
            with metrics_col3:
                st.metric("Total Conversions", f"{total_conversions:,}")
            
            with metrics_col4:
                roi = (total_conversions * 50 - ad_budget * campaign_duration) / (ad_budget * campaign_duration) * 100
                st.metric("ROI", f"{roi:.2f}%")
            
            # Create a simple chart using ASCII characters
            st.subheader("Campaign Trends")
            
            # Normalize the data for ASCII chart
            max_impressions = max(impressions)
            normalized_impressions = [imp / max_impressions * 20 for imp in impressions]
            
            # Create the ASCII chart
            chart_data = ""
            for i in range(campaign_duration):
                date_str = dates[i].strftime("%m-%d")
                bars = "█" * int(normalized_impressions[i])
                chart_data += f"{date_str} | {bars} {impressions[i]:,}\n"
            
            st.text("Daily Impressions Chart:")
            st.code(chart_data)
            
            # Add recommendations based on the "data"
            st.subheader("Campaign Insights")
            
            day_with_max_impressions = dates[impressions.index(max(impressions))].strftime("%A, %B %d")
            day_with_max_conversions = dates[conversions.index(max(conversions))].strftime("%A, %B %d")
            
            st.markdown(f"✅ **Best day for impressions**: {day_with_max_impressions}")
            st.markdown(f"✅ **Best day for conversions**: {day_with_max_conversions}")
            st.markdown(f"✅ **Average daily CTR**: {sum(clicks)/sum(impressions)*100:.2f}%")
            st.markdown(f"✅ **Average conversion rate**: {sum(conversions)/sum(clicks)*100:.2f}%")
            
            if sum(impressions[0:7]) < sum(impressions[-7:]):
                st.markdown("📈 **Trend analysis**: Your campaign shows an upward trend, suggesting the audience is responding well over time.")
            else:
                st.markdown("📉 **Trend analysis**: Your campaign shows audience fatigue. Consider refreshing creative assets.")
                
            if max(impressions) > 3 * min(impressions):
                st.markdown("⚠️ **High volatility**: Your campaign shows significant day-to-day variations. Consider more consistent messaging.")
            
            st.info("Note: This data is simulated for demonstration purposes only.")

# Footer
st.markdown("---")
st.markdown("<p class='info-text' style='text-align: center;'>Digital Marketing AI Platform (Lite) • Local Demo Version</p>", unsafe_allow_html=True) 