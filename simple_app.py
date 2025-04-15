import streamlit as st
import os
import base64
import io
from PIL import Image
import numpy as np

# Check and import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

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
st.markdown("<h1 class='main-header'>🚀 Digital Marketing AI Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate marketing content, images, and analyze sentiment and emotions</p>", unsafe_allow_html=True)

# Check for required packages
missing_packages = []
if not PANDAS_AVAILABLE:
    missing_packages.append("pandas")
if not MATPLOTLIB_AVAILABLE:
    missing_packages.append("matplotlib")
if not SEABORN_AVAILABLE:
    missing_packages.append("seaborn")

if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.info("Please install the missing packages using: `pip install " + " ".join(missing_packages) + "`")
    st.stop()

# Try importing the AI pipeline with fallbacks
try:
    from marketing_ai_pipeline import MarketingAIPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing marketing AI pipeline: {str(e)}")
    PIPELINE_AVAILABLE = False

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Tools & Settings")
    app_mode = st.radio(
        "Choose a Mode",
        ["Content Generator", "Image Generator", "Trend Analyzer", "Complete Package"]
    )
    
    # Display information based on the selected mode
    if app_mode == "Content Generator":
        st.info("Generate engaging marketing text with emotion analysis")
    elif app_mode == "Image Generator":
        st.info("Create professional marketing images based on prompts")
    elif app_mode == "Trend Analyzer":
        st.info("Analyze marketing trends from historical data")
    elif app_mode == "Complete Package":
        st.info("Generate a complete marketing package with text, images, and analysis")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This platform helps marketers create content, analyze sentiment, 
    and generate images using AI technology.
    
    Features:
    - Text generation
    - Image creation
    - Emotion analysis
    - Named entity recognition
    - Trend prediction
    """)

# Check if pipeline is available and can be loaded
if PIPELINE_AVAILABLE:
    # Initialize pipeline
    @st.cache_resource
    def load_pipeline():
        try:
            pipeline = MarketingAIPipeline()
            pipeline.initialize_models()
            return pipeline
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return None

    with st.spinner("Loading AI models... This might take a few minutes on first run."):
        pipeline = load_pipeline()
        if pipeline:
            st.success("AI models loaded successfully!")
        else:
            st.error("Failed to load AI models. Please check the logs.")
            st.stop()
else:
    st.error("Marketing AI pipeline is not available. Please install the required packages.")
    st.stop()

# Content Generator
if app_mode == "Content Generator":
    st.markdown("<h2 class='sub-header'>📝 Marketing Content Generator</h2>", unsafe_allow_html=True)
    
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
            with st.spinner("Generating content..."):
                try:
                    # Generate text
                    generated_text = pipeline.generate_marketing_text(prompt)
                    
                    # Analyze sentiment
                    sentiment = pipeline.analyze_sentiment(generated_text)
                    
                    # Analyze emotion
                    emotion = pipeline.analyze_emotion(generated_text)
                    
                    # Extract entities
                    entities = pipeline.extract_entities(generated_text)
                
                    # Display results
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.subheader("Generated Content")
                    st.write(generated_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating content: {str(e)}")
    
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
            
            # Create DataFrame for emotions
            emotions_df = pd.DataFrame({
                'Emotion': list(emotion.keys()),
                'Score': list(emotion.values())
            })
            emotions_df = emotions_df.sort_values('Score', ascending=False)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Score', y='Emotion', data=emotions_df, palette='viridis', ax=ax)
            ax.set_xlim(0, 1)
            ax.set_title('Emotion Distribution')
            st.pyplot(fig)
            
            # Top emotion
            top_emotion = emotions_df.iloc[0]
            st.markdown(f"<p class='highlight'>Primary Emotion: {top_emotion['Emotion']} ({top_emotion['Score']:.2f})</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Entity Analysis
            if entities and entities.get("entity_counts"):
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.subheader("Named Entity Recognition")
                
                entity_counts = entities.get("entity_counts", {})
                
                if entity_counts:
                    # Create DataFrame for entities
                    entities_df = pd.DataFrame({
                        'Entity Type': list(entity_counts.keys()),
                        'Count': list(entity_counts.values())
                    })
                    entities_df = entities_df.sort_values('Count', ascending=False).head(8)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='Count', y='Entity Type', data=entities_df, palette='muted', ax=ax)
                    ax.set_title('Entity Types Mentioned')
                    st.pyplot(fig)
                    
                    # Marketing categories
                    marketing_categories = entities.get("marketing_categories", {})
                    if marketing_categories:
                        st.markdown("### Marketing Categories")
                        for category, items in marketing_categories.items():
                            if items:
                                st.write(f"**{category}:** {', '.join(items[:3])}")
                else:
                    st.info("No entities detected in the generated content.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Generated content will be analyzed here.")

# Image Generator
elif app_mode == "Image Generator":
    st.markdown("<h2 class='sub-header'>🖼️ Marketing Image Generator</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image_style = st.selectbox(
            "Image Style",
            ["Realistic Photo", "Abstract", "Minimalist", "Artistic", "Vintage", "Futuristic", "Corporate", "Lifestyle"]
        )
        
        image_subject = st.selectbox(
            "Image Subject",
            ["Product", "People Using Product", "Lifestyle", "Abstract Concept", "Brand Elements", "Nature", "Urban", "Technology"]
        )
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Vibrant", "Pastel", "Monochrome", "Dark", "Light", "Brand Colors", "Complementary", "Natural"]
        )
        
        description = st.text_area("Additional Description", height=100)
        
        # Create prompt for image generation
        image_prompt = f"Marketing image in {image_style} style featuring {image_subject} with {color_scheme} color scheme"
        if description:
            image_prompt += f", {description}"
        
        if st.button("Generate Image"):
            with st.spinner("Generating marketing image... This might take a minute."):
                try:
                    # Generate the image
                    generated_image = pipeline.generate_marketing_image(image_prompt)
                    
                    # Convert to buffer for display
                    buf = io.BytesIO()
                    generated_image.save(buf, format="PNG")
                    buf.seek(0)
                    
                    # Convert to base64
                    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
    
    with col2:
        # Display the generated image
        if 'generated_image' in locals():
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.subheader("Generated Marketing Image")
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(generated_image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="marketing_image.png",
                mime="image/png"
            )
            
            st.markdown("<p class='info-text'>This image is generated by AI and can be used for your marketing campaigns.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Your generated image will appear here.")
            
            # Display a placeholder
            placeholder_image = Image.new('RGB', (512, 512), color=(240, 240, 240))
            # Add text to the placeholder
            st.image(placeholder_image, caption="Image will be generated here", use_column_width=True)

# Trend Analyzer
elif app_mode == "Trend Analyzer":
    st.markdown("<h2 class='sub-header'>📊 Marketing Trend Analyzer</h2>", unsafe_allow_html=True)
    
    # Option to use example data or upload own
    data_source = st.radio(
        "Data Source",
        ["Use Example Data", "Upload Your Own Data"]
    )
    
    if data_source == "Use Example Data":
        # Load example data
        try:
            social_df = pd.read_csv('data/Social_Media_Advertising.csv', nrows=1000)
            st.success(f"Loaded example data with {len(social_df)} records")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(social_df.head(5))
            
            # Select features for trend analysis
            st.subheader("Select Features for Trend Analysis")
            
            # Prepare features and target
            categorical_columns = social_df.select_dtypes(include=['object']).columns.tolist()
            numeric_columns = social_df.select_dtypes(include=['number']).columns.tolist()
            
            # Let user select features
            selected_features = st.multiselect(
                "Select Features",
                options=numeric_columns,
                default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
            )
            
            target_feature = st.selectbox(
                "Select Target Variable",
                options=numeric_columns,
                index=numeric_columns.index('Conversion_Rate') if 'Conversion_Rate' in numeric_columns else 0
            )
            
            # Train model button
            if st.button("Analyze Trends"):
                if len(selected_features) > 0 and target_feature:
                    with st.spinner("Analyzing trends..."):
                        try:
                            # Prepare data
                            X = social_df[selected_features].values
                            y = social_df[target_feature].values
                            
                            # Normalize data
                            X_mean = np.mean(X, axis=0)
                            X_std = np.std(X, axis=0)
                            X_norm = (X - X_mean) / X_std
                            
                            # Train the model
                            losses = pipeline.train_trend_model(X_norm, y, epochs=50)
                            
                            # Create a plot of the loss
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(losses)
                            ax.set_title('Training Loss Over Time')
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Loss')
                            ax.grid(True)
                            st.pyplot(fig)
                            
                            # Make predictions for visualization
                            predictions = []
                            for i in range(len(X_norm)):
                                pred = pipeline.predict_trends(X_norm[i])
                                predictions.append(pred)
                            
                            # Compare predictions with actual
                            results_df = pd.DataFrame({
                                'Actual': y,
                                'Predicted': predictions
                            })
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
                            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
                            ax.set_title('Actual vs Predicted Values')
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            st.pyplot(fig)
                            
                            # Show prediction metrics
                            mae = np.mean(np.abs(results_df['Actual'] - results_df['Predicted']))
                            st.markdown(f"<p class='highlight'>Mean Absolute Error: {mae:.4f}</p>", unsafe_allow_html=True)
                            
                            # Allow user to test predictions
                            st.subheader("Test Predictions")
                            st.markdown("Input values to predict the target variable:")
                            
                            test_values = []
                            for i, feature in enumerate(selected_features):
                                min_val = float(social_df[feature].min())
                                max_val = float(social_df[feature].max())
                                mean_val = float(social_df[feature].mean())
                                test_values.append(st.slider(f"{feature}", min_val, max_val, mean_val))
                            
                            # Normalize test values
                            test_values_norm = (np.array(test_values) - X_mean) / X_std
                            
                            # Make prediction
                            prediction = pipeline.predict_trends(test_values_norm)
                            
                            st.markdown(f"<p class='highlight'>Predicted {target_feature}: {prediction:.4f}</p>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error during trend analysis: {str(e)}")
                else:
                    st.error("Please select at least one feature and a target variable.")
            
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
    
    else:  # Upload own data
        st.markdown("Upload your own data file (CSV format)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded data with {len(user_df)} records and {len(user_df.columns)} columns")
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(user_df.head(5))
                
                # Same trend analysis as for example data...
                st.info("Select columns for analysis in the sidebar")
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")

# Complete Package
elif app_mode == "Complete Package":
    st.markdown("<h2 class='sub-header'>🎁 Complete Marketing Package</h2>", unsafe_allow_html=True)
    
    # User inputs
    col1, col2 = st.columns([1, 1])
    
    with col1:
        product_name = st.text_input("Product or Service Name")
        
        industry = st.selectbox(
            "Industry",
            ["Technology", "Fashion", "Food & Beverage", "Health & Wellness", "Finance", "Travel", "Entertainment", "Education", "Retail"]
        )
        
        audience = st.selectbox(
            "Target Audience",
            ["Young Adults (18-24)", "Adults (25-34)", "Middle Age (35-44)", "Older Adults (45-60)", "Seniors (60+)", "Professionals", "Students", "Parents"]
        )
    
    with col2:
        marketing_goal = st.selectbox(
            "Marketing Goal",
            ["Brand Awareness", "Product Launch", "Promotion", "Lead Generation", "Customer Retention", "Event Marketing"]
        )
        
        content_platform = st.selectbox(
            "Content Platform",
            ["Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok", "Email", "Blog", "Website"]
        )
        
        unique_selling_point = st.text_area("Unique Selling Point", height=100)
    
    # Create comprehensive prompt
    if product_name:
        comprehensive_prompt = f"Create a compelling {marketing_goal.lower()} campaign for '{product_name}', a {industry.lower()} product/service targeting {audience.lower()} on {content_platform}."
        if unique_selling_point:
            comprehensive_prompt += f" Unique selling point: {unique_selling_point}."
    else:
        comprehensive_prompt = f"Create a compelling {marketing_goal.lower()} campaign for a {industry.lower()} product/service targeting {audience.lower()} on {content_platform}."
        if unique_selling_point:
            comprehensive_prompt += f" Unique selling point: {unique_selling_point}."
    
    # Generate button
    if st.button("Generate Complete Package"):
        if comprehensive_prompt:
            with st.spinner("Generating complete marketing package... This may take a minute or two."):
                try:
                    # Generate complete package
                    package = pipeline.generate_complete_marketing_package(comprehensive_prompt)
                
                    # Display results
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    
                    # Text content
                    st.subheader("📝 Marketing Content")
                    st.write(package["text"])
                    
                    # Image
                    st.subheader("🖼️ Marketing Image")
                    image_data = base64.b64decode(package["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, use_column_width=True)
                    
                    # Download image button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Download Image",
                        data=buf.getvalue(),
                        file_name="marketing_image.png",
                        mime="image/png"
                    )
                    
                    # Analysis results
                    st.subheader("📊 Content Analysis")
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["Sentiment & Emotion", "Entities", "Visualization"])
                    
                    with tab1:
                        # Sentiment analysis
                        sentiment = package["analysis"]["sentiment"]
                        st.markdown(f"**Sentiment**: {sentiment['label']} (Score: {sentiment['score']:.2f})")
                        
                        # Emotion analysis
                        emotions = package["analysis"]["emotion"]
                        
                        # Create DataFrame for emotions
                        emotions_df = pd.DataFrame({
                            'Emotion': list(emotions.keys()),
                            'Score': list(emotions.values())
                        })
                        emotions_df = emotions_df.sort_values('Score', ascending=False)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(x='Score', y='Emotion', data=emotions_df, palette='viridis', ax=ax)
                        ax.set_xlim(0, 1)
                        ax.set_title('Emotion Distribution')
                        st.pyplot(fig)
                    
                    with tab2:
                        # Entity analysis
                        entities = package["analysis"]["entities"]
                        
                        if entities and "entity_counts" in entities and entities["entity_counts"]:
                            entity_counts = entities["entity_counts"]
                            
                            # Create DataFrame for entities
                            entities_df = pd.DataFrame({
                                'Entity Type': list(entity_counts.keys()),
                                'Count': list(entity_counts.values())
                            })
                            entities_df = entities_df.sort_values('Count', ascending=False)
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.barplot(x='Count', y='Entity Type', data=entities_df, palette='muted', ax=ax)
                            ax.set_title('Entity Types Mentioned')
                            st.pyplot(fig)
                            
                            # Display marketing categories
                            st.subheader("Marketing Focused Entities")
                            marketing_categories = entities.get("marketing_categories", {})
                            
                            for category, items in marketing_categories.items():
                                if items:
                                    st.write(f"**{category}**: {', '.join(items)}")
                        else:
                            st.info("No entities detected in the generated content.")
                    
                    with tab3:
                        # Display the visualization if available
                        if package["visualization"]:
                            st.image(f"data:image/png;base64,{package['visualization']}")
                        else:
                            st.info("Visualization not available.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Summary and recommendations
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.subheader("📋 Summary & Recommendations")
                    
                    # Extract dominant emotion
                    if emotions_df is not None and not emotions_df.empty:
                        dominant_emotion = emotions_df.iloc[0]['Emotion']
                        emotion_score = emotions_df.iloc[0]['Score']
                        
                        st.markdown(f"✅ **Primary Emotion**: The content primarily conveys **{dominant_emotion}** (score: {emotion_score:.2f})")
                    
                    # Sentiment insight
                    if sentiment:
                        if sentiment['score'] > 0.7:
                            st.markdown("✅ **Sentiment**: The content has a very positive tone, which is excellent for engagement.")
                        elif sentiment['score'] > 0.5:
                            st.markdown("✅ **Sentiment**: The content has a positive tone, good for audience reception.")
                        else:
                            st.markdown("⚠️ **Sentiment**: The content has a neutral or negative tone. Consider revising for a more positive impact.")
                    
                    # Entity recommendations
                    if entities and "marketing_categories" in entities:
                        categories = entities["marketing_categories"]
                        
                        if not categories.get("PRODUCT", []) and not categories.get("BRAND", []):
                            st.markdown("⚠️ **Product Focus**: Consider adding more specific product or brand mentions.")
                        else:
                            st.markdown("✅ **Product Focus**: Good product/brand visibility in the content.")
                        
                        if not categories.get("AUDIENCE", []):
                            st.markdown("⚠️ **Target Audience**: Consider making the target audience more explicit in the content.")
                    
                    # Platform-specific recommendations
                    st.markdown(f"📱 **Platform Optimization** ({content_platform}):")
                    if content_platform == "Instagram":
                        st.markdown("- Use more visual descriptors to complement the image")
                        st.markdown("- Consider adding relevant hashtags")
                        st.markdown("- Keep text concise and engaging")
                    elif content_platform == "LinkedIn":
                        st.markdown("- Add more professional context")
                        st.markdown("- Include industry statistics or data points")
                        st.markdown("- Emphasize business value and professional benefits")
                    elif content_platform == "Twitter":
                        st.markdown("- Shorten the content to fit Twitter's style")
                        st.markdown("- Make it more punchy and direct")
                        st.markdown("- Consider adding relevant hashtags")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error generating complete package: {str(e)}")
        else:
            st.error("Please enter a product name and complete the form.")

# Footer
st.markdown("---")
st.markdown("<p class='info-text' style='text-align: center;'>Digital Marketing AI Platform • Powered by AI</p>", unsafe_allow_html=True) 