# Digital Marketing AI Platform 🚀

A comprehensive AI-powered platform for digital marketing content generation, image creation, sentiment analysis, and trend prediction.

## Features

- **Text Generation**: Create engaging marketing content for various platforms and industries
- **Image Generation**: Generate professional marketing images based on text prompts
- **Emotion Detection**: Analyze emotions in marketing text (joy, sadness, anger, fear, surprise)
- **Sentiment Analysis**: Determine the sentiment of marketing content
- **Named Entity Recognition**: Extract and categorize entities in marketing text
- **Trend Analysis**: Predict marketing trends from historical data

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (but not required)
- Internet connection for model downloads

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd digital-marketing-ai
```

2. Run the setup script to prepare the environment:
```bash
python setup.py
```

This will:
- Install all required dependencies
- Download necessary models
- Check for GPU availability
- Create necessary directories

## Usage

### Running the Web Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

This will launch a browser window with the interactive UI where you can:
- Generate marketing content
- Create marketing images
- Analyze marketing trends
- Create complete marketing packages

### Using the API

You can also use the platform programmatically by importing the `MarketingAIPipeline` class:

```python
from marketing_ai_pipeline import MarketingAIPipeline

# Initialize the pipeline
pipeline = MarketingAIPipeline()
pipeline.initialize_models()

# Generate marketing text
text = pipeline.generate_marketing_text("Create a social media post for a tech product")

# Generate marketing image
image = pipeline.generate_marketing_image("Tech product advertisement")
image.save("marketing_image.png")

# Analyze sentiment
sentiment = pipeline.analyze_sentiment(text)
print(f"Sentiment: {sentiment}")

# Analyze emotion
emotion = pipeline.analyze_emotion(text)
print(f"Emotion: {emotion}")

# Extract entities
entities = pipeline.extract_entities(text)
print(f"Entities: {entities}")
```

## Data Input

The platform works with the following data formats:

- **Text Generation**: Uses the Reviews.csv dataset for fine-tuning
- **Trend Analysis**: Uses the Social_Media_Advertising.csv dataset for training
- **Campaign Analysis**: Uses the digital_marketing_campaigns_smes.csv dataset for insights

You can also upload your own data in the web interface.

## Model Details

The platform uses the following AI models:

- **Text Generation**: GPT-2 (fine-tuned on marketing data)
- **Image Generation**: Stable Diffusion 2.1
- **Sentiment Analysis**: BERT-based multilingual sentiment model
- **Emotion Classification**: DistilRoBERTa emotion classification model
- **Named Entity Recognition**: SpaCy's en_core_web_sm model
- **Trend Prediction**: Custom neural network

## Project Structure

- `marketing_ai_pipeline.py`: Main AI pipeline implementation
- `app.py`: Streamlit web application
- `setup.py`: Environment setup script
- `requirements.txt`: Dependencies list
- `models/`: Directory for storing trained models
- `data/`: Contains training datasets
- `generated_images/`: Storage for generated marketing images

## Customizing for Your Brand

The platform can be customized to your specific brand by:
1. Fine-tuning the text model on your brand's content
2. Training the trend model on your marketing performance data
3. Adjusting the image generation prompts to match your brand style

## Troubleshooting

**Models running slowly?**
- Check GPU availability with `python check_gpu.py`
- For CPU-only environments, reduce model sizes in the pipeline

**Out of memory errors?**
- Reduce batch sizes during fine-tuning
- Use smaller model variants

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Uses Hugging Face Transformers for NLP models
- Uses Stable Diffusion for image generation
- Datasets from public marketing data collections

## Future Enhancements

- Multi-language support
- Video generation capabilities
- A/B testing framework
- Campaign scheduling and automation
- Enhanced personalization features 