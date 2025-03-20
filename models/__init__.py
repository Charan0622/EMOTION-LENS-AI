import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline
)

class ModelManager:
    """
    Centralized model management class for loading and initializing 
    pre-trained models for sentiment analysis and emotion detection
    """
    
    # Define model configurations
    MODELS = {
        'sentiment': {
            'roberta': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment',
                'description': 'RoBERTa model fine-tuned on Twitter sentiment data'
            },
            'bert': {
                'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'description': 'Multilingual BERT for sentiment analysis'
            }
        },
        'emotion': {
            'goemotion': {
                'model_name': 'google/bert-base-uncased-goemotion',
                'description': 'BERT model trained on GoEmotions dataset'
            },
            'xlnet': {
                'model_name': 'joeddav/xlnet-base-cased-emotion',
                'description': 'XLNet model for emotion classification'
            }
        }
    }

    @classmethod
    def load_sentiment_model(cls, model_type='roberta'):
        """
        Load a pre-trained sentiment analysis model
        
        Args:
            model_type (str): Type of sentiment model to load
        
        Returns:
            tuple: Loaded model and tokenizer
        """
        try:
            model_config = cls.MODELS['sentiment'].get(model_type)
            if not model_config:
                raise ValueError(f"Unsupported sentiment model: {model_type}")
            
            model_name = model_config['model_name']
            
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create inference pipeline
            sentiment_pipeline = pipeline(
                'sentiment-analysis', 
                model=model, 
                tokenizer=tokenizer
            )
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'pipeline': sentiment_pipeline,
                'description': model_config['description']
            }
        
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            return None

    @classmethod
    def load_emotion_model(cls, model_type='goemotion'):
        """
        Load a pre-trained emotion detection model
        
        Args:
            model_type (str): Type of emotion model to load
        
        Returns:
            dict: Loaded model resources
        """
        try:
            model_config = cls.MODELS['emotion'].get(model_type)
            if not model_config:
                raise ValueError(f"Unsupported emotion model: {model_type}")
            
            model_name = model_config['model_name']
            
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create emotion detection pipeline
            emotion_pipeline = pipeline(
                'text-classification', 
                model=model, 
                tokenizer=tokenizer,
                top_k=None  # Return all emotion probabilities
            )
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'pipeline': emotion_pipeline,
                'description': model_config['description']
            }
        
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            return None

    @staticmethod
    def check_cuda_availability():
        """
        Check and report CUDA availability for GPU acceleration
        
        Returns:
            bool: Whether CUDA is available
        """
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA is available. Current device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available. Models will run on CPU.")
        
        return cuda_available

    @classmethod
    def get_available_models(cls):
        """
        Retrieve all available models
        
        Returns:
            dict: Available sentiment and emotion models
        """
        return {
            'sentiment_models': list(cls.MODELS['sentiment'].keys()),
            'emotion_models': list(cls.MODELS['emotion'].keys())
        }

# Example usage demonstration
def model_demo():
    # Check CUDA availability
    ModelManager.check_cuda_availability()
    
    # Load sentiment model
    sentiment_model = ModelManager.load_sentiment_model('roberta')
    print("Sentiment Model Description:", sentiment_model['description'])
    
    # Load emotion model
    emotion_model = ModelManager.load_emotion_model('goemotion')
    print("Emotion Model Description:", emotion_model['description'])
    
    # Get available models
    available_models = ModelManager.get_available_models()
    print("Available Models:", available_models)

# Uncomment to run demo
# if __name__ == "__main__":
#     model_demo()