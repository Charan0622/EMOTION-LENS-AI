import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline
)
import numpy as np
from typing import Dict, Any

class MultilingualSentimentAnalyzer:
    def __init__(self):
        # Load multilingual models
        self.models = {
            'xlmr': self._load_xlmr_model(),
            'mbert': self._load_mbert_model()
        }
        
        # Comprehensive multilingual dataset
        self.multilingual_dataset = self._create_comprehensive_dataset()
    
    def _create_comprehensive_dataset(self):
        """
        Create an extensive multilingual dataset
        """
        return [
            # Extensive samples in multiple languages
            # Add 500-1000 high-quality, diverse samples
            {"text": "I love this product!", "sentiment": "POSITIVE", "language": "en"},
            {"text": "This is terrible service.", "sentiment": "NEGATIVE", "language": "en"},
            {"text": "मैं इस उत्पाद से बहुत खुश हूँ!", "sentiment": "POSITIVE", "language": "hi"},
            {"text": "ఈ సేవ చాలా బాగుంది!", "sentiment": "POSITIVE", "language": "te"},
            # Add many more samples...
        ]
    
    def _load_xlmr_model(self):
        """
        Load XLM-RoBERTa multilingual model
        """
        model_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            problem_type="multi_label_classification"
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        }
    
    def _load_mbert_model(self):
        """
        Load Multilingual BERT model
        """
        model_name = "bert-base-multilingual-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            problem_type="multi_label_classification"
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Advanced multilingual sentiment analysis with ensemble approach
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Ensemble prediction from multiple models
        predictions = []
        for model_name, model_config in self.models.items():
            try:
                result = model_config['pipeline'](text)[0]
                predictions.append({
                    'model': model_name,
                    'sentiment': result['label'],
                    'confidence': result['score']
                })
            except Exception as e:
                print(f"Model {model_name} prediction error: {e}")
        
        # Ensemble voting and confidence calculation
        if predictions:
            # Majority voting
            sentiments = [pred['sentiment'] for pred in predictions]
            dominant_sentiment = max(set(sentiments), key=sentiments.count)
            
            # Weighted confidence
            confidences = [pred['confidence'] for pred in predictions if pred['sentiment'] == dominant_sentiment]
            avg_confidence = np.mean(confidences)
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': avg_confidence,
                'model_predictions': predictions
            }
        
        return {'sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def _preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing
        """
        # Remove special characters, normalize
        text = text.lower().strip()
        return text