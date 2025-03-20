import nltk
import numpy as np
import re
from typing import List, Tuple, Dict, Any
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    confusion_matrix, 
    classification_report
)

class SentimentAnalyzer:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Initialize models
        self.transformer_pipeline = pipeline("sentiment-analysis")
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def hybrid_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Multi-model sentiment classification with improved scoring
        """
        try:
            # Preprocess text
            preprocessed_text = self.preprocess_text(text)
            
            # Transformer-based sentiment
            transformer_result = self.transformer_pipeline(text)[0]
            
            # Lexicon-based sentiment
            lexicon_result = self.sia.polarity_scores(text)
            
            # Advanced scoring mechanism with more nuanced approach
            transformer_score = transformer_result['score']
            lexicon_score = lexicon_result['compound']
            
            # Weighted hybrid scoring with dynamic thresholds
            final_score = (
                0.6 * transformer_score + 
                0.4 * (lexicon_score + 1) / 2
            )
            
            # More granular sentiment classification
            if final_score > 0.7:
                sentiment_label = 'POSITIVE'
            elif final_score < 0.3:
                sentiment_label = 'NEGATIVE'
            else:
                sentiment_label = 'NEUTRAL'
            
            return {
                'text': text,
                'final_sentiment': {
                    'label': sentiment_label,
                    'score': final_score
                },
                'detailed_scores': {
                    'transformer_score': transformer_score,
                    'lexicon_score': lexicon_score
                }
            }
        
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'text': text,
                'final_sentiment': {
                    'label': 'NEUTRAL',
                    'score': 0.5
                },
                'detailed_scores': {}
            }

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score with advanced tokenization
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            # Tokenize reference and hypothesis
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
            
            # Calculate BLEU score
            bleu = sentence_bleu([reference_tokens], hypothesis_tokens)
            return bleu
        except Exception as e:
            print(f"BLEU score calculation error: {e}")
            return 0.0

    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Advanced METEOR score calculation
        """
        try:
            from nltk.translate.meteor_score import single_meteor_score
            
            # Preprocess and tokenize
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
            
            # Calculate METEOR score
            meteor = single_meteor_score(reference_tokens, hypothesis_tokens)
            return meteor
        except Exception as e:
            print(f"METEOR score calculation error: {e}")
            return 0.0

    def evaluate_model(self, dataset: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with dynamic metrics
        """
        try:
            # Separate texts and true labels
            texts, true_labels = zip(*dataset)
            
            # Predict labels with more context-aware approach
            predicted_labels = []
            for text in texts:
                sentiment_result = self.hybrid_sentiment_analysis(text)
                predicted_labels.append(sentiment_result['final_sentiment']['label'])
            
            # Calculate performance metrics
            metrics = {
                'precision': precision_score(true_labels, predicted_labels, average='weighted'),
                'recall': recall_score(true_labels, predicted_labels, average='weighted'),
                'f1_score': f1_score(true_labels, predicted_labels, average='weighted'),
                'accuracy': accuracy_score(true_labels, predicted_labels)
            }
            
            # Calculate additional scores for each text
            bleu_scores = [
                self.calculate_bleu(true, pred) 
                for true, pred in zip(true_labels, predicted_labels)
            ]
            
            meteor_scores = [
                self.calculate_meteor(true, pred) 
                for true, pred in zip(true_labels, predicted_labels)
            ]
            
            # Add average BLEU and METEOR scores
            metrics['avg_bleu_score'] = np.mean(bleu_scores)
            metrics['avg_meteor_score'] = np.mean(meteor_scores)
            
            # Confusion matrix with label order
            label_order = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            conf_matrix = confusion_matrix(
                true_labels, 
                predicted_labels, 
                labels=label_order
            )
            
            # Detailed classification report
            class_report = classification_report(
                true_labels, 
                predicted_labels,
                target_names=label_order
            )
            
            return {
                'metrics': metrics,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report
            }
        
        except Exception as e:
            print(f"Model evaluation error: {e}")
            return {
                'metrics': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'accuracy': 0.0,
                    'avg_bleu_score': 0.0,
                    'avg_meteor_score': 0.0
                },
                'confusion_matrix': None,
                'classification_report': ''
            }