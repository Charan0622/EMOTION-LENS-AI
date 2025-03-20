from transformers import pipeline

class EmotionDetector:
    EMOTION_CATEGORIES = [
        'joy', 'sadness', 'anger', 'fear', 
        'surprise', 'love', 'neutral'
    ]

    def __init__(self, model_name='j-hartmann/emotion-english-distilroberta-base'):
        """
        Initialize advanced emotion detection
        
        Args:
            model_name (str): Pretrained emotion detection model
        """
        try:
            self.emotion_pipeline = pipeline(
                "text-classification", 
                model=model_name, 
                top_k=None
            )
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            print("Using fallback emotion detection method.")
            self.emotion_pipeline = self._fallback_emotion_detection()

    def _fallback_emotion_detection(self):
        """
        Fallback method for emotion detection if model loading fails
        
        Returns:
            A simple emotion detection function
        """
        def simple_emotion_detection(text):
            # Basic emotion detection using keywords
            emotions = {
                'joy': ['happy', 'excited', 'delighted', 'glad'],
                'sadness': ['sad', 'depressed', 'unhappy', 'gloomy'],
                'anger': ['angry', 'furious', 'irritated', 'mad'],
                'fear': ['scared', 'afraid', 'terrified', 'anxious'],
                'surprise': ['surprised', 'shocked', 'amazed'],
                'love': ['love', 'adore', 'cherish', 'fond']
            }
            
            # Convert text to lowercase for easier matching
            text_lower = text.lower()
            
            # Detect emotions
            detected_emotions = []
            for emotion, keywords in emotions.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_emotions.append({
                        'label': emotion,
                        'score': 0.5  # Default confidence score
                    })
            
            # If no emotions detected, return neutral
            if not detected_emotions:
                detected_emotions = [{'label': 'neutral', 'score': 1.0}]
            
            return [detected_emotions]
        
        return simple_emotion_detection

    def detect_emotions(self, text, top_n=3):
        """
        Advanced emotion detection with multiple features
        
        Args:
            text (str): Input text
            top_n (int): Number of top emotions to return
        
        Returns:
            dict: Emotion detection results
        """
        try:
            # Detect emotions
            emotions = self.emotion_pipeline(text)[0]
            
            # Sort and filter emotions
            sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
            top_emotions = sorted_emotions[:top_n]
            
            return {
                'input_text': text,
                'top_emotions': [
                    {
                        'emotion': emotion['label'],
                        'probability': emotion['score']
                    } for emotion in top_emotions
                ],
                'dominant_emotion': top_emotions[0]['label'] if top_emotions else 'Unknown'
            }
        
        except Exception as e:
            print(f"Emotion detection error: {e}")
            
            # Fallback to simple keyword-based detection
            fallback_emotions = self._fallback_emotion_detection()(text)[0]
            top_emotions = sorted(fallback_emotions, key=lambda x: x['score'], reverse=True)[:top_n]
            
            return {
                'input_text': text,
                'top_emotions': [
                    {
                        'emotion': emotion['label'],
                        'probability': emotion['score']
                    } for emotion in top_emotions
                ],
                'dominant_emotion': top_emotions[0]['label'] if top_emotions else 'Unknown'
            }