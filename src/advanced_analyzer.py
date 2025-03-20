from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

class AdvancedAnalyzer:
    @staticmethod
    def generate_word_cloud(text):
        """
        Generate a word cloud from the input text
        
        Args:
            text (str): Input text
        
        Returns:
            str: Base64 encoded word cloud image
        """
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white'
        ).generate(text)
        
        # Convert to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    @staticmethod
    def extract_key_phrases(text):
        """
        Extract key phrases and their sentiment
        
        Args:
            text (str): Input text
        
        Returns:
            list: Key phrases with their sentiments
        """
        # Use TextBlob for phrase extraction and sentiment analysis
        blob = TextBlob(text)
        
        # Extract noun phrases
        noun_phrases = blob.noun_phrases
        
        # Analyze sentiment for each phrase
        phrase_sentiments = []
        for phrase in noun_phrases:
            sentiment = TextBlob(phrase).sentiment.polarity
            phrase_sentiments.append({
                'phrase': phrase,
                'sentiment': sentiment,
                'sentiment_label': (
                    'Positive' if sentiment > 0 else 
                    'Negative' if sentiment < 0 else 
                    'Neutral'
                )
            })
        
        return sorted(phrase_sentiments, key=lambda x: abs(x['sentiment']), reverse=True)[:5]

    @staticmethod
    def emotion_intensity_analysis(emotions):
        """
        Provide insights based on emotion intensities
        
        Args:
            emotions (list): List of detected emotions
        
        Returns:
            dict: Emotion intensity insights
        """
        # Calculate emotional complexity
        emotion_intensities = {
            'emotional_range': max(e['probability'] for e in emotions) - 
                               min(e['probability'] for e in emotions),
            'dominant_emotion': max(emotions, key=lambda x: x['probability']),
            'emotional_complexity': len(emotions)
        }
        
        # Generate emotional insight
        if emotion_intensities['emotional_range'] > 0.5:
            insight = "You're experiencing a complex emotional state with significant variations."
        elif emotion_intensities['emotional_complexity'] > 2:
            insight = "Your emotions are nuanced and multifaceted."
        else:
            insight = "Your emotional state appears relatively consistent."
        
        emotion_intensities['emotional_insight'] = insight
        
        return emotion_intensities