import os
import sys
import random
import traceback

# Dynamically add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Robust Plotly Import
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objs as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not fully installed. Using Matplotlib for visualizations.")

# Import Project Modules
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_detector import EmotionDetector
from src.language_translator import LanguageTranslator
from models.emotion_model import EmotionModel

# Enhanced Emoji Mapping
EMOTION_EMOJIS = {
    'POSITIVE': '🌞', 
    'NEGATIVE': '🌧️', 
    'NEUTRAL': '😐',
    'joy': '🎉', 
    'sadness': '😢', 
    'anger': '🔥', 
    'fear': '😱', 
    'surprise': '🤯', 
    'love': '❤️',
    'happiness': '😊',
    'excitement': '🚀'
}

def create_visual_enhancements():
    """
    Create advanced visual enhancements for the Streamlit app
    """
    st.markdown("""
    <style>
    /* Global Background and Typography */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Animated Gradient Title */
    .gradient-title {
        background: linear-gradient(45deg, #3494E6, #EC6EAD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-animation 5s ease infinite;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
    }

    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Effect for Containers */
    .glassmorphism {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.125);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }

    .glassmorphism:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }

    /* Animated Quote Scroll */
    @keyframes scrollQuotes {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    .quote-banner {
        display: inline-block;
        animation: scrollQuotes 60s linear infinite;
        padding-right: 50px;
        font-size: 1.2em;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* Button Styles */
    .stButton>button {
        background: linear-gradient(45deg, #3494E6, #EC6EAD);
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def create_scrolling_quotes():
    """
    Create an attractive, dynamic quote display
    """
    quotes = [
       "💖 Your feelings are the language of your inner self."
        "🌿 Growth begins when you embrace your emotions."
        "🌞 A heart full of gratitude shines the brightest."
        "🌸 Love yourself first, and the world will follow."
        "🌊 Let your emotions flow like a peaceful river."
        "🌠 Your dreams reflect the whispers of your soul."
        "💕 Kindness is the most beautiful emotion of all."
        "☀️ Happiness blooms where love is planted."
        "💫 Every emotion is a step toward self-discovery."
        "🍃 Let go of worries and dance with the wind."
        "🌙 Silence holds the deepest emotions."
        "🌈 Embrace your feelings; they color your world."
        "💙 A peaceful heart makes the world feel lighter."
        "🌻 You grow through what you go through."
        "🦋 Transformation begins in the heart."
        "🔥 Passion fuels the soul’s journey."
        "🏞️ Find peace in the present moment."
        "💭 Your thoughts shape your reality."
        "🧘‍♂️ A calm mind opens the door to wisdom."
        "❤️ Love heals even the deepest wounds."
        "🎶 Music is the voice of the heart."
        "🎨 Your emotions are the brushstrokes of life."
        "🍂 Change is the melody of the universe."
        "🌷 Hope grows in the darkest moments."
        "🌟 Shine like the star that you are."
        "💞 Connection makes life meaningful."
        "⏳ Every feeling has its time and place."
        "🚀 Let your emotions propel you to greatness."
        "🌙 Even the moon has its phases; so do you."
        "🌸 Love starts from within and spreads outward."
        "🌍 Your heartbeats are part of the universe’s rhythm."
        "✨ A kind word can change someone's world."
        "🎈 Let go of the past and float toward joy."
        "🍁 Accept change like trees embrace the seasons."
        "🌊 Let your fears wash away like ocean waves."
        "🔮 Your intuition speaks in emotions—listen."
        "🌅 Every sunrise brings a new chance for happiness."
        "📖 Your heart writes the most beautiful stories."
        "🦋 Emotions are the wings that let you fly."
        "💡 Light up your soul with positive thoughts."
        "🌛 The stars listen to your unspoken dreams."
        "🌱 A kind heart nurtures the garden of love."
        "🎭 Embrace every emotion—it’s part of your story."
        "🌤️ After every storm, the sun always rises."
        "🕊️ Let peace be your soul’s melody."
        "🍃 Breathe in love, breathe out worries."
        "🧡 Trust your heart; it knows the way."
        "🔥 Inner strength is born from deep emotions."
        "🎇 Your energy radiates what your heart feels."
        "💖 Love yourself like the universe loves you."
    ]
    
    # Generate unique quotes HTML
    quotes_html = f"""
    <style>
    .quotes-container {{
        width: 100%;
        height: 60px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }}

    .quote-display {{
        position: absolute;
        width: 100%;
        text-align: center;
        color: white;
        font-size: 1em;
        white-space: nowrap;
        animation: scrollQuote 20s linear infinite;
    }}

    @keyframes scrollQuote {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    </style>

    <div class="quotes-container">
        <div class="quote-display">
            {' 🌈 | '.join(quotes)} 🌈
        </div>
    </div>
    """
    
    return quotes_html

@st.cache_resource
def load_models():
    """
    Cache model initialization with error handling
    """
    try:
        sentiment_analyzer = SentimentAnalyzer()
        emotion_detector = EmotionDetector()
        language_translator = LanguageTranslator()
        return sentiment_analyzer, emotion_detector, language_translator
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None, None

def create_emotion_visualization(data, title='Emotion Analysis'):
    """
    Create interactive emotion probability visualization
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure we have data
        if df.empty:
            st.warning("No data available for visualization")
            return None
        
        # Plotly Visualization
        if PLOTLY_AVAILABLE:
            try:
                fig = go.Figure(data=[
                    go.Bar(
                        x=df['emotion'],
                        y=df['probability'],
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FDCB6E', '#6C5CE7'],
                        text=[f'{p:.2%}' for p in df['probability']],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title=f'🎨 {title}',
                    xaxis_title='Emotions',
                    yaxis_title='Probability',
                    template='plotly_white',
                    height=400,
                )
                
                return fig
            
            except Exception as plotly_err:
                st.warning(f"Visualization error: {plotly_err}")
                return None
        
        return None
    
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def create_confusion_matrix_visualization(confusion_matrix):
    """
    Create a compact confusion matrix visualization
    """
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['NEG', 'NEUT', 'POS'],
            yticklabels=['NEG', 'NEUT', 'POS'],
            cbar=False
        )
        plt.title('Confusion Matrix', fontsize=10)
        plt.xlabel('Predicted', fontsize=8)
        plt.ylabel('Actual', fontsize=8)
        plt.tight_layout()
        return plt
    except Exception as e:
        st.error(f"Confusion matrix visualization error: {e}")
        return None

def display_performance_metrics(performance_metrics):
    """
    Display comprehensive performance metrics with all requested metrics
    """
    st.markdown("### 📊 Model Performance Metrics")
    
    # Create a more comprehensive metrics display
    metrics_data = [
        {
            'Metric': 'Precision',
            'Value': f"{performance_metrics['metrics']['precision']:.4f}",
            'Description': 'Accuracy of positive predictions'
        },
        {
            'Metric': 'Recall',
            'Value': f"{performance_metrics['metrics']['recall']:.4f}",
            'Description': 'Proportion of actual positives correctly identified'
        },
        {
            'Metric': 'F1 Score',
            'Value': f"{performance_metrics['metrics']['f1_score']:.4f}",
            'Description': 'Harmonic mean of precision and recall'
        },
        {
            'Metric': 'Accuracy',
            'Value': f"{performance_metrics['metrics']['accuracy']:.4f}",
            'Description': 'Overall correct predictions'
        },
        {
            'Metric': 'BLEU Score',
            'Value': f"{performance_metrics['metrics']['avg_bleu_score']:.4f}",
            'Description': 'Text translation quality metric'
        },
        {
            'Metric': 'METEOR Score',
            'Value': f"{performance_metrics['metrics']['avg_meteor_score']:.4f}",
            'Description': 'Semantic similarity metric'
        }
    ]
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Detailed Metrics")
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Classification Report")
        st.text(performance_metrics['classification_report'])

def main():
    # Page Configuration
    st.set_page_config(
        page_title="EMOTION LENS AI", 
        page_icon="🌐", 
        layout="wide"
    )

    # Apply Visual Enhancements
    create_visual_enhancements()

    # Scrolling Quotes Banner
    st.markdown(create_scrolling_quotes(), unsafe_allow_html=True)

    # Animated Gradient Title
    st.markdown(
        '<div class="gradient-title">🌐 EMOTION LENS AI</div>', 
        unsafe_allow_html=True
    )
    
    # Tagline
    st.markdown(
        '<div style="text-align: center; color: #2c3e50; font-size: 1.2em; margin-top: -20px; margin-bottom: 20px; font-style:italic;">Multilingual Sentiment Unveiled</div>', 
        unsafe_allow_html=True
    )
    # Load models
    sentiment_analyzer, emotion_detector, language_translator = load_models()
    
    # Comprehensive test dataset
    test_data = [
        # Positive Sentiments
        ("I absolutely love this product!", "POSITIVE"),
        ("What an amazing and inspiring day!", "POSITIVE"),
        ("The service was exceptional and made me very happy.", "POSITIVE"),
        
        # Negative Sentiments
        ("This is the worst experience ever.", "NEGATIVE"),
        ("Absolutely terrible customer service.", "NEGATIVE"),
        ("I'm deeply disappointed and frustrated.", "NEGATIVE"),
        
        # Neutral Sentiments
        ("The service was neither good nor bad.", "NEUTRAL"),
        ("I'm not sure how I feel about this.", "NEUTRAL"),
        ("It was an okay experience, nothing special.", "NEUTRAL")
    ]
    
    # Glassmorphism Container for Input
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    
    # Multilingual Text Input
    user_input = st.text_area(
        "Enter text in any language...", 
        height=150, 
        help="Share your thoughts in any language to uncover emotional insights 🌈"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Multilingual Analysis Button
    if st.button("Analyze Multilingual Text 🔮", type="primary"):
        if not user_input:
            st.warning("Please enter some text to analyze 📝")
            return
        
        try:
            # Translate text to English
            translation_result = language_translator.translate_to_english(user_input)
            translated_text = translation_result['translated_text']
            
            # Glassmorphism Container for Results
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            
            # Translation Details
            st.subheader("🌍 Translation Details")
            translation_df = pd.DataFrame([
                {"Attribute": "Original Language", "Value": f"{translation_result['original_language_name']} {translation_result['original_language_emoji']}"},
                {"Attribute": "Translation Confidence", "Value": f"{translation_result['translation_confidence']:.2%}"},
                {"Attribute": "Original Text", "Value": user_input},
                {"Attribute": "Translated Text", "Value": translated_text}
            ])
            st.dataframe(translation_df, use_container_width=True)
            
            # Perform sentiment and emotion analysis
            sentiment_result = sentiment_analyzer.hybrid_sentiment_analysis(translated_text)
            emotion_result = emotion_detector.detect_emotions(translated_text)
            
            # Performance Metrics
            performance_metrics = sentiment_analyzer.evaluate_model(test_data)
            
            # Emotion and Sentiment Details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🎭 Dominant Emotion")
                dominant_emotion = emotion_result['dominant_emotion']
                st.metric(
                    "Dominant Emotion", 
                    f"{dominant_emotion} {EMOTION_EMOJIS.get(dominant_emotion.lower(), '😐')}"
                )
                st.metric(
                    "Confidence Score", 
                    f"{sentiment_result['final_sentiment']['score']:.2%}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("🌈 Emotional Spectrum")
                for emotion in emotion_result['top_emotions'][:3]:
                    st.metric(
                        f"{emotion['emotion'].capitalize()} {EMOTION_EMOJIS.get(emotion['emotion'].lower(), '😐')}", 
                        f"{emotion['probability']:.2%}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Emotion Probability Visualization
            st.subheader("📈 Emotion Probability Distribution")
            emotions_df = pd.DataFrame(emotion_result['top_emotions'])
            
            fig = create_emotion_visualization(
                emotions_df, 
                title='Emotion Probability Distribution'
            )
            
            if PLOTLY_AVAILABLE and fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Model Performance Section
            st.subheader("🔬 Model Performance and Evaluation")
            display_performance_metrics(performance_metrics)
            
            # Confusion Matrix Visualization
            if performance_metrics['confusion_matrix'] is not None:
                st.subheader("🌐 Confusion Matrix")
                conf_matrix_plt = create_confusion_matrix_visualization(
                    performance_metrics['confusion_matrix']
                )
                
                if conf_matrix_plt:
                    st.pyplot(conf_matrix_plt)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Analysis Error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()