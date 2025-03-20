import os
import sys
import re
import streamlit as st
import langdetect
from deep_translator import GoogleTranslator

class LanguageTranslator:
    def __init__(self):
        """
        Initialize language translation and detection services
        """
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'pt': 'Portuguese',
            'unknown': 'Unknown'
        }
        
        self.language_emojis = {
            'en': '🇬🇧',
            'hi': '🇮🇳',
            'te': '🇮🇳',
            'ta': '🇮🇳',
            'ml': '🇮🇳',
            'kn': '🇮🇳',
            'es': '🇪🇸',
            'fr': '🇫🇷',
            'de': '🇩🇪',
            'it': '🇮🇹',
            'ru': '🇷🇺',
            'zh': '🇨🇳',
            'ar': '🇸🇦',
            'pt': '🇵🇹',
            'unknown': '🌍'
        }
    
    def detect_language(self, text):
        """
        Advanced language detection with multiple fallback mechanisms
        """
        try:
            # Primary detection using langdetect
            detected_lang = langdetect.detect(text)
            
            # Validate and normalize language code
            if detected_lang in self.language_names:
                return detected_lang
            
            # Additional script-based detection
            language_ranges = {
                'hi': ('\u0900', '\u097F'),   # Devanagari script
                'te': ('\u0C00', '\u0C7F'),   # Telugu script
                'ta': ('\u0B80', '\u0BFF'),   # Tamil script
                'ml': ('\u0D00', '\u0D7F'),   # Malayalam script
                'kn': ('\u0C80', '\u0CFF'),   # Kannada script
                'zh': ('\u4E00', '\u9FFF')    # Chinese characters
            }
            
            for lang, (start, end) in language_ranges.items():
                if any(start <= char <= end for char in text):
                    return lang
            
            # Fallback to English
            return 'en'
        
        except Exception as e:
            st.warning(f"Language detection error: {e}")
            return 'en'
    
    def translate_to_english(self, text):
        """
        Translate text to English with comprehensive error handling
        """
        try:
            # Detect source language
            source_lang = self.detect_language(text)
            
            # Attempt translation with deep-translator
            try:
                translated_text = GoogleTranslator(
                    source=source_lang, 
                    target='en'
                ).translate(text)
                translation_confidence = 0.9
            except Exception as primary_error:
                st.warning(f"Primary translation failed: {primary_error}")
                
                # Fallback translation method
                try:
                    translated_text = GoogleTranslator(
                        source='auto', 
                        target='en'
                    ).translate(text)
                    translation_confidence = 0.7
                except Exception as fallback_error:
                    st.error(f"Fallback translation failed: {fallback_error}")
                    translated_text = text
                    translation_confidence = 0.5
            
            # Prepare translation result
            return {
                'original_text': text,
                'original_language': source_lang,
                'original_language_name': self.language_names.get(source_lang, 'Unknown'),
                'original_language_emoji': self.language_emojis.get(source_lang, '🌍'),
                'translated_text': translated_text,
                'translation_confidence': translation_confidence
            }
        
        except Exception as e:
            st.error(f"Comprehensive translation error: {e}")
            return {
                'original_text': text,
                'original_language': 'unknown',
                'original_language_name': 'Unknown',
                'original_language_emoji': '🌍',
                'translated_text': text,
                'translation_confidence': 0.5
            }