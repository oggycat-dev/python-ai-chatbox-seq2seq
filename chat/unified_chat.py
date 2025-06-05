#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: d:\python-ai-chatbox-seq2seq\chat\unified_chat.py

"""
Unified Chat Interface for Seq2Seq Chatbot
------------------------------------------
Hỗ trợ cả tiếng Anh và tiếng Việt

Features:
- Tự động phát hiện và sử dụng mô hình tiếng Việt nếu có
- Dùng mô hình tiếng Anh làm dự phòng
- Hỗ trợ chuyển đổi giữa tiếng Anh và tiếng Việt
- Cung cấp các câu trả lời đã định nghĩa sẵn cho các câu hỏi thông dụng
"""

import torch
import sys
import os
import re
import nltk
import argparse
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from src.model import Seq2Seq, Encoder, Decoder
from src.preprocess import normalize_string, normalize_vietnamese


class ChatbotModel:
    """Wrapper class for a chatbot model with vocabulary and model parameters."""
    
    def __init__(self, model_path, input_word2idx_path, input_idx2word_path,
                 target_word2idx_path, target_idx2word_path, hid_dim=128):
        self.model_path = model_path
        self.input_word2idx_path = input_word2idx_path
        self.input_idx2word_path = input_idx2word_path
        self.target_word2idx_path = target_word2idx_path
        self.target_idx2word_path = target_idx2word_path
        self.hid_dim = hid_dim
        
        self.input_word2idx = None
        self.input_idx2word = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.model = None
        self.input_dim = None
        self.output_dim = None
        self.emb_dim = None
        
    def load(self):
        """Load model and vocabulary files."""
        try:
            self.input_word2idx = torch.load(self.input_word2idx_path)
            self.input_idx2word = torch.load(self.input_idx2word_path)
            self.target_word2idx = torch.load(self.target_word2idx_path)
            self.target_idx2word = torch.load(self.target_idx2word_path)
            
            # Load state dict
            state_dict = torch.load(self.model_path)
            
            # Get dimensions from state dict
            encoder_embedding_shape = state_dict['encoder.embedding.weight'].shape
            decoder_embedding_shape = state_dict['decoder.embedding.weight'].shape
            
            # Initialize model with correct dimensions
            self.input_dim = encoder_embedding_shape[0]
            self.output_dim = decoder_embedding_shape[0]
            self.emb_dim = encoder_embedding_shape[1]
            
            # Check for dimension mismatch
            if self.input_dim != len(self.input_word2idx):
                print(f"Warning: Model expects {self.input_dim} input words, but dictionary has {len(self.input_word2idx)} words")
            
            # Create model
            encoder = Encoder(self.input_dim, self.emb_dim, self.hid_dim)
            decoder = Decoder(self.output_dim, self.emb_dim, self.hid_dim)
            self.model = Seq2Seq(encoder, decoder)
            
            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            return True
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading model: {e}")
            return False
    
    def decode(self, sequence):
        """Convert a sequence of indices to text."""
        words = []
        for idx in sequence:
            if idx == self.target_word2idx['<EOS>']:
                break
            words.append(self.target_idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def predict(self, tokens, max_length=50):
        """Predict a response for the given tokens."""
        # Filter valid tokens
        valid_tokens = []
        for token in tokens:
            if token in self.input_word2idx:
                valid_tokens.append(token)
        
        if not valid_tokens:
            return None
        
        # Convert tokens to indices
        indices = [self.input_word2idx[token] for token in valid_tokens] + [self.input_word2idx['<EOS>']]
        
        # Check indices validity
        max_idx = self.input_dim - 1
        safe_indices = []
        for idx in indices:
            if 0 <= idx <= max_idx:
                safe_indices.append(idx)
        
        if not safe_indices:
            return None
        
        # Convert to tensor
        input_tensor = torch.tensor(safe_indices).unsqueeze(1)  # (seq_len, batch_size=1)
        
        with torch.no_grad():
            # Encoder
            hidden = self.model.encoder(input_tensor)
            
            # Decoder - starting with first token as EOS (representing START)
            trg_idx = [self.target_word2idx['<EOS>']]
            
            for _ in range(max_length):
                # Prepare input tensor for decoder
                trg_tensor = torch.tensor([trg_idx[-1]], dtype=torch.long)
                output, hidden = self.model.decoder(trg_tensor, hidden)
                
                # Get token with highest probability
                pred_token = output.argmax(1).item()
                trg_idx.append(pred_token)
                
                # Stop if EOS is encountered
                if pred_token == self.target_word2idx['<EOS>']:
                    break
            
            response = self.decode(trg_idx[1:])  # Skip the first token (START)
            return response


class UnifiedChatbot:
    """Unified chatbot supporting both English and Vietnamese."""
    
    def __init__(self):
        # Paths
        self.models_dir = os.path.join(base_dir, 'models')
        
        # Models
        self.english_model = None
        self.vietnamese_model = None
        self.current_model = None
        
        # Translation dictionaries
        self.vietnamese_to_english_dict = self.create_vi2en_dict()
        self.english_to_vietnamese_dict = self.create_en2vi_dict()
        
        # Common phrases with predefined responses
        self.common_phrases = self.create_common_phrases()
        
        # Auto-detect language
        self.using_vietnamese = False
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load available models."""
        # Try to load Vietnamese model first
        self.vietnamese_model = ChatbotModel(
            model_path=os.path.join(self.models_dir, 'vietnamese_chatbot.pt'),
            input_word2idx_path=os.path.join(self.models_dir, 'vietnamese_input_word2idx.pt'),
            input_idx2word_path=os.path.join(self.models_dir, 'vietnamese_input_idx2word.pt'),
            target_word2idx_path=os.path.join(self.models_dir, 'vietnamese_target_word2idx.pt'),
            target_idx2word_path=os.path.join(self.models_dir, 'vietnamese_target_idx2word.pt')
        )
        
        vietnamese_available = self.vietnamese_model.load()
        
        # Always load English model as fallback
        self.english_model = ChatbotModel(
            model_path=os.path.join(self.models_dir, 'chatbot_new.pt'),
            input_word2idx_path=os.path.join(self.models_dir, 'input_word2idx.pt'),
            input_idx2word_path=os.path.join(self.models_dir, 'input_idx2word.pt'),
            target_word2idx_path=os.path.join(self.models_dir, 'target_word2idx.pt'),
            target_idx2word_path=os.path.join(self.models_dir, 'target_idx2word.pt')
        )
        
        english_available = self.english_model.load()
        
        # Set default model
        if vietnamese_available:
            self.current_model = self.vietnamese_model
            self.using_vietnamese = True
            print("Đã tải mô hình tiếng Việt!")
        elif english_available:
            self.current_model = self.english_model
            self.using_vietnamese = False
            print("Loaded English model!")
        else:
            print("Không thể tải bất kỳ mô hình nào. Vui lòng kiểm tra lại thư mục models.")
            sys.exit(1)
    
    def create_vi2en_dict(self):
        """Create Vietnamese to English translation dictionary."""
        return {
            "xin": "hello", "chào": "hello", "chao": "hello",
            "tên": "name", "ten": "name", "là": "is", "la": "is",
            "gì": "what", "gi": "what", "bạn": "you", "ban": "you",
            "ai": "who", "tạo": "create", "tao": "create",
            "ra": "up", "làm": "do", "lam": "do",
            "thế": "how", "the": "how", "nào": "how", "nao": "how",
            "khỏe": "fine", "khoe": "fine", "không": "no", "khong": "no",
            "có": "yes", "co": "yes", "biết": "know", "biet": "know",
            "cảm": "thank", "cam": "thank", "ơn": "thank", "on": "thank",
            "tuổi": "age", "tuoi": "age", "được": "can", "duoc": "can",
            "giúp": "help", "giup": "help", "hiểu": "understand", "hieu": "understand",
            "thích": "like", "thich": "like", "đẹp": "beautiful", "dep": "beautiful",
            "từ": "from", "tu": "from", "đâu": "where", "dau": "where",
            "tạm": "good", "tam": "good", "biệt": "bye", "biet": "bye",
            "ngày": "day", "ngay": "day", "bây": "now", "bay": "now",
            "giờ": "time", "gio": "time"
        }
    
    def create_en2vi_dict(self):
        """Create English to Vietnamese translation dictionary."""
        return {
            "hello": "xin chào", "hi": "chào",
            "how are you": "bạn khỏe không", 
            "what is your name": "tên của bạn là gì",
            "my name is": "tên tôi là",
            "who created you": "ai đã tạo ra bạn",
            "thank you": "cảm ơn bạn", "thanks": "cảm ơn",
            "goodbye": "tạm biệt", "bye": "tạm biệt",
            "yes": "có", "no": "không",
            "i don't know": "tôi không biết",
            "i don't understand": "tôi không hiểu",
            "sorry": "xin lỗi", "excuse me": "xin lỗi",
            "good morning": "chào buổi sáng",
            "good afternoon": "chào buổi chiều",
            "good evening": "chào buổi tối",
            "how can i help you": "tôi có thể giúp gì cho bạn",
            "i'm a chatbot": "tôi là một chatbot",
            "i was created by a developer using pytorch": "tôi được tạo ra bởi một lập trình viên sử dụng PyTorch"
        }
    
    def create_common_phrases(self):
        """Create dictionary of common phrases and their responses."""
        return {
            # Vietnamese phrases
            "xin chào": "xin chào! tôi có thể giúp gì cho bạn hôm nay?",
            "chào": "xin chào! tôi có thể giúp gì cho bạn hôm nay?",
            "bạn tên gì": "tôi là chatbot AI",
            "bạn tên là gì": "tôi là chatbot AI",
            "ai tạo ra bạn": "tôi được tạo ra bởi một lập trình viên sử dụng PyTorch",
            "bạn khỏe không": "tôi khỏe, cảm ơn bạn đã hỏi",
            "cảm ơn": "không có gì!",
            # English phrases
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! How can I help you?",
            "what is your name": "I am an AI chatbot",
            "who created you": "I was created by a developer using PyTorch",
            "how are you": "I'm fine, thank you for asking!",
            "thank you": "You're welcome!",
            "thanks": "You're welcome!"
        }
    
    def is_vietnamese(self, text):
        """Check if the text contains Vietnamese characters."""
        vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        for char in text.lower():
            if char in vietnamese_chars:
                return True
        return False
    
    def translate_vi_to_en(self, text):
        """Translate Vietnamese words to English for model input."""
        if not self.is_vietnamese(text):
            return text
            
        tokens = word_tokenize(text.lower(), preserve_line=True)
        translated_tokens = []
        
        for token in tokens:
            if token in self.vietnamese_to_english_dict:
                translated_tokens.append(self.vietnamese_to_english_dict[token])
            else:
                # Remove diacritics for better chance of matching
                normalized = normalize_string(token)
                if normalized != token and normalized in self.english_model.input_word2idx:
                    translated_tokens.append(normalized)
                else:
                    translated_tokens.append(token)
        
        return ' '.join(translated_tokens)
    
    def translate_en_to_vi(self, response):
        """Translate English response to Vietnamese."""
        # Skip if already using Vietnamese model
        if self.using_vietnamese:
            return response
            
        # If empty response, return default message
        if not response or response.strip() == "":
            return "Tôi không hiểu câu hỏi của bạn."
        
        # Check for complete phrases
        response_lower = response.lower()
        for eng, viet in sorted(self.english_to_vietnamese_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if eng in response_lower:
                response_lower = response_lower.replace(eng, viet)
        
        return response_lower
    
    def preprocess_input(self, text):
        """Preprocess input text based on detected language."""
        is_vi = self.is_vietnamese(text)
        
        # Switch models if needed and available
        if is_vi and self.vietnamese_model and not self.using_vietnamese:
            self.current_model = self.vietnamese_model
            self.using_vietnamese = True
            print("Chuyển sang mô hình tiếng Việt.")
        elif not is_vi and not self.using_vietnamese:
            # Already using English model, no need to switch
            pass
        elif not is_vi and self.using_vietnamese:
            # Consider switching to English model for non-Vietnamese input
            if self.english_model:
                self.current_model = self.english_model
                self.using_vietnamese = False
                print("Switching to English model.")
        
        # Check for common phrases first
        text_lower = text.lower()
        if text_lower in self.common_phrases:
            return self.common_phrases[text_lower]
        
        # Remove diacritics and normalize if using English model
        if self.using_vietnamese:
            normalized_text = normalize_vietnamese(text)
            tokens = word_tokenize(normalized_text, preserve_line=True)
        else:
            # Using English model but input might be Vietnamese
            if is_vi:
                translated_text = self.translate_vi_to_en(text)
                normalized_text = normalize_string(translated_text)
            else:
                normalized_text = normalize_string(text)
            tokens = word_tokenize(normalized_text, preserve_line=True)
        
        return tokens
    
    def postprocess_output(self, response, input_text):
        """Post-process model output."""
        if response is None:
            if self.is_vietnamese(input_text):
                return "Xin lỗi, tôi không hiểu câu hỏi của bạn."
            else:
                return "Sorry, I don't understand your question."
        
        # Translate response if needed
        if self.is_vietnamese(input_text) and not self.using_vietnamese:
            return self.translate_en_to_vi(response)
        
        return response
    
    def chat(self):
        """Run the chatbot interface."""
        print("=" * 50)
        if self.using_vietnamese:
            print("Chatbot đa ngôn ngữ đã sẵn sàng! (Mô hình tiếng Việt)")
            print("Gõ 'exit' để thoát. Gõ 'switch' để đổi ngôn ngữ.")
        else:
            print("Multilingual Chatbot is ready! (English model)")
            print("Type 'exit' to quit. Type 'switch' to change language.")
        print("=" * 50)
        
        while True:
            if self.using_vietnamese:
                prompt = "Bạn: "
            else:
                prompt = "You: "
                
            user_input = input(prompt)
            
            # Check for exit command
            if user_input.lower() == 'exit':
                break
                
            # Check for switch command
            if user_input.lower() == 'switch':
                if self.vietnamese_model and self.english_model:
                    self.using_vietnamese = not self.using_vietnamese
                    self.current_model = self.vietnamese_model if self.using_vietnamese else self.english_model
                    if self.using_vietnamese:
                        print("Đã chuyển sang mô hình tiếng Việt.")
                    else:
                        print("Switched to English model.")
                    continue
                else:
                    print("Cannot switch models because one of the models is not available.")
                    continue
            
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            
            # Check if we got a predefined response
            if isinstance(processed_input, str):
                if self.using_vietnamese:
                    print("Bot:", processed_input)
                else:
                    print("Bot:", processed_input)
                continue
            
            # Get model prediction
            response = self.current_model.predict(processed_input)
            
            # Post-process output
            final_response = self.postprocess_output(response, user_input)
            
            # Print response
            if self.using_vietnamese:
                print("Bot:", final_response)
            else:
                print("Bot:", final_response)


def main():
    """Main function to run the chatbot."""
    parser = argparse.ArgumentParser(description='Unified Seq2Seq Chatbot')
    parser.add_argument('--vietnamese', action='store_true', help='Start with Vietnamese model')
    parser.add_argument('--english', action='store_true', help='Start with English model')
    args = parser.parse_args()
    
    chatbot = UnifiedChatbot()
    
    # Override model selection if specified
    if args.vietnamese and chatbot.vietnamese_model:
        chatbot.current_model = chatbot.vietnamese_model
        chatbot.using_vietnamese = True
    elif args.english and chatbot.english_model:
        chatbot.current_model = chatbot.english_model
        chatbot.using_vietnamese = False
    
    chatbot.chat()


if __name__ == "__main__":
    main()
