import torch
import nltk
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nltk.download('punkt')
from src.preprocess import read_data, tokenize, normalize_string
from src.model import Encoder, Decoder, Seq2Seq
from nltk.tokenize import word_tokenize

# Define models directory
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Load model state_dict
state_dict = torch.load(os.path.join(models_dir, 'chatbot.pt'))

# Xác định kích thước từ state_dict
encoder_embedding_shape = state_dict['encoder.embedding.weight'].shape
decoder_embedding_shape = state_dict['decoder.embedding.weight'].shape

# Hiển thị thông tin
print(f"Kích thước encoder embedding: {encoder_embedding_shape}")
print(f"Kích thước decoder embedding: {decoder_embedding_shape}")

# Khởi tạo model với kích thước đúng
INPUT_DIM = encoder_embedding_shape[0]
OUTPUT_DIM = decoder_embedding_shape[0]
EMB_DIM = encoder_embedding_shape[1]
HID_DIM = 128

# Tạo một encoder và decoder mới
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2Seq(encoder, decoder)

# Load state_dict
model.load_state_dict(state_dict)
model.eval()

# Đọc lại conversation để tái tạo từ điển đúng
data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conversation.txt")
input_texts, target_texts = read_data(data_file)
print(f"Đọc được {len(input_texts)} cặp hội thoại từ file.")

# Tạo từ điển từ -> số
input_seqs, input_word2idx, _ = tokenize(input_texts)
target_seqs, target_word2idx, target_idx2word = tokenize(target_texts)

# Lưu lại từ điển mới
torch.save(input_word2idx, os.path.join(models_dir, 'input_word2idx.pt'))
torch.save(target_word2idx, os.path.join(models_dir, 'target_word2idx.pt'))
torch.save(target_idx2word, os.path.join(models_dir, 'target_idx2word.pt'))

print(f"Từ điển đã được tái tạo và lưu lại.")
print(f"Kích thước từ điển input: {len(input_word2idx)}")
print(f"Kích thước từ điển output: {len(target_word2idx)}")

print("Hoàn tất!")
