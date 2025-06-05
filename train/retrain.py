import torch
import torch.nn as nn
import nltk
nltk.download('punkt')
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess import read_data, tokenize
from src.model import Encoder, Decoder, Seq2Seq
import os

print("Bắt đầu quá trình huấn luyện chatbot từ đầu...")

# 1. Đọc dữ liệu
input_texts, target_texts = read_data("data/conversation_new.txt")
print(f"Đọc được {len(input_texts)} cặp hội thoại từ file.")

# 2. Tạo từ điển từ -> số
input_seqs, input_word2idx, input_idx2word = tokenize(input_texts)
target_seqs, target_word2idx, target_idx2word = tokenize(target_texts)

# 3. Chuyển đổi thành tensor và padding
def pad_sequences(seqs, max_len):
    return [seq + [0]*(max_len - len(seq)) for seq in seqs]

input_maxlen = max(len(seq) for seq in input_seqs)
target_maxlen = max(len(seq) for seq in target_seqs)

input_tensor = torch.tensor(pad_sequences(input_seqs, input_maxlen)).T  # (seq_len, batch)
target_tensor = torch.tensor(pad_sequences(target_seqs, target_maxlen)).T

# 4. Dataloader
dataset = TensorDataset(input_tensor.T, target_tensor.T)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Khởi tạo mô hình
INPUT_DIM = len(input_word2idx)
OUTPUT_DIM = len(target_word2idx)
EMB_DIM = 64
HID_DIM = 128

print(f"Kích thước từ điển đầu vào: {INPUT_DIM}")
print(f"Kích thước từ điển đầu ra: {OUTPUT_DIM}")

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2Seq(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 6. Huấn luyện
EPOCHS = 100

for epoch in range(EPOCHS):
    total_loss = 0
    for src_batch, trg_batch in loader:
        src = src_batch.T  # (seq_len, batch)
        trg = trg_batch.T

        output = model(src, trg)
        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# 7. Lưu mô hình
os.makedirs("models", exist_ok=True)  # Tạo thư mục models nếu chưa tồn tại
torch.save(model.state_dict(), "models/chatbot_new.pt")
torch.save(input_word2idx, "models/input_word2idx.pt")
torch.save(target_word2idx, "models/target_word2idx.pt")
torch.save(input_idx2word, "models/input_idx2word.pt") 
torch.save(target_idx2word, "models/target_idx2word.pt")

print("\nHuấn luyện hoàn tất và mô hình đã được lưu vào thư mục models/")
