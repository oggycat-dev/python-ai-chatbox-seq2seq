import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import Encoder, Decoder, Seq2Seq
from src.preprocess import normalize_vietnamese, read_data, tokenize

# Đảm bảo kết quả có tính tái lập
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Cấu hình mô hình
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMB_DIM = 256
HID_DIM = 128
EPOCHS = 50

# Đọc dữ liệu
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, 'data', 'conversation_vietnamese.txt')
input_texts, target_texts = read_data(file_path)

# Kiểm tra xem có đủ dữ liệu không
if len(input_texts) < 5:
    print(f"CẢNH BÁO: Chỉ tìm thấy {len(input_texts)} cặp câu hỏi-trả lời. Cần ít nhất 5 cặp để huấn luyện hiệu quả.")
    print("Vui lòng thêm dữ liệu vào file data/conversation_vietnamese.txt")
else:
    print(f"Tìm thấy {len(input_texts)} cặp câu hỏi-trả lời trong dữ liệu huấn luyện.")

# In ra một số ví dụ để kiểm tra
print("\nMột số ví dụ từ dữ liệu huấn luyện:")
for i in range(min(3, len(input_texts))):
    print(f"Q: {input_texts[i]}")
    print(f"A: {target_texts[i]}")
    print("---")

# Sử dụng normalize_vietnamese thay vì normalize_string
input_texts = [normalize_vietnamese(text) for text in input_texts]
target_texts = [normalize_vietnamese(text) for text in target_texts]

# Tokenize và tạo từ điển
input_sequences, input_word2idx, input_idx2word = tokenize(input_texts)
target_sequences, target_word2idx, target_idx2word = tokenize(target_texts)

# Lưu từ điển
import os
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)  # Tạo thư mục models nếu chưa tồn tại
torch.save(input_word2idx, os.path.join(models_dir, 'vietnamese_input_word2idx.pt'))
torch.save(input_idx2word, os.path.join(models_dir, 'vietnamese_input_idx2word.pt'))
torch.save(target_word2idx, os.path.join(models_dir, 'vietnamese_target_word2idx.pt'))
torch.save(target_idx2word, os.path.join(models_dir, 'vietnamese_target_idx2word.pt'))

# Tính toán kích thước từ điển
INPUT_DIM = len(input_word2idx)
OUTPUT_DIM = len(target_word2idx)

print(f"Kích thước từ điển đầu vào: {INPUT_DIM}")
print(f"Kích thước từ điển đầu ra: {OUTPUT_DIM}")

# Tạo bộ dữ liệu huấn luyện
def create_batch(sequences, word2idx, batch_size=32):
    random.shuffle(sequences)
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        max_len = max(len(seq) for seq in batch)
        padded_batch = [seq + [word2idx['<PAD>']] * (max_len - len(seq)) for seq in batch]
        tensor_batch = torch.tensor(padded_batch, dtype=torch.long).t()  # (seq_len, batch_size)
        batches.append(tensor_batch)
    return batches

# Tạo cặp batch đầu vào/đầu ra
def create_batch_pairs(input_sequences, target_sequences, input_word2idx, target_word2idx, batch_size=32):
    # Sắp xếp các cặp (đầu vào, đầu ra) theo độ dài của đầu vào
    pairs = list(zip(input_sequences, target_sequences))
    random.shuffle(pairs)
    
    batches = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        # Nếu batch không đủ lớn, bỏ qua batch này
        if len(batch_pairs) < 2:
            continue
            
        input_batch = [pair[0] for pair in batch_pairs]
        target_batch = [pair[1] for pair in batch_pairs]
        
        max_input_len = max(len(seq) for seq in input_batch)
        max_target_len = max(len(seq) for seq in target_batch)
        
        padded_input = [seq + [input_word2idx['<PAD>']] * (max_input_len - len(seq)) for seq in input_batch]
        padded_target = [seq + [target_word2idx['<PAD>']] * (max_target_len - len(seq)) for seq in target_batch]
        
        # Chuyển đổi thành tensor và đảm bảo contiguous
        input_tensor = torch.tensor(padded_input, dtype=torch.long).t().contiguous()  # (seq_len, batch_size)
        target_tensor = torch.tensor(padded_target, dtype=torch.long).t().contiguous()  # (seq_len, batch_size)
        
        batches.append((input_tensor, target_tensor))
    
    if not batches:
        print("CẢNH BÁO: Không thể tạo batch. Vui lòng kiểm tra dữ liệu đầu vào.")
    
    return batches

# Tạo mô hình
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2Seq(encoder, decoder)

# Định nghĩa hàm mất mát và optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Bỏ qua PAD token (index 0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Hàm huấn luyện
def train(model, batch_pairs, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for input_batch, target_batch in batch_pairs:
        optimizer.zero_grad()
        
        # Dự đoán của mô hình
        output = model(input_batch, target_batch)
        
        # Tính toán mất mát
        output_dim = output.shape[-1]
        # Sử dụng reshape thay vì view để xử lý tensor không liên tục
        output = output[1:].reshape(-1, output_dim)  # Bỏ qua token đầu tiên (bắt đầu)
        target = target_batch[1:].reshape(-1)  # Bỏ qua token đầu tiên
        
        loss = criterion(output, target)
        loss.backward()
        
        # Clip gradient để tránh exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(batch_pairs)

# Huấn luyện mô hình
print("Bắt đầu quá trình huấn luyện...")
batch_pairs = create_batch_pairs(input_sequences, target_sequences, input_word2idx, target_word2idx, BATCH_SIZE)

# Thông tin về kích thước batch và tensor để gỡ lỗi
if len(batch_pairs) > 0:
    sample_input, sample_target = batch_pairs[0]
    print(f"Thông tin batch đầu tiên:")
    print(f"  - Kích thước input: {sample_input.shape}")
    print(f"  - Kích thước target: {sample_target.shape}")
else:
    print("CẢNH BÁO: Không tìm thấy batch nào, vui lòng kiểm tra dữ liệu đầu vào.")

for epoch in range(EPOCHS):
    try:
        train_loss = train(model, batch_pairs, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")
          # Lưu mô hình sau mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(models_dir, f'vietnamese_chatbot_epoch{epoch+1}.pt'))
    except Exception as e:
        print(f"Lỗi trong epoch {epoch+1}: {str(e)}")
        print("Cố gắng tiếp tục với epoch tiếp theo...")
        continue

# Lưu mô hình cuối cùng
torch.save(model.state_dict(), os.path.join(models_dir, 'vietnamese_chatbot.pt'))
print("Huấn luyện hoàn tất!")
