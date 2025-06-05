import torch

# Load từ điển và model
word2idx = torch.load('models/word2idx.pt')
idx2word = torch.load('models/idx2word.pt')
state_dict = torch.load('models/chatbot.pt')

# In thông tin về từ điển
print(f"Kích thước từ điển word2idx: {len(word2idx)}")
print(f"Một số từ trong từ điển: {list(word2idx.items())[:10]}")
print(f"Giá trị index lớn nhất: {max(word2idx.values())}")

# In thông tin về state_dict
print("\nThông tin về model state_dict:")
encoder_embedding_shape = state_dict['encoder.embedding.weight'].shape
print(f"encoder.embedding.weight shape: {encoder_embedding_shape}")

# Kiểm tra xem có sự không khớp giữa từ điển và model không
if encoder_embedding_shape[0] != len(word2idx):
    print(f"\nCHÚ Ý: Không khớp kích thước! Model cần {encoder_embedding_shape[0]} từ, nhưng từ điển có {len(word2idx)} từ.")
