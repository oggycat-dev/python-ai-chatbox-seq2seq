import re
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def normalize_string(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", s)
    return s.strip()

def normalize_vietnamese(s):
    """Normalize Vietnamese text while preserving characters with diacritics.
    This is useful for training models that need to understand Vietnamese text."""
    s = s.lower()
    # Only remove characters that aren't Vietnamese, English, or common punctuation
    s = re.sub(r"[^a-zA-Z0-9?.!,àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+", " ", s)
    return s.strip()

def read_data(file_path):
    input_texts, target_texts = [], []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            pair = line.strip().split('\t')
            if len(pair) == 2:
                input_texts.append(normalize_string(pair[0]))
                target_texts.append(normalize_string(pair[1]))
    return input_texts, target_texts

def tokenize(texts, word2idx=None):
    # Sử dụng preserve_line=True để tránh dùng sent_tokenize bên trong
    all_tokens = [word_tokenize(txt, preserve_line=True) for txt in texts]
    if word2idx is None:
        vocab = set(token for line in all_tokens for token in line)
        word2idx = {word: i+2 for i, word in enumerate(vocab)}
        word2idx['<PAD>'] = 0
        word2idx['<EOS>'] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    sequences = [[word2idx[token] for token in tokens if token in word2idx] + [word2idx['<EOS>']] for tokens in all_tokens]
    return sequences, word2idx, idx2word
