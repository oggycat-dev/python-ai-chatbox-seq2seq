import nltk
import os
import shutil

# Tải gói punkt cơ bản
nltk.download('punkt')

# Một số phiên bản NLTK trên Python 3.13 yêu cầu tải thêm punkt_tab
try:
    # Tạo thư mục tokenizers/punkt_tab/english
    from nltk.data import find

    # Tìm thư mục punkt đã tải
    punkt_dir = find('tokenizers/punkt')
    
    # Lấy đường dẫn tới thư mục nltk_data
    nltk_data = os.path.dirname(os.path.dirname(punkt_dir))
    
    # Tạo thư mục punkt_tab và thư mục con english
    punkt_tab_dir = os.path.join(nltk_data, 'tokenizers', 'punkt_tab')
    english_dir = os.path.join(punkt_tab_dir, 'english')
    
    os.makedirs(english_dir, exist_ok=True)
    
    # Sao chép một số file từ punkt sang punkt_tab/english
    punkt_pickle = os.path.join(punkt_dir, 'english.pickle')
    if os.path.exists(punkt_pickle):
        shutil.copy2(punkt_pickle, os.path.join(english_dir, 'english.pickle'))
    
    print("Đã tạo thành công thư mục punkt_tab!")
except Exception as e:
    print(f"Có lỗi khi tạo thư mục punkt_tab: {e}")
    print("Bạn có thể cần tải thủ công punkt_tab nếu có.")

print("Đã tải xong các tài nguyên NLTK cần thiết!")