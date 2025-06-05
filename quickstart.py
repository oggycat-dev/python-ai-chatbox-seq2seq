import os
import subprocess
import sys
import argparse

def print_header(title):
    """In tiêu đề với định dạng đẹp"""
    print("\n" + "=" * 80)
    print(" " * ((80 - len(title)) // 2) + title)
    print("=" * 80 + "\n")

def run_command(command, description=None):
    """Chạy lệnh và hiển thị kết quả"""
    if description:
        print(f"{description}...\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("LỖI:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Lỗi khi chạy lệnh: {e}")
        return False

def check_files_exist(files):
    """Kiểm tra các tệp tồn tại hay không"""
    missing = []
    for file in files:
        if not os.path.exists(file):
            missing.append(file)
    return missing

def setup_environment():
    """Cài đặt môi trường cần thiết"""
    print_header("THIẾT LẬP MÔI TRƯỜNG")
    
    # Kiểm tra và cài đặt các thư viện cần thiết
    run_command("pip install -r requirements.txt", "Cài đặt các thư viện cần thiết")
    
    # Tải tài nguyên NLTK
    run_command("python tools/download_nltk_resources.py", "Tải tài nguyên NLTK")
    
    print("\nThiết lập môi trường hoàn tất!\n")

def train_english_model():
    """Huấn luyện mô hình tiếng Anh"""
    print_header("HUẤN LUYỆN MÔ HÌNH TIẾNG ANH")
    
    # Kiểm tra dữ liệu
    if not os.path.exists("data/conversation_new.txt"):
        print("CẢNH BÁO: Không tìm thấy file data/conversation_new.txt")
        if os.path.exists("data/conversation.txt"):
            print("Sử dụng data/conversation.txt thay thế")
        else:
            print("LỖI: Không tìm thấy dữ liệu huấn luyện tiếng Anh!")
            return False
    
    # Huấn luyện mô hình
    success = run_command("python train/train.py", "Đang huấn luyện mô hình tiếng Anh")
    
    if success:
        print("\nHuấn luyện mô hình tiếng Anh hoàn tất!")
    else:
        print("\nHuấn luyện mô hình tiếng Anh thất bại!")
    
    return success

def train_vietnamese_model():
    """Huấn luyện mô hình tiếng Việt"""
    print_header("HUẤN LUYỆN MÔ HÌNH TIẾNG VIỆT")
      # Kiểm tra dữ liệu tiếng Việt
    if not os.path.exists("data/conversation_vietnamese.txt"):
        print("LỖI: Không tìm thấy file data/conversation_vietnamese.txt!")
        return False
    
    # Huấn luyện mô hình
    success = run_command("python train/train_vietnamese.py", "Đang huấn luyện mô hình tiếng Việt")
    
    if success:
        print("\nHuấn luyện mô hình tiếng Việt hoàn tất!")
    else:
        print("\nHuấn luyện mô hình tiếng Việt thất bại!")
    
    return success

def start_chat(mode):
    """Khởi chạy chatbot theo chế độ đã chọn"""
    print_header(f"KHỞI CHẠY CHATBOT CHẾ ĐỘ: {mode.upper()}")
    
    # Kiểm tra file chat thống nhất
    if not os.path.exists("chat/unified_chat.py"):
        print("LỖI: Không tìm thấy file chat/unified_chat.py!")
        return False
    
    if mode == "english":
        # Kiểm tra mô hình tiếng Anh
        missing = check_files_exist(["models/chatbot_new.pt", "models/input_word2idx.pt", "models/target_word2idx.pt"])
        if missing:
            print(f"LỖI: Không tìm thấy các file: {', '.join(missing)}")
            print("Bạn cần huấn luyện mô hình tiếng Anh trước!")
            return False
        
        run_command("python chat/unified_chat.py --english", "Khởi chạy chatbot tiếng Anh")
    
    elif mode == "vietnamese_translate":
        # Kiểm tra mô hình tiếng Anh (cần cho phiên dịch)
        missing = check_files_exist(["models/chatbot_new.pt", "models/input_word2idx.pt", "models/target_word2idx.pt"])
        if missing:
            print(f"LỖI: Không tìm thấy các file: {', '.join(missing)}")            
            print("Bạn cần huấn luyện mô hình tiếng Anh trước!")
            return False
        
        run_command("python chat/unified_chat.py --english", "Khởi chạy chatbot hỗ trợ tiếng Việt (chế độ phiên dịch)")
    
    elif mode == "vietnamese_native":
        # Kiểm tra mô hình tiếng Việt
        missing = check_files_exist([
            "models/vietnamese_chatbot.pt", 
            "models/vietnamese_input_word2idx.pt", 
            "models/vietnamese_target_word2idx.pt"
        ])
        if missing:
            print(f"LỖI: Không tìm thấy các file: {', '.join(missing)}")
            print("Bạn cần huấn luyện mô hình tiếng Việt trước!")
            return False
        
        run_command("python chat/unified_chat.py --vietnamese", "Khởi chạy chatbot tiếng Việt (chế độ bản địa)")
    
    elif mode == "auto":
        # Tự động phát hiện mô hình tốt nhất
        run_command("python chat/unified_chat.py", "Khởi chạy chatbot tự động phát hiện mô hình tốt nhất")
    
    else:
        print(f"LỖI: Chế độ '{mode}' không hợp lệ!")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Chatbot Seq2Seq với hỗ trợ tiếng Việt')
    
    # Các lệnh chính
    parser.add_argument('--setup', action='store_true', help='Cài đặt môi trường')
    parser.add_argument('--train', choices=['english', 'vietnamese', 'all'], help='Huấn luyện mô hình')
    parser.add_argument('--chat', choices=['english', 'vietnamese_translate', 'vietnamese_native', 'auto'], 
                        help='Khởi chạy chatbot')
      # Bảng trợ giúp nếu không có tham số
    if len(sys.argv) == 1:
        print_header("CHATBOT SEQ2SEQ VỚI HỖ TRỢ TIẾNG VIỆT")
        print("Sử dụng: python quickstart.py [LỆNH]")
        print("\nCác lệnh:")
        print("  --setup                                  Cài đặt môi trường")
        print("  --train {english,vietnamese,all}         Huấn luyện mô hình")
        print("  --chat {english,vietnamese_translate,vietnamese_native,auto}  Khởi chạy chatbot")
        print("\nVí dụ:")
        print("  python quickstart.py --setup              Cài đặt môi trường")
        print("  python quickstart.py --train english      Huấn luyện mô hình tiếng Anh")
        print("  python quickstart.py --chat auto          Khởi chạy chatbot tự động phát hiện mô hình tốt nhất")
        print("\nGhi chú:")
        print("  Tất cả các chế độ chat đều sử dụng giao diện thống nhất (unified_chat.py)")
        return
    
    args = parser.parse_args()
    
    # Xử lý các lệnh
    if args.setup:
        setup_environment()
    
    if args.train:
        if args.train == 'english':
            train_english_model()
        elif args.train == 'vietnamese':
            train_vietnamese_model()
        elif args.train == 'all':
            train_english_model()
            train_vietnamese_model()
    
    if args.chat:
        start_chat(args.chat)

if __name__ == "__main__":
    main()
