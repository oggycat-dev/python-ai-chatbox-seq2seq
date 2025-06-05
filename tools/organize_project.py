import os
import shutil
import sys

def print_header(title):
    """In tiêu đề với định dạng đẹp"""
    print("\n" + "=" * 80)
    print(" " * ((80 - len(title)) // 2) + title)
    print("=" * 80 + "\n")

def create_directory(path):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Đã tạo thư mục: {path}")
    return path

def move_file(src, dest):
    """Di chuyển file từ src đến dest"""
    if os.path.exists(src):
        # Tạo thư mục đích nếu chưa tồn tại
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # Di chuyển file
        shutil.move(src, dest)
        print(f"Đã di chuyển: {src} -> {dest}")
    else:
        print(f"Không tìm thấy file: {src}")

def organize_project():
    """Tổ chức lại cấu trúc dự án"""
    print_header("TỔ CHỨC LẠI CẤU TRÚC DỰ ÁN")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Tạo các thư mục cần thiết
    scripts_dir = create_directory(os.path.join(base_dir, "scripts"))
    models_dir = create_directory(os.path.join(base_dir, "models"))
    data_dir = create_directory(os.path.join(base_dir, "data"))
    src_dir = create_directory(os.path.join(base_dir, "src"))
    
    # Danh sách các file cần di chuyển
    files_to_move = [
        # Di chuyển các file tiện ích vào thư mục scripts
        (os.path.join(base_dir, "check_model.py"), os.path.join(scripts_dir, "check_model.py")),
        (os.path.join(base_dir, "recreate_dictionary.py"), os.path.join(scripts_dir, "recreate_dictionary.py")),
        (os.path.join(base_dir, "download_nltk_resources.py"), os.path.join(scripts_dir, "download_nltk_resources.py")),
        (os.path.join(base_dir, "check_vietnamese_data.py"), os.path.join(scripts_dir, "check_vietnamese_data.py")),
        (os.path.join(base_dir, "create_vietnamese_dictionary.py"), os.path.join(scripts_dir, "create_vietnamese_dictionary.py")),
        
        # Di chuyển model.py vào thư mục src nếu không ở đó
        (os.path.join(base_dir, "model.py"), os.path.join(src_dir, "model.py")),
    ]
    
    # Di chuyển các file
    for src, dest in files_to_move:
        if os.path.exists(src) and not os.path.exists(dest):
            move_file(src, dest)
    
    print("\nTổ chức lại cấu trúc dự án hoàn tất!")

if __name__ == "__main__":
    organize_project()
