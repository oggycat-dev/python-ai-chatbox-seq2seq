# Chatbot Seq2Seq đa ngôn ngữ (Tiếng Anh & Tiếng Việt)

Đây là một dự án chatbot sử dụng mô hình Sequence-to-Sequence (Seq2Seq) được xây dựng bằng PyTorch. Chatbot được thiết kế để hỗ trợ cả tiếng Anh và tiếng Việt thông qua một giao diện đơn giản.

## Tính năng chính

- **Hỗ trợ đa ngôn ngữ**: Tự động phát hiện và chuyển đổi giữa tiếng Anh và tiếng Việt
- **Kiến trúc Seq2Seq**: Sử dụng mô hình encoder-decoder với GRU
- **Giao diện thống nhất**: Một file duy nhất cho giao diện chat (`chat/unified_chat.py`)
- **Các mô hình đã huấn luyện**: Có sẵn mô hình cho cả tiếng Anh và tiếng Việt

## Cấu trúc dự án

```
.
├── chat/                      # Giao diện chat
│   └── unified_chat.py        # Giao diện chat thống nhất hỗ trợ đa ngôn ngữ
├── data/                      # Dữ liệu huấn luyện
│   ├── conversation.txt       # Dữ liệu tiếng Anh gốc
│   ├── conversation_new.txt   # Dữ liệu tiếng Anh mở rộng
│   └── conversation_vietnamese.txt  # Dữ liệu tiếng Việt
├── models/                    # Các mô hình đã huấn luyện
│   ├── chatbot.pt             # Mô hình tiếng Anh ban đầu
│   ├── chatbot_new.pt         # Mô hình tiếng Anh cải tiến
│   ├── vietnamese_chatbot.pt  # Mô hình tiếng Việt
│   └── ...                    # Các từ điển và mô hình checkpoint
├── src/                       # Mã nguồn chính
│   ├── model.py               # Định nghĩa kiến trúc mô hình Seq2Seq
│   └── preprocess.py          # Xử lý dữ liệu đầu vào
├── tools/                     # Các công cụ và tiện ích
│   ├── check_model.py         # Kiểm tra mô hình
│   ├── download_nltk_resources.py # Tải các tài nguyên NLTK
│   ├── organize_project.py    # Tổ chức cấu trúc dự án
│   └── recreate_dictionary.py # Tạo lại từ điển
├── train/                     # Các file huấn luyện mô hình
│   ├── train.py               # Huấn luyện mô hình tiếng Anh
│   ├── train_vietnamese.py    # Huấn luyện mô hình tiếng Việt
│   └── retrain.py             # Huấn luyện lại mô hình
├── quickstart.py              # Script tự động hoá quá trình thiết lập và chạy
├── requirements.txt           # Các thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn
```

## Khởi động nhanh

### Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Tải tài nguyên NLTK (nếu cần):
```bash
python tools/download_nltk_resources.py
```

Hoặc sử dụng script tự động:
```bash
python quickstart.py --setup
```

### Sử dụng chatbot

Cách đơn giản nhất để bắt đầu là sử dụng giao diện chat thống nhất:
```bash
python chat/unified_chat.py
```

Giao diện này sẽ tự động phát hiện ngôn ngữ và sử dụng mô hình phù hợp.

Hoặc sử dụng script quickstart với các chế độ khác nhau:
```bash
python quickstart.py --chat english            # Chế độ tiếng Anh
python quickstart.py --chat vietnamese_native  # Chế độ tiếng Việt (mô hình bản địa)
python quickstart.py --chat vietnamese_translate # Chế độ tiếng Việt (phiên dịch)
python quickstart.py --chat auto               # Tự động phát hiện mô hình tốt nhất
```

## Hướng dẫn sử dụng chi tiết

### 1. Huấn luyện mô hình

#### Huấn luyện mô hình tiếng Anh:
```bash
python train/train.py
```

#### Huấn luyện mô hình tiếng Việt:
```bash
python train/train_vietnamese.py
```

Hoặc sử dụng script tự động:
```bash
python quickstart.py --train english    # Huấn luyện mô hình tiếng Anh
python quickstart.py --train vietnamese # Huấn luyện mô hình tiếng Việt
python quickstart.py --train all        # Huấn luyện cả hai mô hình
```

Bạn có thể theo dõi tiến trình huấn luyện qua thông tin hiển thị và mô hình sẽ được lưu vào thư mục `models/`.

### 2. Thêm dữ liệu huấn luyện

#### Tiếng Anh:
Thêm các cặp hội thoại vào file `data/conversation.txt` hoặc `data/conversation_new.txt` theo định dạng:
```
Câu hỏi tiếng Anh [tab] Câu trả lời tiếng Anh
```

#### Tiếng Việt:
Thêm các cặp hội thoại vào file `data/conversation_vietnamese.txt` theo định dạng:
```
Câu hỏi tiếng Việt [tab] Câu trả lời tiếng Việt
```

## Kiến trúc mô hình

Dự án sử dụng kiến trúc Sequence-to-Sequence với cơ chế Encoder-Decoder:

### Encoder
Encoder nhận chuỗi đầu vào (câu hỏi của người dùng) và mã hóa thành một vector biểu diễn có kích thước cố định:
- **Embedding Layer**: Chuyển đổi mỗi từ thành một vector đặc trưng
- **GRU (Gated Recurrent Unit)**: Xử lý chuỗi các vector embedding để tạo ra một vector biểu diễn cuối cùng

### Decoder
Decoder nhận vector biểu diễn từ Encoder và tạo ra chuỗi đầu ra (câu trả lời):
- **Embedding Layer**: Chuyển đổi từng từ thành vector đặc trưng
- **GRU**: Sử dụng trạng thái ẩn từ Encoder để dự đoán từng từ trong chuỗi đầu ra
- **Linear Layer**: Chuyển đổi vector đặc trưng thành phân phối xác suất cho từng từ trong từ điển

### Hỗ trợ tiếng Việt
Dự án hỗ trợ tiếng Việt thông qua hai phương pháp:

1. **Phương pháp phiên dịch**: Chuyển đổi ký tự tiếng Việt sang tiếng Anh, sử dụng mô hình tiếng Anh, sau đó dịch phản hồi trở lại thành tiếng Việt.

2. **Mô hình tiếng Việt bản địa**: Huấn luyện một mô hình mới với dữ liệu tiếng Việt, cho phép xử lý trực tiếp các ký tự tiếng Việt.

## Tùy chỉnh mô hình

Bạn có thể điều chỉnh các thông số trong các file huấn luyện để cải thiện hiệu suất:

- `EMB_DIM`: Kích thước embedding (mặc định: 256)
- `HID_DIM`: Kích thước lớp ẩn (mặc định: 128) 
- `BATCH_SIZE`: Kích thước batch (mặc định: 64)
- `LEARNING_RATE`: Tốc độ học (mặc định: 0.001)
- `EPOCHS`: Số epoch (mặc định: 50)

## Xử lý lỗi phổ biến

### Lỗi "Token không có trong từ điển"
Xảy ra khi người dùng nhập một từ không có trong từ điển huấn luyện. Mô hình chỉ có thể nhận ra các từ đã xuất hiện trong dữ liệu huấn luyện.

### Lỗi "không đủ dữ liệu huấn luyện"
Nếu bạn thấy cảnh báo về việc thiếu dữ liệu huấn luyện, bạn cần thêm nhiều cặp câu hỏi-trả lời vào file dữ liệu. Mô hình cần ít nhất 10-20 cặp để huấn luyện một cách hiệu quả, và càng nhiều dữ liệu thì kết quả càng tốt.

### Lỗi "CUDA out of memory"
Xảy ra khi GPU không đủ bộ nhớ để huấn luyện mô hình. Giải pháp:
- Giảm kích thước batch
- Giảm kích thước mô hình (EMB_DIM, HID_DIM)
- Sử dụng CPU thay vì GPU

## Mẹo cải thiện chất lượng mô hình

### 1. Tăng cường dữ liệu huấn luyện
Thêm nhiều cặp câu hỏi-trả lời đa dạng. Một số cách để mở rộng dữ liệu:
- Thêm các câu hỏi và câu trả lời trong nhiều chủ đề khác nhau
- Thêm các biến thể của cùng một câu hỏi (ví dụ: "bạn là ai?" và "bạn là người nào?")
- Đảm bảo có các câu trả lời ngắn và dài để mô hình học cách phản hồi đa dạng

### 2. Điều chỉnh tham số huấn luyện
Trong file huấn luyện, bạn có thể điều chỉnh:
- `EPOCHS`: Tăng lên 100-200 để mô hình học sâu hơn
- `EMB_DIM`: Tăng lên 512 để mô hình nắm bắt ngữ nghĩa tốt hơn
- `HID_DIM`: Tăng lên 256 để mô hình có khả năng ghi nhớ ngữ cảnh tốt hơn
- `LEARNING_RATE`: Giảm xuống 0.0005 nếu mô hình không hội tụ

## Tổng quan về lệnh quickstart

Script `quickstart.py` là công cụ giúp bạn thực hiện các tác vụ phổ biến một cách nhanh chóng.

### Cú pháp chung
```bash
python quickstart.py [LỆNH]
```

### Các lệnh có sẵn

1. **Cài đặt môi trường**
```bash
python quickstart.py --setup
```
Lệnh này sẽ cài đặt các thư viện cần thiết và tải tài nguyên NLTK.

2. **Huấn luyện mô hình**
```bash
python quickstart.py --train english      # Huấn luyện mô hình tiếng Anh
python quickstart.py --train vietnamese   # Huấn luyện mô hình tiếng Việt
python quickstart.py --train all          # Huấn luyện cả hai mô hình
```

3. **Khởi chạy chatbot**
```bash
python quickstart.py --chat english            # Chế độ tiếng Anh
python quickstart.py --chat vietnamese_translate # Chế độ tiếng Việt (phiên dịch)
python quickstart.py --chat vietnamese_native  # Chế độ tiếng Việt (mô hình bản địa)
python quickstart.py --chat auto               # Tự động phát hiện mô hình tốt nhất
```

Tất cả các lệnh chat đều sử dụng giao diện thống nhất (`unified_chat.py`) với các tham số phù hợp.


