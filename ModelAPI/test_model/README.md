# Hướng Dẫn Sử Dụng API Trong `test_model`

Tài liệu này dành cho người dùng khác khi clone hoặc tải project về và muốn dùng lại model đã train mà không cần hiểu chi tiết phần xử lý bên trong.

## 1. Tải full project

Do Git thường không đẩy hết các file dữ liệu lớn hoặc artifact cần thiết, bạn nên tải full project tại đây:

`https://drive.google.com/drive/folders/1mee-aajYwBRndOxirjvUhU7EAwThtJDV?usp=sharing`

Nếu clone repo nhưng thiếu `dataset.csv`, thiếu thư mục `models`, hoặc notebook / API chạy không đủ dữ liệu, hãy dùng bản full project từ link trên.

## 2. Cấu trúc tối thiểu cần có

```text
Stock_Prediction/
|-- dataset.csv
|-- models/
|   |-- lgbm_t3.txt
|   |-- lgbm_t7.txt
|   |-- lgbm_t15.txt
|   |-- lgbm_t30.txt
|   |-- meta_t3.pkl
|   |-- meta_t7.pkl
|   |-- meta_t15.pkl
|   |-- meta_t30.pkl
|-- test_model/
|   |-- model_apply_api.py
|   |-- model_apply.ipynb
|   |-- README.md
```

## 3. API chính nằm ở đâu

Toàn bộ luồng áp dụng model hiện nằm trong file `test_model/model_apply_api.py`.

Người dùng cuối chỉ cần import file này và gọi hàm. Không cần tự tạo feature bằng tay.

## 4. Dữ liệu đầu vào

API nhận 1 file CSV thô dạng master, thường là `dataset.csv`.

Khuyến nghị:
- Mỗi dòng là 1 mã cổ phiếu tại 1 ngày giao dịch.
- Dữ liệu nên giữ đúng schema gốc của `dataset.csv`.
- Nên có lịch sử dài, tốt nhất từ năm 2020 đến hiện tại.
- Nếu không có full lịch sử, nên có ít nhất khoảng 220 phiên giao dịch cho mỗi mã.

## 5. Model đang dự đoán gì

API hiện chạy đồng thời 4 horizon:
- `T+3`
- `T+7`
- `T+15`
- `T+30`

Ngoài ra còn có `Ensemble` để tổng hợp tín hiệu từ nhiều horizon.

Kết quả đầu ra chính là hướng dự đoán `TĂNG` hoặc `GIẢM`.

## 6. Mặc định hệ thống chọn bao nhiêu mã

Nếu bạn không truyền danh sách mã cụ thể, API sẽ tự chọn 10 mã phổ biến nhất trong 24 tháng gần nhất.

## 7. Cách import API

Nếu bạn chạy code từ thư mục gốc của project:

```python
from test_model.model_apply_api import predict_default_symbols
```

Nếu bạn gọi từ project khác:

```python
from pathlib import Path
import sys

PROJECT_DIR = Path(r"C:\\duong_dan\\den\\Stock_Prediction")
sys.path.insert(0, str(PROJECT_DIR / "test_model"))

from model_apply_api import predict_default_symbols
```

## 8. Các hàm nên dùng

### `predict_default_symbols(...)`
Dùng khi bạn muốn cách gọi đơn giản nhất.

### `predict_selected_symbols(...)`
Dùng khi bạn muốn chỉ định danh sách mã cụ thể.

### `predict_one_symbol(...)`
Dùng khi bạn chỉ muốn dự đoán cho một mã duy nhất.

### `predict_symbols_from_master_csv(...)`
Dùng khi bạn muốn kiểm soát chi tiết hơn các tham số như `symbols`, `recent_months`, `mode`, `score_start`, `score_end`.

Ví dụ gọi nhanh:

```python
from test_model.model_apply_api import predict_default_symbols

result = predict_default_symbols(
    master_csv_path=r"C:\\duong_dan\\den\\Stock_Prediction\\dataset.csv",
    models_dir=r"C:\\duong_dan\\den\\Stock_Prediction\\models",
    output_dir=r"C:\\duong_dan\\den\\Stock_Prediction\\test_model\\output",
)
```

## 9. Kết quả trả về gồm gì

Các hàm trả về một dictionary `result`.

Các key quan trọng:
- `result["selected_symbols"]`: danh sách mã đang được dự đoán.
- `result["latest_selected_symbol_table"]`: bảng kết quả mới nhất cho các mã đã chọn.
- `result["symbol_exported_files"]`: danh sách file ảnh đã lưu ra thư mục `output`.
- `result["summary"]`: thông tin tóm tắt về phiên dự đoán.
- `result["horizons"]`: danh sách horizon đang dùng, thường là `3, 7, 15, 30`.

## 10. Output được lưu ra đâu

Nếu bạn truyền `output_dir`, API sẽ lưu ảnh PNG vào thư mục đó.

Thông thường sẽ có:
- 10 ảnh riêng theo mã, ví dụ: `ACB_Prediction_30-03-2026.png`
- 1 ảnh tổng hợp horizon: `All_Horizons_Prediction_30-03-2026.png`

Mỗi lần chạy, code sẽ xóa các file `.png` cũ trong `output/` trước khi lưu bộ ảnh mới.

## 11. Cách dùng trong project khác

Flow tối giản:
1. Crawl dữ liệu mới của ngày hiện tại.
2. Gộp dữ liệu mới vào file master `dataset.csv`.
3. Gọi một trong các hàm API ở trên.
4. Lấy `result["latest_selected_symbol_table"]` để đọc kết quả mới nhất.
5. Hoặc lấy ảnh trong `output/` để hiển thị trực tiếp trên dashboard.

## 12. Notebook minh họa

Nếu muốn xem cách chạy từng bước và hiển thị biểu đồ, mở `test_model/model_apply.ipynb`.

Notebook này dùng chính các hàm trong `model_apply_api.py`, nhưng có thêm phần giải thích và trực quan hóa kết quả.
