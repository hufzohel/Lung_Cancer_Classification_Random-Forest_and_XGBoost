# Lung Cancer Classification — Random Forest & XGBoost

Dự án bài tập lớn môn **Học máy (CO3117)**, nghiên cứu bài toán **phân loại nhị phân nhãn ung thư phổi** (`LUNG_CANCER`: `YES`/`NO`) từ dữ liệu khảo sát về nhân khẩu học, hành vi và triệu chứng. Dự án so sánh kết quả thực nghiệm của **Random Forest** và **XGBoost** với baseline `DummyClassifier` (luôn dự đoán nhãn chiếm đa số).

> **Phạm vi:** Đây là thực nghiệm học máy trên bộ dữ liệu khảo sát nhỏ, **không phải công cụ chẩn đoán hoặc sàng lọc y tế đã được kiểm định lâm sàng**.

## Bài toán và dữ liệu

- Nguồn: bộ dữ liệu *Lung Cancer* (Kaggle), 309 bản ghi và 16 cột ban đầu.
- Đầu vào: **15 đặc trưng** (tuổi, giới tính và 13 biến khảo sát nhị phân); đầu ra: nhãn `LUNG_CANCER` (`YES`/`NO`).
- Làm sạch: loại **33 dòng trùng hoàn toàn**, còn **276 bản ghi**: 238 `YES` (86,23%) và 38 `NO` (13,77%). Không phát hiện giá trị khuyết hay giá trị ngoài miền được quy định trong dữ liệu khảo sát.
- Mã hóa: `YES` = 1, `NO` = 0; các đặc trưng phân loại được mã hóa nhị phân.
- Chia dữ liệu **stratified 80/20** (`random_state=42`): train **220 mẫu** (190 YES, 30 NO), test giữ riêng **56 mẫu** (48 YES, 8 NO).

## Tech stack

- **Ngôn ngữ:** Python
- **Machine learning:** scikit-learn (Random Forest, DummyClassifier, GridSearchCV, cross-validation, metrics), XGBoost
- **Xử lý dữ liệu:** pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Môi trường phân tích:** Python scripts và Jupyter notebooks (nếu sử dụng)

Thư viện chính được khai báo trong [`requirements.txt`](requirements.txt).

## Phương pháp

1. Tiền xử lý và mã hóa dữ liệu, loại trùng trước khi chia train/test.
2. Dùng **5-fold stratified cross-validation** trên tập train để đánh giá các phương pháp.
3. **Random Forest:** thử 18 cấu hình `max_depth`, `min_samples_leaf` và `class_weight` bằng `GridSearchCV`, lựa chọn theo *balanced accuracy* trung bình; cấu hình được chọn: `max_depth=4`, `min_samples_leaf=1`, `class_weight="balanced"`, `n_estimators=200`. Fit lại trên 220 mẫu train.
4. **XGBoost:** dùng cấu hình cố định (`binary:logistic`, 100 estimators, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`) và `scale_pos_weight = N_NO / N_YES` trong từng fold. Giữ hai checkpoint được chọn từ các fold: **XGBoost A** (fold 3) và **XGBoost B** (fold 5); mỗi checkpoint đã fit trên 176 mẫu.
5. Đánh giá các mô hình đã lưu trên **cùng tập holdout 56 mẫu**, đối chiếu confusion matrix và các chỉ số.

**Lưu ý về phép so sánh:** Random Forest được fit lại trên toàn bộ 220 mẫu sau bước chọn cấu hình, còn XGBoost A/B là checkpoint từ hai fold (176 mẫu/checkpoint). Vì chiến lược chọn và lượng dữ liệu fit khác nhau, kết quả **so sánh các mô hình đã lưu**, không chứng minh thuật toán nào vượt trội trong điều kiện huấn luyện được kiểm soát tương đương.

## Kết quả

### Cross-validation (trung bình 5 fold trên tập train)

| Phương pháp | Accuracy | Balanced accuracy | Recall NO | Recall YES |
|---|---:|---:|---:|---:|
| Dummy — lớp đa số | 86,36% | 50,00% | 0,00% | 100,00% |
| Random Forest cơ sở | 89,55% | 71,49% | 46,67% | 96,32% |
| Random Forest tinh chỉnh | 85,45% | **77,54%** | **66,67%** | 88,42% |
| XGBoost cấu hình cố định | 85,00% | 75,88% | 63,33% | 88,42% |

### Holdout test (56 mẫu: 8 NO, 48 YES)

| Phương pháp | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy — lớp đa số | 85,71% | 50,00% | 0,4615 | 0,5000 | 0 | 8 | 0 | 48 |
| Random Forest | 92,86% | **95,83%** | 0,8783 | **0,9766** | 8 | 0 | 4 | 44 |
| XGBoost A | 92,86% | **95,83%** | 0,8783 | 0,9661 | 8 | 0 | 4 | 44 |
| XGBoost B | **94,64%** | 91,67% | **0,8960** | 0,9714 | 7 | 1 | 2 | 46 |

XGBoost B có accuracy cao nhất và bỏ sót ít mẫu YES nhất trong lần thử này (2/48). Random Forest và XGBoost A nhận đúng toàn bộ 8/8 mẫu NO trên holdout. Chênh lệch 94,64% và 92,86% chỉ tương đương **một dự đoán đúng trên 56 mẫu**, nên không được diễn giải thành ưu thế ổn định ngoài mẫu. Các con số trên lấy từ báo cáo CO3117, tháng 11/2026.

## Cấu trúc repository

```text
.
├── .github/workflows/      # GitHub Actions
├── data/                  # Dữ liệu khảo sát
├── src/
│   ├── models/
│   │   ├── random_forest/ # Huấn luyện Random Forest
│   │   └── xgboost/       # Huấn luyện XGBoost
│   ├── evaluation/        # Đánh giá mô hình
│   └── visualization/     # Trực quan hóa
├── results/               # Kết quả, metrics, model artifacts và hình
├── requirements.txt       # Dependencies
└── readme.md
```

> Cây trên là **cấu trúc thư mục chính** đã xác nhận; tên file chi tiết và đường dẫn checkpoint cần đối chiếu với phiên bản code đang sử dụng. `results/` là nơi đích thống nhất cho các artifact, không phải mã nguồn.

## Cài đặt

Yêu cầu: Python 3.10+ được khuyến nghị (cần kiểm tra tương thích theo hệ điều hành và phiên bản thư viện được cài).

```bash
git clone https://github.com/hufzohel/Lung_Cancer_Classification_Random-Forest_and_XGBoost.git
cd Lung_Cancer_Classification_Random-Forest_and_XGBoost
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Cài thư viện:

```bash
python -m pip install -r requirements.txt
```

## Chạy pipeline

**Luôn chạy từ thư mục gốc của repository** để các import `src.*` và đường dẫn tương đối có cùng điểm xuất phát.

```bash
# Bước tiền xử lý: lệnh được ghi trong README gốc
python -m src.data.preprocess
```

Sau đó chạy **entrypoint huấn luyện Random Forest và XGBoost hiện có** trong `src/models/random_forest/` và `src/models/xgboost/`, rồi chạy notebook/script đánh giá và visualization tương ứng. **Tên entrypoint chính xác chưa thể xác nhận từ phiên bản mã nguồn đã truy cập**, nên không đưa ra lệnh module suy đoán có thể gây lỗi. Khi tích hợp hoàn chỉnh, bổ sung tại đây chính xác các lệnh `python -m ...`/`jupyter ...` đã được kiểm thử.

### Quy ước lưu và nạp XGBoost

Các file model XGBoost được ghi vào **`<repository-root>/results/`** thay vì thư mục mã nguồn. Trong cả training và notebook evaluation, dùng đường dẫn tuyệt đối dựa trên repository root (không phụ thuộc thư mục đang mở notebook); **giữ nguyên tên file và định dạng artifact hiện có** để không thay đổi logic chọn mô hình. Nếu thay đổi đường dẫn output, cập nhật đường dẫn input ở notebook đọc kết quả tương ứng.

```python
from pathlib import Path

# Trong file Python thuộc src/models/xgboost/:
ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Thay đường dẫn output cũ bằng RESULTS_DIR / <existing_filename>.
```

Trong notebook (không dùng `__file__`), xác định `ROOT` bằng cách tìm thư mục có `requirements.txt` và `src/` trong các thư mục cha của `Path.cwd()`, rồi dùng `ROOT / "results" / <existing_filename>` khi load. Nếu notebook sử dụng đường dẫn khác, cần cập nhật đồng bộ trước khi chạy.

## Giới hạn

- Tập test chỉ có 56 mẫu, riêng lớp NO chỉ có 8 mẫu; một trường hợp sai làm recall NO thay đổi 12,5 điểm phần trăm.
- Dữ liệu khảo sát nhỏ và mất cân bằng, chưa đủ để suy rộng ra dữ liệu lâm sàng thực tế.
- Hai nhánh mô hình dùng quy trình lưu/chọn checkpoint khác nhau; phép so sánh không phải đối chứng đồng nhất.

## Tài liệu

- Báo cáo nhóm: *Phân loại nhãn ung thư phổi từ dữ liệu khảo sát*, CO3117 (11/2026).
- Breiman (2001), *Random Forests*, Machine Learning.
- Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD.
