# Lung Cancer Classification — Random Forest & XGBoost

Dự án bài tập lớn môn **Học máy (CO3117)**, thực hiện bài toán **phân loại nhị phân nhãn ung thư phổi** (`LUNG_CANCER`: `YES`/`NO`) từ dữ liệu khảo sát về nhân khẩu học, hành vi và triệu chứng. Dự án triển khai **Random Forest**, **XGBoost** và baseline **DummyClassifier**, đồng thời lưu kết quả đánh giá và biểu đồ để đối chiếu.

> **Giới hạn sử dụng:** Đây là thực nghiệm học máy trên dữ liệu khảo sát quy mô nhỏ; **không phải** công cụ chẩn đoán hoặc sàng lọc y tế đã được kiểm định lâm sàng.

## 1. Dữ liệu và tiền xử lý

- **Nguồn dữ liệu:** bộ dữ liệu *Lung Cancer* từ Kaggle, lưu tại `data/raw/survey_lung_cancer.csv`.
- **Dữ liệu ban đầu:** 309 bản ghi, 16 cột; trong đó có **15 đặc trưng đầu vào** và nhãn `LUNG_CANCER`.
- **Tiền xử lý:** kiểm tra schema và miền giá trị, loại 33 dòng trùng hoàn toàn, mã hóa các biến phân loại thành giá trị số. Dữ liệu còn **276 bản ghi**: 238 `YES` (86,23%) và 38 `NO` (13,77%). Không phát hiện giá trị khuyết hay ngoài miền kiểm tra ở dữ liệu hiện có.
- **Mã hóa nhãn:** `YES = 1`, `NO = 0`.
- **Chia dữ liệu:** stratified train/test với `test_size=0.2`, `random_state=42`: **220 mẫu train** (190 YES, 30 NO) và **56 mẫu test** (48 YES, 8 NO).

**Lưu ý:** Bộ dữ liệu **mất cân bằng lớp**, không phải balanced dataset. Phân chia stratified chỉ giữ tỷ lệ nhãn tương đối nhất quán giữa train/test, không tự cân bằng lại dữ liệu.

## 2. Tech stack

- **Ngôn ngữ:** Python.
- **ML:** scikit-learn (`DummyClassifier`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedKFold`, metrics), XGBoost.
- **Data processing:** pandas, NumPy, SciPy.
- **Visualization:** Matplotlib, Seaborn, Jupyter Notebook.
- **Model/artifact:** `joblib` (Random Forest), XGBoost JSON, CSV, JSON, PNG và SVG.

Các dependency chính có trong [`requirements.txt`](requirements.txt). **Jupyter chưa được khai báo trong file này**; cần cài thêm nếu chạy notebook bằng JupyterLab/Notebook thay vì VS Code có môi trường notebook tương thích. `requirements.txt` dùng các khoảng phiên bản, không phải một môi trường đã khóa phiên bản tuyệt đối; khi tái lập mô hình đã lưu, cần chú ý tính tương thích thư viện.

## 3. Phương pháp

1. Kiểm tra, làm sạch, mã hóa dữ liệu; loại trùng trước khi chia train/test.
2. Dùng **5-fold stratified cross-validation** trên tập train.
3. **Random Forest:** đánh giá baseline và sử dụng `GridSearchCV` để chọn cấu hình theo balanced accuracy; sau đó fit lại mô hình được chọn trên toàn bộ 220 mẫu train và lưu `results/random_forest/rf_model.joblib`.
4. **XGBoost:** dùng cấu hình cố định (`binary:logistic`, 100 estimators, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `random_state=42`), cùng `scale_pos_weight = N_NO / N_YES` tính theo từng fold. Script lưu **5 file checkpoint JSON tương ứng với tiêu chí lựa chọn** AUC, Accuracy, Precision, Recall và F1; các file này không nhất thiết chứa 5 mô hình khác nhau.
5. Testing sử dụng **các model đã lưu** và cùng tập holdout 56 mẫu. Notebook visualization đọc dữ liệu, model và các bảng kết quả để dựng biểu đồ, đồng thời kiểm tra tính nhất quán của một số metric.

**Giới hạn phép so sánh:** Random Forest được fit lại trên toàn bộ 220 mẫu train; checkpoint XGBoost được chọn từ từng fold và fit trên tập con train của fold. Vì vậy bảng holdout mô tả **kết quả của các artifact đã lưu**, không phải thí nghiệm đối chứng với quy trình huấn luyện tương đương hoàn toàn giữa hai thuật toán.

## 4. Kết quả thực nghiệm đã lưu

Các số sau được ghi trong bản README của dự án và các artifact kết quả đi kèm. **Chạy lại training có thể tạo ra artifact khác**, đặc biệt khi môi trường thư viện thay đổi.

### 4.1. Cross-validation (5 fold trên tập train)

| Phương pháp | Accuracy | Balanced accuracy | Recall NO | Recall YES |
|---|---:|---:|---:|---:|
| Dummy — dự đoán lớp đa số | 86,36% | 50,00% | 0,00% | 100,00% |
| Random Forest cơ sở | 89,55% | 71,49% | 46,67% | 96,32% |
| Random Forest tinh chỉnh | 85,45% | **77,54%** | **66,67%** | 88,42% |
| XGBoost cấu hình cố định | 85,00% | 75,88% | 63,33% | 88,42% |

### 4.2. Holdout test (56 mẫu: 8 NO, 48 YES)

| Phương pháp | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy — lớp đa số | 85,71% | 50,00% | 0,4615 | 0,5000 | 0 | 8 | 0 | 48 |
| Random Forest | 92,86% | **95,83%** | 0,8783 | **0,9766** | 8 | 0 | 4 | 44 |
| XGBoost A (AUC/Accuracy/Precision) | 92,86% | **95,83%** | 0,8783 | 0,9661 | 8 | 0 | 4 | 44 |
| XGBoost B (Recall/F1) | **94,64%** | 91,67% | **0,8960** | 0,9714 | 7 | 1 | 2 | 46 |

XGBoost B đạt accuracy cao nhất trên tập test này; Random Forest có ROC-AUC cao nhất. Chênh lệch 94,64% và 92,86% tương đương **một dự đoán đúng** trên 56 mẫu. Tập test có 8 mẫu NO, nên không nên suy rộng kết quả thành ưu thế ổn định hoặc hiệu quả sử dụng lâm sàng.

**Phân biệt metric:** File `results/XGBoost/xgb_final_test_results.csv` lưu F1/Recall/Precision theo lớp dương (`YES`) của XGBoost; **Macro F1** trong bảng so sánh được tính từ dự đoán hai lớp trong notebook visualization, không nên lấy trực tiếp cột `F1-Score` của CSV để so với macro F1 của Random Forest.

## 5. Cấu trúc repository thực tế

```text
.
├── .github/workflows/                 # GitHub Actions
├── data/
│   ├── raw/
│   │   └── survey_lung_cancer.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── common.py                      # Tiện ích chung cho Random Forest
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── random_forest/
│   │   │   ├── train.py
│   │   │   └── test.py
│   │   └── XGBoost/                   # Chú ý chữ hoa/thường
│   │       ├── train.py
│   │       ├── test.py
│   │       └── saved/                  # Các checkpoint XGBoost JSON
│   │           ├── best_xgb_auc_lung_cancer.json
│   │           ├── best_xgb_accuracy_lung_cancer.json
│   │           ├── best_xgb_precision_lung_cancer.json
│   │           ├── best_xgb_recall_lung_cancer.json
│   │           └── best_xgb_f1_lung_cancer.json
│   ├── evaluation/                     # Hiện chỉ chứa .gitkeep
│   └── visualization/
│       └── lung_cancer_visualization.ipynb
├── results/
│   ├── random_forest/
│   │   ├── rf_model.joblib            # Model RF đã huấn luyện
│   │   ├── training_metadata.json
│   │   ├── baseline_summary.csv
│   │   ├── dummy_cv.csv
│   │   ├── rf_baseline_cv.csv
│   │   ├── search_results.csv
│   │   └── holdout/                   # Metrics và dự đoán trên test
│   │       ├── test_metrics.json
│   │       ├── classification_report.json
│   │       ├── confusion_matrix.csv
│   │       └── test_predictions.csv
│   ├── XGBoost/
│   │   ├── xgb_kfold_results.csv
│   │   └── xgb_final_test_results.csv
│   └── vis_res/                      # Hình PNG/SVG, bảng và manifest
│       ├── README.md                 # Giải thích từng hình
│       ├── visualization_manifest.json
│       └── ...
├── requirements.txt
└── readme.md
```

**Quy ước quan trọng:** Trong **phiên bản code hiện tại**, XGBoost **ghi và đọc model JSON tại `src/models/XGBoost/saved/`**; `results/XGBoost/` chứa **CSV kết quả**. Đây là đường dẫn thực tế của `src/models/XGBoost/train.py`, `test.py` và notebook visualization. **Không cần di chuyển các JSON này sang `results/` để chạy code hiện có.** Các JSON trong `results/random_forest/` lại chủ yếu là metadata và metrics, không phải checkpoint XGBoost.

## 6. Cài đặt

Thực hiện từ **thư mục gốc của repository**. Cần phiên bản Python tương thích với dependencies trong `requirements.txt` (khuyến nghị tạo môi trường ảo riêng).

```bash
git clone https://github.com/hufzohel/Lung_Cancer_Classification_Random-Forest_and_XGBoost.git
cd Lung_Cancer_Classification_Random-Forest_and_XGBoost
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Cài đặt:

```bash
python -m pip install -r requirements.txt
# Nếu muốn chạy notebook bằng JupyterLab:
python -m pip install jupyterlab ipykernel
```

> **Tương thích model:** File Random Forest (`.joblib`) là artifact của một môi trường scikit-learn cụ thể. Cài các phiên bản khác dù thỏa `requirements.txt` có thể tạo cảnh báo hoặc lỗi tương thích khi tải model. Thông tin môi trường training RF có trong `results/random_forest/training_metadata.json`.

## 7. Hướng dẫn chạy và lưu ý về việc ghi đè

**Cảnh báo:** Repository **đã kèm** dữ liệu processed, model và kết quả thực nghiệm. Chạy lại `preprocess.py`, các script `train.py`, `test.py` hoặc `Run All` notebook **có thể ghi đè file hiện có**. Nếu chỉ cần đọc, xem số liệu hoặc trình bày báo cáo, **không cần chạy lại các bước này**.

**Để kiểm thử an toàn:** sao chép toàn bộ repository sang một thư mục khác trước; chỉ thực hiện các lệnh ghi file trên **bản sao**. Kiểm tra kết quả và so sánh với artifact gốc trước khi quyết định cập nhật bản chính thức.

### 7.1. Tiền xử lý (ghi lại `data/processed/train.csv`, `test.csv`)

```bash
python -m src.data.preprocess
```

Script sử dụng các đường dẫn tương đối `data/raw/survey_lung_cancer.csv` và `data/processed/`, nên chạy từ thư mục gốc; **`data/processed/` phải tồn tại** trong phiên bản code hiện tại (đã có trong ZIP).

### 7.2. Random Forest training/testing

**Lưu ý import trong bản code hiện tại:** Cả `src/models/random_forest/train.py` và `test.py` có `from common import ...`, trong khi `common.py` nằm ở `src/common.py`. Do đó, với lệnh chạy module từ repository root, cần thêm `src` vào `PYTHONPATH` **hoặc** sửa import trong code thành `from src.common import ...` và chạy theo `python -m ...`. README này **không giả định code đã được sửa**.

Chạy **nguyên trạng code hiện có**, trên macOS/Linux:

```bash
PYTHONPATH=src python -m src.models.random_forest.train
PYTHONPATH=src python -m src.models.random_forest.test
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m src.models.random_forest.train
python -m src.models.random_forest.test
```

- **Training** ghi lại model `results/random_forest/rf_model.joblib`, CV CSV và metadata.
- **Testing** nạp model `.joblib` và `data/processed/test.csv`, ghi lại các file trong `results/random_forest/holdout/`.
- Khi **chỉ cần xem kết quả đã lưu**, đọc các CSV/JSON hiện có; không phải chạy training/testing.

### 7.3. XGBoost training/testing

Chạy từ repository root:

```bash
python -m src.models.XGBoost.train
python -m src.models.XGBoost.test
```

- **Training** đọc `data/processed/train.csv`, ghi `results/XGBoost/xgb_kfold_results.csv` và **ghi đè 5 model JSON** trong `src/models/XGBoost/saved/`.
- **Testing** đọc `data/processed/test.csv`, **nạp 5 model JSON từ `src/models/XGBoost/saved/`**, ghi `results/XGBoost/xgb_final_test_results.csv`.
- Thư mục `src/models/XGBoost/saved/` **phải tồn tại** trước khi training; bản ZIP hiện tại đã có sẵn.
- Trên hệ thống phân biệt chữ hoa/chữ thường (Linux/macOS tùy cấu hình), giữ đúng tên thư mục **`XGBoost`**, không đổi thành `xgboost`.

**Quan trọng:** JSON trong `src/models/XGBoost/saved/` là **model artifacts** được `load_model()` sử dụng để testing, không chỉ là file số liệu tham khảo. Ngược lại, các file CSV tại `results/XGBoost/` phục vụ lưu kết quả và đối chiếu báo cáo.

### 7.4. Visualization notebook

Notebook: `src/visualization/lung_cancer_visualization.ipynb`.

```bash
jupyter lab
```

Mở notebook và chọn **Run All**. Cell khởi tạo dò repository root dựa trên sự tồn tại của `data/raw/survey_lung_cancer.csv`; nếu không tìm được, thiết lập `ROOT_OVERRIDE` trong cell đầu tiên. Notebook đọc model RF, checkpoint XGBoost, dữ liệu và metrics đã lưu; **không training hoặc tuning lại model**. Tuy nhiên, khi chạy notebook vẫn **ghi lại các ảnh PNG/SVG, CSV dẫn xuất và `results/vis_res/visualization_manifest.json`**.

Xem [`results/vis_res/README.md`](results/vis_res/README.md) để biết ý nghĩa từng hình. Nếu không nạp được JSON XGBoost, notebook có nhánh dùng kết quả tổng hợp từ CSV; chế độ này **không tương đương suy luận từ model** và có thể không tạo đầy đủ đường ROC/PR hay so sánh lỗi theo từng mẫu.

## 8. Đối chiếu kết quả mà không chạy lại training

Có thể đọc trực tiếp:

| Mục đích | File |
|---|---|
| Cấu hình, thông tin training Random Forest | `results/random_forest/training_metadata.json` |
| Model Random Forest đã lưu | `results/random_forest/rf_model.joblib` |
| RF holdout metrics và confusion matrix | `results/random_forest/holdout/test_metrics.json`, `confusion_matrix.csv` |
| RF dự đoán theo từng mẫu | `results/random_forest/holdout/test_predictions.csv` |
| XGBoost 5-fold CV | `results/XGBoost/xgb_kfold_results.csv` |
| XGBoost holdout metrics (5 checkpoint) | `results/XGBoost/xgb_final_test_results.csv` |
| Model XGBoost đã lưu | `src/models/XGBoost/saved/*.json` |
| Biểu đồ, bảng tổng hợp, thông tin nguồn | `results/vis_res/` |

Việc đọc những file này **không ghi đè kết quả**. Nếu cần chạy suy luận hoặc kiểm chứng bằng code, nên thực hiện trên bản sao của repository.

## 9. Giới hạn và khả năng tái lập

- Tập test chỉ gồm 56 mẫu, với 8 mẫu NO; một dự đoán sai làm recall NO thay đổi **12,5 điểm phần trăm**.
- Dữ liệu khảo sát nhỏ và mất cân bằng; kết quả chưa được xác thực trên dữ liệu lâm sàng thực tế.
- Random Forest được refit trên toàn bộ training set; model XGBoost được lưu từ các fold CV; không so sánh chúng như thể có cùng quy trình fit.
- Các script training/testing có thao tác ghi đè artifact hiện có; giữ một bản sao kết quả trước khi chạy lại.
- Các đường dẫn tương đối của preprocessing và XGBoost phụ thuộc vào **current working directory**; chạy từ repository root như hướng dẫn.
- Môi trường chưa được khóa phiên bản đầy đủ; khác biệt phiên bản thư viện có thể ảnh hưởng khả năng nạp model hoặc kết quả training lại.

## 10. Tài liệu tham khảo

- Báo cáo nhóm: *Phân loại nhãn ung thư phổi từ dữ liệu khảo sát*, CO3117 (11/2026).
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
