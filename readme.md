# Lung Cancer Classification

Đồ án Machine Learning sử dụng **Random Forest** và **XGBoost** cho bài toán phân loại ung thư phổi.

## Cấu trúc dự án

```text
project/
├── .github/
│   └── workflows/
│       ├── issue-assigned.yml
│       └── review-label.yml
│
├── data/
│   └── lung_cancer.csv
│
├── notebooks/
│
├── src/
│   ├── models/
│   │   ├── random_forest/
│   │   └── xgboost/
│   ├── evaluation/
│   └── visualization/
│
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Phạm vi của các thư mục

| Thư mục              | Mục đích                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| `.github/workflows/` | Các workflow tự động của GitHub. Không chỉnh sửa nếu không liên quan đến workflow. |
| `data/`              | Dataset được sử dụng cho dự án.                                                    |
| `notebooks/`         | Notebook dùng cho việc khám phá dữ liệu, thử nghiệm và trực quan hóa.              |
| `src/models/`        | Phần cài đặt các mô hình. Mỗi mô hình làm việc trong thư mục riêng.                |
| `src/evaluation/`    | Code và thử nghiệm liên quan đến đánh giá mô hình.                                 |
| `src/visualization/` | Code phục vụ việc trực quan hóa kết quả.                                           |
| `results/`           | Các kết quả, biểu đồ và output được sinh ra trong quá trình thực nghiệm.           |

## Quy tắc làm việc

Khi thực hiện một task, **ưu tiên chỉ chỉnh sửa trong thư mục tương ứng với task đó**.

Ví dụ:

* Random Forest → `src/models/random_forest/`
* XGBoost → `src/models/xgboost/`
* Visualization → `src/visualization/`
* Evaluation → `src/evaluation/`
* Khám phá dữ liệu → `notebooks/`

Không tự ý chỉnh sửa phần việc của thành viên khác, trừ khi task yêu cầu hoặc đã trao đổi với người đang phụ trách phần đó.
