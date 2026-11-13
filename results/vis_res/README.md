# Visualization output

Phân tích các biểu đồ
Thư mục chứa ảnh PNG/SVG do src/visualization/lung_cancer_visualization.ipynb tạo ra. Hai định dạng có cùng nội dung; PNG phù hợp xem nhanh, SVG phù hợp phóng to hoặc đưa vào báo cáo.

Các nhận xét định lượng dưới đây dựa trên dữ liệu và kết quả đang lưu trong repository, chưa dựa trên việc kiểm tra ảnh đã render. Khi chạy lại mô hình hoặc thay dữ liệu, cần cập nhật các nhận xét tương ứng. Quy ước: NO = 0, YES = 1.
01 — Chất lượng dữ liệu
Ảnh: 01_data_quality.png

Dữ liệu gốc có 309 dòng, trong đó 33 dòng trùng hoàn toàn; sau loại trùng còn 276 dòng. Không phát hiện ô trống hoặc giá trị ngoài miền kiểm tra. Vì vậy, xử lý dòng trùng là bước làm sạch có căn cứ; không có bằng chứng cần điền giá trị thiếu ở phiên bản dữ liệu này. Dòng giống nhau không tự chứng minh đó là cùng một người.
02 — Phân bố nhãn
Ảnh: 02_class_balance.png

Dữ liệu mất cân bằng, không phải balanced: sau loại trùng có 238 YES và 38 NO, tương đương 86,2% và 13,8%. Train gồm 190 YES/30 NO; test gồm 48 YES/8 NO. Stratified split giữ tỷ lệ hai lớp tương đối nhất quán nhưng không khắc phục mất cân bằng. Luôn đoán YES cũng đạt 85,7% accuracy trên test.
03 — Phân bố tuổi
Ảnh: 03_age_distribution.png

Tuổi trung vị sau loại trùng là 60,5 ở nhóm NO và 63 ở nhóm YES. Biểu đồ histogram và box plot giúp xem mức chồng lấp và độ phân tán, thay vì chỉ nhìn trung bình. Chênh lệch trung vị này chưa đủ để kết luận tuổi có thể tự phân loại hai nhóm; tuổi hợp lý nhưng ít gặp không nên bị xóa chỉ vì nằm xa phần lớn dữ liệu.
04 — Tỷ lệ triệu chứng theo nhãn
Ảnh: 04_symptom_prevalence.png

Mỗi ô là tỷ lệ trả lời YES cho một triệu chứng trong từng lớp, nên không bị chi phối trực tiếp bởi số người ở hai lớp khác nhau. Ví dụ, FATIGUE xuất hiện ở 69,3% nhóm YES và 47,4% nhóm NO; SHORTNESS OF BREATH lần lượt là 64,3% và 55,3%. Đây là liên hệ mô tả trong mẫu khảo sát, không phải bằng chứng nhân quả.
05 — Tương quan đặc trưng
Ảnh: 05_feature_correlations.png

Màu thể hiện hệ số tương quan Pearson giữa các biến đã mã hóa. Các cặp có trị tuyệt đối lớn đáng được kiểm tra để hiểu quan hệ giữa đặc trưng và khả năng trùng lặp thông tin. Tương quan với nhãn không đồng nghĩa với độ quan trọng mà Random Forest hoặc XGBoost học được, và không tự chứng minh rò rỉ dữ liệu.
06.1 — Confusion matrix của Random Forest
Ảnh: 06_confusion_matrix_1.png

Model dự đoán đúng 8/8 mẫu NO và 44/48 mẫu YES: TN = 8, FP = 0, FN = 4, TP = 44. Recall NO đạt 100%, recall YES đạt 91,67%. Biểu đồ chuẩn hóa theo hàng làm rõ tỷ lệ đúng/sai riêng cho từng lớp dù số lượng hai lớp chênh lệch.
06.2 — Confusion matrix của XGBoost chọn theo AUC/Accuracy/Precision
Ảnh: 06_confusion_matrix_2.png

Ba file được chọn theo AUC, Accuracy và Precision có nội dung model giống nhau nên được gộp thành một nhóm. Kết quả tổng hợp cho ma trận TN = 8, FP = 0, FN = 4, TP = 44, giống số đếm của Random Forest. Cùng confusion matrix không có nghĩa hai model sai trên cùng các dòng hoặc trả về cùng xác suất.
06.3 — Confusion matrix của XGBoost chọn theo Recall/F1
Ảnh: 06_confusion_matrix_3.png

Hai file chọn theo Recall và F1 cũng là cùng một model. Kết quả tổng hợp cho ma trận TN = 7, FP = 1, FN = 2, TP = 46. So với Random Forest, nhóm này giảm hai false negative nhưng tăng một false positive: recall YES tăng lên 95,83%, trong khi recall NO giảm còn 87,5%.
Thứ tự ảnh 06 tương ứng với các artifact hiện tại. Nếu không tải được XGBoost để suy luận, notebook chỉ khôi phục confusion matrix khi có duy nhất một ma trận nguyên khớp các metric và số mẫu từng lớp. Trường hợp đó giả định CSV kết quả thuộc tập test hiện tại, không xác minh được dự đoán từng dòng.

07 — Bảng màu so sánh metric
Ảnh: 07_model_metric_comparison.png

Biểu đồ đặt các model cạnh baseline luôn chọn lớp đa số và sử dụng cùng định nghĩa metric. Random Forest đạt accuracy 92,86%, balanced accuracy 95,83%, macro F1 87,83% và ROC-AUC 0,9766. XGBoost Recall/F1 có accuracy cao hơn (94,64%) nhưng balanced accuracy thấp hơn (91,67%). Không có một nhóm model tốt nhất trên mọi tiêu chí.
08 — So sánh bằng biểu đồ cột
Ảnh: 08_grouped_model_comparison.png

Biểu đồ nhấn mạnh bốn góc nhìn: accuracy, balanced accuracy, macro F1 và ROC-AUC. Accuracy chịu ảnh hưởng của lớp đông mẫu; balanced accuracy và macro F1 cho hai lớp trọng số ngang nhau; ROC-AUC đánh giá khả năng xếp hạng theo xác suất. Không so trực tiếp macro F1 của Random Forest với F1 riêng lớp YES của XGBoost.
09 — So sánh loại lỗi
Ảnh: 09_error_counts.png

Random Forest và XGBoost AUC/Accuracy/Precision có 0 FP, 4 FN; XGBoost Recall/F1 có 1 FP, 2 FN. Biểu đồ giúp trình bày sự đánh đổi giữa hai loại lỗi thay vì chỉ báo accuracy. Trên 56 mẫu test, một dự đoán thay đổi tương đương khoảng 1,79 điểm phần trăm accuracy, nên chênh lệch nhỏ cần được diễn giải thận trọng.
10 — ROC và precision–recall
Ảnh: 10_roc_precision_recall.png

ROC thể hiện sự đánh đổi giữa true-positive rate và false-positive rate qua các ngưỡng. AUC đã lưu lần lượt là 0,9766 cho RF; 0,9661 cho XGBoost AUC/Accuracy/Precision; 0,9714 cho XGBoost Recall/F1. Precision–recall tập trung vào lớp YES; baseline precision đã cao (85,7%) do YES chiếm đa số. AP được tính từ xác suất, không phải F1 hay diện tích hình thang dưới đường PR.

Ảnh chỉ có đường cong cho model có xác suất dự đoán từng mẫu. Nếu không tải được XGBoost, các đường của XGBoost bị bỏ qua; không dựng đường cong từ một số AUC tổng hợp. Không dùng test này để tối ưu ngưỡng rồi xem nó là đánh giá độc lập.
11 — Độ ổn định qua các fold
Ảnh: 11_cv_stability.png

Các đường thể hiện accuracy, ROC-AUC và recall YES qua 5 fold. XGBoost có accuracy khoảng 79,55–90,91% và AUC khoảng 0,8553–0,9737, cho thấy kết quả phụ thuộc đáng kể vào cách chia mẫu. RF tuned được chọn trên chính CV này, nên điểm của nó là kết quả dùng để chọn model, không phải đánh giá nested CV độc lập. Độ lệch chuẩn giữa fold không phải khoảng tin cậy.
12 — Các lỗi chung giữa model
Ảnh: 12_shared_errors.png — chỉ tạo khi có dự đoán từng dòng của ít nhất hai model.

Ô chéo chính là tổng số mẫu model dự đoán sai; ô ngoài đường chéo là số mẫu cả hai model cùng sai. Biểu đồ giúp tìm các mẫu khó chung và xem các model có mắc lỗi khác nhau hay không. Chỉ kết luận từ dự đoán thực tế; hai confusion matrix bằng nhau không đủ để suy ra lỗi chung. Khi thiếu suy luận XGBoost, ảnh này không được tạo.
Lưu ý khi sử dụng kết quả
- RF cuối cùng được refit trên toàn bộ train; các model XGBoost hiện tại được giữ lại từ từng fold. So sánh mô tả các artifact đã lưu, chưa phải thí nghiệm đồng nhất hoàn toàn giữa hai thuật toán.
- Test chỉ có 56 mẫu, gồm 8 NO. Một mẫu NO làm recall NO thay đổi 12,5 điểm phần trăm; chưa nên khẳng định ưu thế tổng quát từ chênh lệch nhỏ.
- Notebook chỉ đọc dữ liệu/kết quả và suy luận bằng model đã lưu, không huấn luyện lại.
