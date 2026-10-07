# XAI Stability on Telco Customer Churn
Đánh giá độ ổn định của SHAP và LIME trên dữ liệu dạng bảng
Dataset: Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

## Dữ liệu

- **Nguồn:** Telco Customer Churn (Kaggle, tác giả blastchar), gồm 7.043 khách hàng và 21 cột.
- **Biến mục tiêu:** `Churn` (khách có rời bỏ dịch vụ hay không).
- **Làm sạch:** 11 dòng thiếu `TotalCharges` đều là khách mới (`tenure = 0`), điền 0. Bỏ cột `customerID`, đổi `Churn` thành 0/1. Giữ nguyên 22 dòng trùng.

## Kết quả EDA

- **Dữ liệu mất cân bằng vừa phải:** 26,5% khách rời đi (1.869 / 7.043). Khi chia dữ liệu cần giữ nguyên tỉ lệ này, và đánh giá mô hình không chỉ dựa vào accuracy.
- **Một số feature liên hệ rõ với churn:** tỉ lệ churn là 43% với hợp đồng theo tháng, 11% với hợp đồng 1 năm và 3% với hợp đồng 2 năm. Đây sẽ là mốc để đối chiếu với kết quả SHAP/LIME ở các bước sau.