# XAI Stability on Telco Customer Churn
Đánh giá độ ổn định của SHAP và LIME trên dữ liệu dạng bảng
Dataset: Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

## Dữ liệu

- **Nguồn:** Telco Customer Churn (Kaggle, tác giả blastchar), gồm 7.043 khách hàng và 21 cột.
- **Biến mục tiêu:** `Churn` (khách có rời bỏ dịch vụ hay không).
- **Làm sạch:** 11 dòng thiếu `TotalCharges` đều là khách mới (`tenure = 0`), điền 0. Bỏ cột `customerID`, đổi `Churn` thành 0/1. Giữ nguyên 22 dòng trùng.

## Kết quả EDA

- **Dữ liệu mất cân bằng vừa phải:** 26,5% khách rời đi (1.869 / 7.043) (Khi chia dữ liệu cần giữ nguyên tỉ lệ này, và đánh giá mô hình không chỉ dựa vào accuracy)
- **Một số feature liên hệ rõ với churn:** tỉ lệ churn là 43% với hợp đồng theo tháng, 11% với hợp đồng 1 năm và 3% với hợp đồng 2 năm. (Đây là mốc để đối chiếu với kết quả SHAP/LIME ở các bước sau)

## Chia dữ liệu và tiền xử lý 
Chia dữ liệu thành 80/20 cho tập train và test. Dữ nguyên tỉ lệ churn (26.5%)
- **Mã hóa:** 15 cột dạng chữ được tách thành các cột 0/1 (one-hot), 3 cột số (`tenure`, `MonthlyCharges`, `TotalCharges`) được chuẩn hóa. Kết quả là 45 feature, mỗi feature giữ một cái tên đọc được (ví dụ `Contract_Month-to-month`) để có thể đọc kết quả của SHAP và LIME.
- **Tránh rò rỉ dữ liệu:** các thông số chuẩn hóa chỉ được học từ tập train, rồi áp dụng nguyên xi cho tập test.
- **Vai trò trong đề tài:** tập train làm dữ liệu tham chiếu cho các phương pháp giải thích, tập test cung cấp những khách hàng cụ thể cần giải thích. Cố định cách chia dữ liệu giúp mọi biến động quan sát được ở các bước sau đến từ chính SHAP/LIME, không đến từ việc dữ liệu bị chia khác đi.

## Mô hình baseline
Ba mô hình họ khác nhau dược train với tham số cố định: Logistic Regression (tuyến tính), Random Forest (nhiều cây chạy song song) và XGBoost (nhiều cây học nối tiếp). Dự án không tinh chỉnh sâu, vì mục tiêu là đánh giá lời giải thích chứ không phải tìm mô hình tốt nhất.
- **Cả ba đều học được tín hiệu thật:** accuracy khoảng 0,80, cao hơn mốc 0,735 của một mô hình luôn đoán "ở lại".
- **Chất lượng ngang nhau:** ROC-AUC của ba mô hình xấp xỉ 0,84, nên nếu độ ổn định của giải thích khác nhau giữa các mô hình thì điều đó không thể quy cho việc mô hình này giỏi hơn mô hình kia.
- **Hạn chế:** recall chỉ khoảng 0,52-0,56 (mô hình bỏ sót gần một nửa khách rời đi) do dữ liệu mất cân bằng và dùng ngưỡng 0,5. Dự án giữ nguyên ngưỡng này.