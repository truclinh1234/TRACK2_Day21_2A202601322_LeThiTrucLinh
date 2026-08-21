# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Lê Thị Trúc Linh |
| MSSV | 2A202601322 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/truclinh1234/TRACK2_Day21_2A202601322_LeThiTrucLinh |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 200 | 0.2 | 3 | 0.7032 | 0.870 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 đạt f1_score cao nhất (0.7149), vượt ngưỡng chất lượng 0.65 của lab. Đáng chú ý, lần chạy có accuracy cao nhất lại là lần 1 (0.878), không trùng với lần có f1_score cao nhất — cho thấy accuracy có thể đánh lừa khi chọn mô hình cho dữ liệu mất cân bằng, vì nó bị chi phối bởi lớp đa số (thu nhập thấp). Về đánh đổi n_estimators/learning_rate: lần 2 dùng cả hai giá trị thấp (n_estimators=50, learning_rate=0.05) khiến mô hình chưa học đủ, f1_score giảm mạnh xuống 0.6051. Khi tăng độ sâu cây (max_depth=5) thay vì tăng learning_rate (lần 4, learning_rate=0.2), mô hình học tốt hơn trên lớp thiểu số, cho thấy độ sâu cây quan trọng hơn tốc độ học trong bài toán này.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult Income có phân bố lớp mất cân bằng: chỉ 24,8% mẫu thuộc lớp thu nhập cao (>50K), 75,2% còn lại thuộc lớp thu nhập thấp. Với tỷ lệ này, một mô hình vô dụng luôn trả lời "thu nhập thấp" cho mọi đầu vào vẫn đạt accuracy 0,752 — con số trông rất cao nhưng gây hiểu nhầm nghiêm trọng, vì mô hình đó không bắt được một trường hợp thu nhập cao nào (f1_score = 0). Ngược lại, f1_score của lớp dương là trung bình điều hòa giữa precision và recall trên đúng lớp thiểu số cần dự đoán, nên phản ánh chính xác khả năng mô hình nhận diện nhóm thu nhập cao — điều accuracy không đo được. Vì lý do này, lab không dùng `average="weighted"` hay `average="macro"` khi gọi `f1_score`, bởi hai cách tính đó gộp cả lớp đa số vào công thức trung bình, kéo điểm số lên cao giả tạo và làm mất hoàn toàn ý nghĩa của ngưỡng chất lượng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `mlflow` báo lỗi `ModuleNotFoundError: pkg_resources` khi chạy cục bộ và cả trong CI | Bản `setuptools` mới nhất đã loại bỏ hẳn module `pkg_resources` mà `mlflow==2.13.0` vẫn phụ thuộc | Ghim `setuptools<81` trong `requirements.txt` để cài đúng bản còn giữ `pkg_resources` |
| IAM user AWS bị chặn quyền tạo S3 bucket và EC2 instance type mặc định (`t2.micro`) không hợp lệ | Tài khoản AWS cá nhân mới, IAM user chưa được cấp policy S3, và free tier hiện dùng `t3.micro` thay vì `t2.micro` | Gắn policy `AmazonS3FullAccess` qua IAM Console và đổi sang instance type `t3.micro` |
| Push code lần đầu không tự kích hoạt GitHub Actions dù path đã khớp filter | Repo GitHub Actions cần được "khởi động" lần đầu khi mới tạo | Chạy thủ công qua nút "Run workflow" (`workflow_dispatch`) một lần; các lần push sau đó tự kích hoạt bình thường |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.882 |

**Nhận xét:** f1_score tăng nhẹ từ 0.7149 lên 0.7354 (+0.0205) khi tăng gấp đôi dữ liệu huấn luyện (22.361 lên 44.722 mẫu). Điều này hợp lý vì `train_batch2` tuy cùng phân phối với `train_batch1` (cả hai đều chia ngẫu nhiên từ cùng nguồn dữ liệu Adult Income), nhưng việc có thêm mẫu vẫn giúp mô hình GradientBoosting ước lượng ranh giới quyết định ổn định hơn một chút, đặc biệt với lớp thiểu số (thu nhập cao) vốn ít mẫu. Mức tăng khiêm tốn (không đột biến) cho thấy mô hình đã học gần hết thông tin có thể khai thác từ 22.361 mẫu đầu, đúng như dự đoán của tài liệu lab. Điều quan trọng hơn con số là quy trình tự động đã hoạt động đúng: từ commit dữ liệu đến mô hình mới được triển khai trên VM, không cần can thiệp thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: quét ngưỡng 0.1-0.9, tìm được ngưỡng tối ưu 0.3 cho F1 = 0.7537, cao hơn ngưỡng mặc định 0.5 (F1 = 0.7354).
- [x] Bonus 3 - Báo cáo precision / recall tự động: ghi `outputs/detail.txt` với confusion matrix và precision/recall từng lớp; upload qua `actions/upload-artifact` trong job Train. Recall lớp thu nhập cao chỉ 0.66 (thấp hơn precision 0.83) — bỏ sót người thu nhập cao tốn kém hơn vì ảnh hưởng trực tiếp đến việc họ bị bỏ lỡ ưu đãi/dịch vụ nhắm đúng đối tượng.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: thêm bước `Compare with previous model version` tải `report.json` cũ từ S3, so sánh F1 mới/cũ trước khi cho phép Release; nếu F1 mới thấp hơn, Quality Gate tự chặn.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: kiểm tra tỷ lệ lớp dương so với 24.8%, in cảnh báo nếu lệch quá 5 điểm phần trăm, ghi `positive_ratio` vào `report.json`.
