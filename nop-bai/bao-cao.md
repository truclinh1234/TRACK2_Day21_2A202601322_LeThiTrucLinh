# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | Lê Thị Trúc Linh |
| MSSV | 2A202601322 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/___/___ |
| Ngày nộp | ___ |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 200 | 0.2 | 3 | 0.7032 | 0.870 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 đạt f1_score cao nhất (0.7149), vượt ngưỡng chất lượng 0.65 của lab. Đáng chú ý, lần chạy có accuracy cao nhất lại là lần 1 (0.878), không trùng với lần có f1_score cao nhất — cho thấy accuracy có thể đánh lừa khi chọn mô hình cho dữ liệu mất cân bằng, vì nó bị chi phối bởi lớp đa số (thu nhập thấp). Về đánh đổi n_estimators/learning_rate: lần 2 dùng cả hai giá trị thấp (n_estimators=50, learning_rate=0.05) khiến mô hình chưa học đủ, f1_score giảm mạnh xuống 0.6051. Khi tăng độ sâu cây (max_depth=5) thay vì tăng learning_rate (lần 4, learning_rate=0.2), mô hình học tốt hơn trên lớp thiểu số, cho thấy độ sâu cây quan trọng hơn tốc độ học trong bài toán này.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

<!-- Khoảng 120 - 150 từ. -->

Tập dữ liệu Adult Income có phân bố lớp mất cân bằng: chỉ 24,8% mẫu thuộc lớp thu nhập cao (>50K), 75,2% còn lại thuộc lớp thu nhập thấp. Với tỷ lệ này, một mô hình vô dụng luôn trả lời "thu nhập thấp" cho mọi đầu vào vẫn đạt accuracy 0,752 — con số trông rất cao nhưng gây hiểu nhầm nghiêm trọng, vì mô hình đó không bắt được một trường hợp thu nhập cao nào (f1_score = 0). Ngược lại, f1_score của lớp dương là trung bình điều hòa giữa precision và recall trên đúng lớp thiểu số cần dự đoán, nên phản ánh chính xác khả năng mô hình nhận diện nhóm thu nhập cao — điều accuracy không đo được. Vì lý do này, lab không dùng `average="weighted"` hay `average="macro"` khi gọi `f1_score`, bởi hai cách tính đó gộp cả lớp đa số vào công thức trung bình, kéo điểm số lên cao giả tạo và làm mất hoàn toàn ý nghĩa của ngưỡng chất lượng.

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| ___ | ___ | ___ |
| ___ | ___ | ___ |
| ___ | ___ | ___ |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
