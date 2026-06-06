# NHẬN DIỆN GIỚI TÍNH QUA CAMERA TRONG THỜI GIAN THỰC

Họ và tên: Nguyễn Trung Thành   MSSV: 2421060077

- **Bài toán:** Sử dụng camera để nhận diện giới tính là giới tính nam hay giới tính nữ.
- **Đầu vào:** trực tiếp camera điện thoại qua app DroidCam.
- **Đầu ra:** Khung cắt mặt và hiển thị phần trăm dự đoán chính xác và giới tính Nam (khung xanh) và giới tính Nữ (khung đỏ).
- **Nguồn dataset sử dụng:** UTK Face (gõ UTK Face trên Kaggle có thể tìm ra dataset) và AFAD-Lite (gõ AFAD-Lite trên Kaggle có thể tìm ra dataset này).
- **Tóm tắt dự án:** Dự án sẽ được huấn luyện bằng cách đọc dữ liệu được định dạng trong 2 nguồn dataset trên. Ta viết đoạn code để giúp máy đọc nhãn và những quy ước theo nguồn dataset sử dụng. VD: ảnh của UTK Face thì giới tính của nó được quy ước theo tên tuổi_giới tính_chủng tộc_thời gian.jpg nhãn là 0 là giới tính Nam còn 1 là giới tính Nữ; với AFAD sẽ chia theo file chính là Tuổi và trong file đó sẽ có 2 file con là 111 tương ứng với Nam và 112 tương ứng với Nữ. Huấn luyện thuật toán Logistic Regression – phân loại nhị phân, chỉ in ra 2 kết quả 0 và 1. Ta viết code để có thể khởi tạo Opencv – đọc xử lý video- và nạp model Yolo Face – phát hiện chính xác khuôn mặt. Sau đó ta dùng OpenCV để vẽ khung chữ nhật, gắn tên, tỷ lệ phần trăm dự đoán chính xác và thêm nút để tắt chương trình.
- **Cách cài đặt các mô hình:**
  + Tải Yolo Face trên GitHub
  + Tải python bản 3.11 để có thể chạy Yolo Face 1 cách mượt nhất
  + Tạo ra môi trường ảo bằng lệnh :
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
  + Tải các thư viện cần thiết:
    ```bash
    pip install opencv-python numpy scikit-learn joblib
    ```
- **Cách sử dụng:** Sau khi có 2 đoạn code huấn luyện và code chạy thì ta ấn nút RUN hoặc biểu tượng tam giác nằm ngang hay là ta viết lệnh dưới Terminal của Visual Code Studio(nếu sử dụng):
 `python [tên file huấn luyện hoặc chạy lưu].`
- Do dữ liệu của file mô hình ‘gender_model.pkl’ vượt quá với mức cho phép của GitHub nên em xin phép gửi link ở đây:
[Bấm vào đây để xem file.pkl](https://drive.google.com/file/d/1Xg3TSEeBU7wp9woKV3f8jZyRghdLYlC0/view?usp=sharing)
- Video Demo của hệ thống:
[Bấm vào đây để xem Demo](https://drive.google.com/file/d/1pFwmpbFUowK2r3oW2GNMLB5IlbicMiXR/view?usp=sharing)
- **Tại sao lại chọn mô hình này?:** Mô hình này là bước đệm để có thể tạo thêm nhiều loại dạng như vậy nữa có thể kể đến như sử dụng Deep Learning hay rộng hơn là phân biệt qua vóc dáng, tuổi,...
- **Thông tin liên hệ:** 2421060077@student.humg.edu.vn
