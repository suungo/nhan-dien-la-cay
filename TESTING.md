# 🧪 Hướng dẫn Test tích hợp React + Flask

## Bước 1: Khởi động Flask Server

### Windows:

```bash
cd server
run_server.bat
```

### Linux/Mac:

```bash
cd server
chmod +x run_server.sh
./run_server.sh
```

### Kiểm tra server hoạt động:

```bash
curl http://localhost:5000/health
```

**Kết quả mong đợi:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Plant health detection API is running"
}
```

## Bước 2: Khởi động React Client

```bash
npm run dev
```

Truy cập: `http://localhost:5173`

## Bước 3: Test các tính năng

### 3.1 Test kết nối server

- Mở trang web
- Kiểm tra phần "Trạng thái Server" phải hiển thị màu xanh
- Nếu màu đỏ, nhấn "Kiểm tra lại"

### 3.2 Test upload ảnh

1. **Chọn ảnh từ máy tính:**

   - Nhấn "Chọn ảnh"
   - Chọn file ảnh cây trồng
   - Kiểm tra ảnh hiển thị trong preview

2. **Chụp ảnh từ webcam:**
   - Nhấn "Chụp ảnh"
   - Cho phép truy cập camera
   - Chụp ảnh cây trồng
   - Nhấn "Xác nhận"

### 3.3 Test nhận diện AI

1. **Auto Recognition (mặc định):**

   - Upload ảnh
   - Hệ thống tự động gửi lên server
   - Chờ kết quả (1-3 giây)

2. **Manual Recognition:**
   - Tắt "Tự động nhận diện"
   - Upload ảnh
   - Nhấn "Kiểm tra"

### 3.4 Kiểm tra kết quả

Kết quả phải hiển thị:

- **Trạng thái**: Khỏe mạnh / Có bệnh / Không tin cậy
- **Độ tin cậy**: Phần trăm
- **Xác suất chi tiết**: Khỏe mạnh vs Có bệnh
- **Xử lý bởi**: Flask Server + TensorFlow

## Bước 4: Test API trực tiếp

### 4.1 Test health endpoint

```bash
curl -X GET http://localhost:5000/health
```

### 4.2 Test model info

```bash
curl -X GET http://localhost:5000/model/info
```

### 4.3 Test prediction endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/your/image.jpg"
```

## Bước 5: Test lỗi

### 5.1 Test server không chạy

1. Tắt Flask server
2. Refresh React client
3. Kiểm tra hiển thị "Server không khả dụng"
4. Nút "Kiểm tra" phải bị disable

### 5.2 Test ảnh không hợp lệ

1. Upload file không phải ảnh
2. Kiểm tra hiển thị lỗi phù hợp

### 5.3 Test ảnh quá lớn

1. Upload ảnh > 10MB
2. Kiểm tra hiển thị lỗi "File too large"

## Bước 6: Test Performance

### 6.1 Test thời gian phản hồi

- Upload ảnh và đo thời gian từ lúc nhấn "Kiểm tra" đến khi có kết quả
- Thời gian mong đợi: 1-3 giây

### 6.2 Test multiple requests

- Upload nhiều ảnh liên tiếp
- Kiểm tra server xử lý được không bị crash

## Bước 7: Test UI/UX

### 7.1 Test responsive design

- Thay đổi kích thước cửa sổ browser
- Kiểm tra UI hiển thị đúng trên mobile/tablet

### 7.2 Test loading states

- Kiểm tra hiển thị loading khi đang xử lý
- Kiểm tra disable buttons khi đang xử lý

## Kết quả mong đợi

✅ **Thành công nếu:**

- Server khởi động và load model thành công
- React client kết nối được với server
- Upload và nhận diện ảnh hoạt động bình thường
- Hiển thị kết quả chính xác
- UI responsive và user-friendly

❌ **Cần sửa nếu:**

- Server không khởi động được
- Model không load được
- CORS errors
- Ảnh không upload được
- Kết quả không hiển thị
- UI bị lỗi

## Debug Tips

### Xem logs server:

```bash
cd server
python app.py
# Logs hiển thị trong console
```

### Xem logs client:

- Mở Developer Tools (F12)
- Xem tab Console
- Tìm các log bắt đầu với emoji (🚀, ✅, ❌)

### Kiểm tra network requests:

- Mở Developer Tools (F12)
- Xem tab Network
- Kiểm tra requests đến `localhost:5000`
