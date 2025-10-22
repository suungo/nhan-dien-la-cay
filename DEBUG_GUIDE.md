# 🔍 Debug Guide - Không hiển thị kết quả sau khi upload ảnh

## 🚨 Vấn đề phổ biến

Khi upload ảnh và nhấn "Kiểm tra" nhưng không hiển thị kết quả, thường do một trong các nguyên nhân sau:

---

## 🔍 Bước 1: Kiểm tra Server Status

### 1.1 Kiểm tra trạng thái server trên UI

- Mở trang web
- Xem phần "Trạng thái Server"
- **Màu xanh**: Server OK ✅
- **Màu đỏ**: Server không kết nối được ❌

### 1.2 Nếu server màu đỏ:

```bash
# Kiểm tra Flask server có chạy không
curl http://localhost:5000/health

# Kết quả mong đợi:
# {"status":"healthy","model_loaded":true,"message":"Plant health detection API is running"}
```

**Nếu lỗi "Connection refused":**

- Flask server chưa chạy
- Chạy: `cd server && python app.py`

---

## 🔍 Bước 2: Kiểm tra Console Logs

### 2.1 Mở Developer Tools

- Nhấn F12 trong browser
- Chuyển sang tab "Console"

### 2.2 Upload ảnh và xem logs

Bạn sẽ thấy các log sau:

**Khi upload ảnh:**

```
🚀 Bắt đầu kiểm tra ảnh: image.jpg 1234567
🔄 Sử dụng Flask server để xử lý ảnh...
🚀 Sending image to Flask server for prediction...
```

**Nếu thành công:**

```
✅ Server prediction successful: {status: "healthy", confidence: 0.85, ...}
✅ Server prediction thành công: {status: "healthy", confidence: 0.85, ...}
```

**Nếu lỗi:**

```
❌ Server prediction failed: Error: ...
❌ Lỗi khi kiểm tra: Error: ...
```

---

## 🔍 Bước 3: Kiểm tra Network Requests

### 3.1 Mở tab Network

- F12 → Network tab
- Upload ảnh và nhấn "Kiểm tra"

### 3.2 Tìm request đến server

- Tìm request đến `localhost:5000/predict`
- Xem Status Code:
  - **200**: Thành công ✅
  - **404**: Endpoint không tồn tại ❌
  - **500**: Lỗi server ❌
  - **CORS error**: Lỗi CORS ❌

### 3.3 Xem Response

Click vào request → Response tab:

```json
{
  "success": true,
  "result": {
    "status": "healthy",
    "confidence": 0.85,
    "healthy_probability": 0.85,
    "unhealthy_probability": 0.15
  }
}
```

---

## 🔍 Bước 4: Debug từng bước

### 4.1 Test server trực tiếp

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/your/image.jpg"
```

### 4.2 Kiểm tra model có load không

```bash
curl http://localhost:5000/model/info
```

**Kết quả mong đợi:**

```json
{
  "input_shape": [null, 224, 224, 3],
  "output_shape": [null, 2],
  "model_loaded": true
}
```

---

## 🛠️ Các lỗi thường gặp và cách sửa

### Lỗi 1: "Server không khả dụng"

**Nguyên nhân:** Flask server chưa chạy
**Cách sửa:**

```bash
cd server
python app.py
```

### Lỗi 2: "CORS error"

**Nguyên nhân:** CORS chưa cấu hình đúng
**Cách sửa:** Sửa `server/app.py`:

```python
CORS(app, origins=["http://localhost:5173"])
```

### Lỗi 3: "Model not loaded"

**Nguyên nhân:** File model không tồn tại
**Cách sửa:** Kiểm tra đường dẫn model trong `server/app.py`

### Lỗi 4: "Connection timeout"

**Nguyên nhân:** Server quá chậm hoặc bị treo
**Cách sửa:** Restart server

### Lỗi 5: "Image too large"

**Nguyên nhân:** Ảnh > 10MB
**Cách sửa:** Nén ảnh hoặc tăng limit trong server

---

## 🔧 Debug Code

### Thêm debug logs vào App.tsx

```typescript
// Trong handleClick function, thêm:
console.log("🔍 Debug info:");
console.log("- Server connected:", serverConnected);
console.log("- Logo file:", logo);
console.log("- Auto recognition:", autoRecognition);
console.log("- Loading state:", isLoading);
```

### Thêm debug logs vào apiService.ts

```typescript
// Trong predictPlantHealth function, thêm:
console.log("📡 Request details:");
console.log("- File name:", file.name);
console.log("- File size:", file.size);
console.log("- File type:", file.type);
console.log("- API URL:", this.baseUrl);
```

---

## 🧪 Test Cases

### Test 1: Server Health

```bash
curl http://localhost:5000/health
```

**Expected:** 200 OK với JSON response

### Test 2: Model Info

```bash
curl http://localhost:5000/model/info
```

**Expected:** 200 OK với model info

### Test 3: Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@test-image.jpg"
```

**Expected:** 200 OK với prediction result

### Test 4: Client Connection

- Mở browser console
- Upload ảnh
- Xem logs có lỗi không

---

## 📋 Checklist Debug

### Trước khi test:

- [ ] Flask server đang chạy
- [ ] Model file đã load
- [ ] Browser console mở
- [ ] Network tab mở

### Khi test:

- [ ] Upload ảnh
- [ ] Xem server status (màu xanh/đỏ)
- [ ] Xem console logs
- [ ] Xem network requests
- [ ] Xem response data

### Nếu vẫn lỗi:

- [ ] Restart Flask server
- [ ] Clear browser cache
- [ ] Kiểm tra file model
- [ ] Kiểm tra CORS config

---

## 🆘 Nếu vẫn không được

### Gửi thông tin debug:

1. **Console logs** (copy/paste)
2. **Network requests** (screenshot)
3. **Server logs** (terminal output)
4. **Browser info** (Chrome/Firefox, version)
5. **OS info** (Windows/Mac/Linux)

### Test với ảnh đơn giản:

- Dùng ảnh nhỏ (< 1MB)
- Format JPG/PNG
- Ảnh rõ nét, không mờ

---

## 🎯 Quick Fix

### Nếu muốn fix nhanh:

1. **Restart tất cả:**

   ```bash
   # Terminal 1: Stop Flask server (Ctrl+C)
   cd server
   python app.py

   # Terminal 2: Restart React
   npm run dev
   ```

2. **Clear browser cache:**

   - Ctrl+Shift+R (hard refresh)
   - Hoặc mở Incognito mode

3. **Test với ảnh mới:**
   - Dùng ảnh khác
   - Đảm bảo ảnh < 2MB

---

**Chúc bạn debug thành công! 🔧**
