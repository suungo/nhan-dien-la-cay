# 🌱 Hệ thống nhận diện sức khỏe cây trồng

Ứng dụng web sử dụng AI để nhận diện sức khỏe cây trồng qua hình ảnh, được xây dựng với React + TypeScript + Flask Server + TensorFlow.

## ✨ Tính năng

- 📸 **Upload ảnh** - Tải ảnh từ máy tính hoặc chụp từ webcam
- 🤖 **AI Recognition** - Sử dụng ResNet50 model (.h5) trên Flask server
- 🔄 **Auto Recognition** - Tự động nhận diện sau khi upload
- 📊 **Confidence Score** - Hiển thị độ tin cậy của kết quả
- 🎯 **Real-time Camera** - Chụp ảnh trực tiếp từ webcam
- 📱 **Responsive Design** - Giao diện thân thiện trên mọi thiết bị
- 🖥️ **Server-side Processing** - Xử lý AI trên server để tăng hiệu suất
- 🔗 **Real-time Status** - Hiển thị trạng thái kết nối server

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống

- Node.js >= 16.0.0
- npm >= 8.0.0
- Python >= 3.8
- Trình duyệt hiện đại (Chrome, Firefox, Safari, Edge)

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd view-kp
```

### Bước 2: Cài đặt React client

```bash
npm install
```

### Bước 3: Chuẩn bị model .h5

Đặt file model .h5 vào đường dẫn: `D:\web-kpdl\vegetable_classifier_resnet50_final (1).h5`

### Bước 4: Chạy Flask server

**Windows:**

```bash
cd server
run_server.bat
```

**Linux/Mac:**

```bash
cd server
chmod +x run_server.sh
./run_server.sh
```

**Hoặc chạy thủ công:**

```bash
cd server
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Bước 5: Chạy React client

```bash
# Development mode
npm run dev

# Build cho production
npm run build

# Preview build
npm run preview
```

### Bước 6: Truy cập ứng dụng

- React client: `http://localhost:5173`
- Flask server: `http://localhost:5000`

## 📁 Cấu trúc project

```
view-kp/
├── src/
│   ├── App.tsx              # Component chính
│   ├── services/
│   │   └── apiService.ts    # API client cho Flask server
│   └── assets/              # Static assets
├── server/                  # Flask server
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── run_server.bat      # Windows startup script
│   ├── run_server.sh       # Linux/Mac startup script
│   └── README.md           # Server documentation
├── public/                 # Public assets
└── package.json
```

## 🧠 AI Model

### ResNet50 Model (.h5)

- **Path**: `D:\web-kpdl\vegetable_classifier_resnet50_final (1).h5`
- **Format**: Keras .h5 model
- **Purpose**: Nhận diện sức khỏe cây trồng
- **Input**: Ảnh cây trồng (JPG, PNG, etc.) - 224x224x3
- **Output**:
  - `healthy` - Cây khỏe mạnh
  - `unhealthy` - Cây có dấu hiệu bệnh
  - `unreliable` - Model không tin cậy

### Server-side Processing

Model được xử lý trên Flask server với TensorFlow:

```python
# Load model trong Flask server
model = tf.keras.models.load_model('path/to/model.h5')

# Predict từ ảnh
predictions = model.predict(preprocessed_image)
```

### API Endpoints

- `GET /health` - Kiểm tra trạng thái server
- `POST /predict` - Dự đoán sức khỏe cây từ ảnh
- `GET /model/info` - Thông tin model
- `POST /model/reload` - Reload model

### Client-side Integration

```typescript
// Gửi ảnh lên server
const result = await apiService.predictPlantHealth(imageFile);

// Kết quả trả về
{
  "status": "healthy|unhealthy|unreliable",
  "confidence": 0.85,
  "healthy_probability": 0.85,
  "unhealthy_probability": 0.15,
  "timestamp": 1699123456789
}
```

## 📸 Sử dụng

### 1. Khởi động hệ thống

1. **Chạy Flask server** (terminal 1):

   ```bash
   cd server
   run_server.bat  # Windows
   # hoặc ./run_server.sh  # Linux/Mac
   ```

2. **Chạy React client** (terminal 2):

   ```bash
   npm run dev
   ```

3. **Truy cập ứng dụng**: `http://localhost:5173`

### 2. Upload ảnh từ máy tính

- Nhấn "Chọn ảnh" hoặc kéo thả ảnh vào vùng upload
- Hỗ trợ: JPG, PNG, GIF, BMP, WebP, HEIC
- Kích thước tối đa: 10MB (xử lý trên server)

### 3. Chụp ảnh từ webcam

- Nhấn "Chụp ảnh" để mở camera
- Đặt cây vào khung hình
- Nhấn "Chụp ảnh" để capture
- Xem preview và "Chụp lại" nếu cần

### 4. Nhận diện

- **Auto Recognition**: Tự động nhận diện sau khi upload
- **Manual**: Nhấn "Kiểm tra" để nhận diện thủ công
- **Server Processing**: Ảnh được gửi lên Flask server để xử lý
- Kết quả hiển thị: Trạng thái + độ tin cậy + xác suất chi tiết

## ⚙️ Cấu hình

### Server Configuration

Chỉnh sửa đường dẫn model trong `server/app.py`:

```python
# Đường dẫn đến file model .h5
model_path = "D:\web-kpdl\vegetable_classifier_resnet50_final (1).h5"
```

### Client Configuration

API endpoint trong `src/services/apiService.ts`:

```typescript
const API_BASE_URL = "http://localhost:5000";
```

### Model Requirements

Model .h5 cần có:

- Input shape: `(None, 224, 224, 3)` - Ảnh RGB 224x224
- Output shape: `(None, 2)` - 2 classes: [unhealthy, healthy]
- Hoặc output shape: `(None, 1)` - Single output: 0=unhealthy, 1=healthy

## 🔧 Development

### Scripts

**React Client:**

```bash
npm run dev          # Development server
npm run build        # Build production
npm run preview      # Preview build
npm run lint         # ESLint check
```

**Flask Server:**

```bash
cd server
python app.py        # Run server directly
# hoặc
run_server.bat       # Windows
./run_server.sh      # Linux/Mac
```

### Dependencies chính

**React Client:**

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Ant Design** - UI components
- **Vite** - Build tool
- **Tailwind CSS** - Styling

**Flask Server:**

- **Flask 2.3.3** - Web framework
- **TensorFlow 2.13.0** - AI inference
- **Pillow 10.0.1** - Image processing
- **NumPy 1.24.3** - Numerical computing

## 🐛 Troubleshooting

### Lỗi thường gặp

#### 1. "Server không khả dụng"

```bash
# Kiểm tra Flask server có chạy không
curl http://localhost:5000/health

# Khởi động lại server
cd server
python app.py
```

#### 2. "Model not loaded"

```bash
# Kiểm tra file model có tồn tại
ls "D:\web-kpdl\vegetable_classifier_resnet50_final (1).h5"

# Reload model qua API
curl -X POST http://localhost:5000/model/reload
```

#### 3. "Camera access denied"

- Cho phép quyền truy cập camera trong trình duyệt
- Sử dụng HTTPS (camera yêu cầu secure context)

#### 4. "Image too large"

- Nén ảnh trước khi upload
- Sử dụng ảnh < 10MB (server limit)

#### 5. "CORS error"

- Đảm bảo Flask server đang chạy
- Kiểm tra CORS configuration trong `app.py`

### Debug Mode

```typescript
// Bật console logs
console.log("Server status:", serverStatus);
console.log("Server connected:", serverConnected);
console.log("Prediction result:", result);
```

### Server Logs

```bash
# Xem server logs
cd server
python app.py
# Logs sẽ hiển thị trong console
```

## 📊 Performance

### Optimization

- **Server-side processing** - Model xử lý trên server, giảm tải client
- **Image compression** - Tự động nén ảnh trước khi gửi
- **Memory management** - Server tự động quản lý memory
- **Caching** - Server cache model trong memory
- **Batch processing** - Server có thể xử lý nhiều request song song

### Browser Support

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

### Server Performance

- **Model loading**: ~2-5 giây khi khởi động
- **Prediction time**: ~0.5-2 giây per ảnh
- **Memory usage**: ~500MB-1GB (tùy model size)
- **Concurrent requests**: Hỗ trợ multiple users

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 🚀 Deployment

### Quick Deploy (15 phút)

Xem [QUICK_DEPLOY.md](./QUICK_DEPLOY.md) để deploy nhanh với Vercel + Railway.

### Chi tiết Deploy

Xem [DEPLOYMENT.md](./DEPLOYMENT.md) để biết các phương án hosting khác nhau.

### Testing

Xem [TESTING.md](./TESTING.md) để test hệ thống sau khi deploy.

## 📞 Support

Nếu gặp vấn đề, vui lòng:

1. Kiểm tra [Issues](../../issues) trước
2. Tạo issue mới với mô tả chi tiết
3. Cung cấp thông tin: OS, Browser, Error logs

---

**Made with ❤️ for plant health detection**
