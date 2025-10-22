# ⚡ Quick Deploy Guide

Hướng dẫn deploy nhanh trong 15 phút với Vercel + Railway.

## 🎯 Mục tiêu

- React Client → Vercel (Free)
- Flask Server → Railway (Free tier)
- Tổng thời gian: ~15 phút

---

## 📋 Chuẩn bị

### 1. Tài khoản cần có:

- [x] GitHub account
- [x] Vercel account (đăng ký free)
- [x] Railway account (đăng ký free)

### 2. Code đã sẵn sàng:

- [x] React client build được
- [x] Flask server chạy được local
- [x] Model file có sẵn

---

## 🚀 Bước 1: Deploy React Client lên Vercel (5 phút)

### 1.1 Push code lên GitHub

```bash
# Nếu chưa có git repo
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### 1.2 Deploy lên Vercel

1. Truy cập [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import từ GitHub repository
4. Vercel tự động detect React
5. Click "Deploy"
6. Chờ 2-3 phút → Done! ✅

**Kết quả:** `https://your-app-name.vercel.app`

---

## 🚀 Bước 2: Deploy Flask Server lên Railway (10 phút)

### 2.1 Chuẩn bị server code

Tạo file `server/Procfile`:

```
web: python app.py
```

### 2.2 Deploy lên Railway

1. Truy cập [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Chọn repository và folder `server/`
4. Railway tự động detect Python
5. Click "Deploy"
6. Chờ 3-5 phút → Done! ✅

**Kết quả:** `https://your-app-name.railway.app`

### 2.3 Upload model file

1. Railway Dashboard → Deployments → Files
2. Upload `vegetable_classifier_resnet50_final (1).h5`
3. Đặt tên: `model.h5`

### 2.4 Cấu hình Environment Variables

Railway Dashboard → Variables:

```
PORT=5000
MODEL_PATH=/app/model.h5
```

---

## 🔧 Bước 3: Cấu hình CORS (2 phút)

### 3.1 Sửa server/app.py

```python
# Thay đổi đường dẫn model
model_path = os.path.join(os.getcwd(), "model.h5")

# Cấu hình CORS
CORS(app, origins=[
    "https://your-vercel-app.vercel.app",  # Thay bằng URL Vercel của bạn
    "http://localhost:5173"  # For development
])
```

### 3.2 Redeploy server

```bash
# Push changes
git add .
git commit -m "Update CORS and model path"
git push origin main

# Railway tự động redeploy
```

---

## 🧪 Bước 4: Test (3 phút)

### 4.1 Test API

```bash
# Test health endpoint
curl https://your-railway-app.railway.app/health

# Kết quả mong đợi:
# {"status":"healthy","model_loaded":true,"message":"Plant health detection API is running"}
```

### 4.2 Test Client

1. Truy cập `https://your-vercel-app.vercel.app`
2. Kiểm tra "Trạng thái Server" hiển thị màu xanh
3. Upload ảnh test
4. Kiểm tra kết quả prediction

---

## ✅ Kết quả

### URLs:

- **Client:** `https://your-vercel-app.vercel.app`
- **API:** `https://your-railway-app.railway.app`

### Features hoạt động:

- [x] Upload ảnh từ máy tính
- [x] Chụp ảnh từ webcam
- [x] AI prediction qua server
- [x] Hiển thị kết quả chi tiết
- [x] Responsive design

---

## 🔍 Troubleshooting

### Lỗi "Server không khả dụng"

1. Kiểm tra Railway app có chạy không
2. Kiểm tra logs trong Railway Dashboard
3. Kiểm tra model file đã upload chưa

### Lỗi CORS

1. Kiểm tra URL trong CORS config
2. Redeploy server sau khi sửa

### Lỗi Model không load

1. Kiểm tra file model đã upload đúng tên
2. Kiểm tra MODEL_PATH environment variable

---

## 📊 Monitoring

### Vercel:

- Dashboard → Functions → Logs
- Real-time analytics

### Railway:

- Dashboard → Deployments → Logs
- Resource usage

---

## 💰 Chi phí

### Vercel (Free tier):

- 100GB bandwidth/tháng
- Unlimited static sites
- ✅ Đủ cho hầu hết use cases

### Railway (Free tier):

- $5 credit/tháng
- 512MB RAM
- ✅ Đủ cho demo và small projects

### Tổng: **$0/tháng** cho personal use! 🎉

---

## 🚀 Next Steps

### Nếu cần scale up:

1. **Railway Pro** ($20/tháng) - More resources
2. **Vercel Pro** ($20/tháng) - More bandwidth
3. **Custom domain** - Professional look

### Nếu cần production:

1. **AWS** - Full control
2. **DigitalOcean** - VPS management
3. **Google Cloud** - Enterprise features

---

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra logs trong Vercel/Railway dashboard
2. Test API endpoints với curl
3. Kiểm tra browser console (F12)
4. Xem file DEPLOYMENT.md để biết thêm chi tiết

**Chúc bạn deploy thành công! 🎉**
