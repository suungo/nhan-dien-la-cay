# 🌐 Deploy React Frontend lên Vercel

## ✅ Checklist

- [x] React app đã chạy được local
- [x] Backend đã deploy lên Railway
- [x] Có URL backend: `https://your-app.railway.app`

---

## 📝 Bước 1: Cấu hình Environment Variables

Tạo file `.env.production`:

```env
VITE_API_URL=https://your-app.railway.app
```

---

## 🔧 Bước 2: Test Build Local

```bash
cd D:\web-kpdl\view-kp

# Build production
npm run build

# Test build
npm run preview
```

Mở: `http://localhost:4173` và test xem có hoạt động không.

---

## 🚀 Bước 3: Deploy lên Vercel

### Option 1: Vercel CLI (Nhanh nhất)

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

### Option 2: Vercel Dashboard

1. Truy cập: https://vercel.com
2. Đăng nhập bằng GitHub
3. Click **"New Project"**
4. Import repository: `plant-health-detection`
5. **Root Directory:** `view-kp`
6. **Framework Preset:** Vite
7. **Build Command:** `npm run build`
8. **Output Directory:** `dist`

---

## ⚙️ Bước 4: Cấu hình Environment Variables trên Vercel

Trong Vercel Dashboard → **Settings → Environment Variables**:

```
Name: VITE_API_URL
Value: https://your-app.railway.app
Environment: Production, Preview, Development
```

---

## 🔄 Bước 5: Redeploy

Sau khi thêm environment variables:

1. Vercel Dashboard → **Deployments**
2. Click **"Redeploy"** để apply env vars

---

## ✅ Hoàn tất!

Frontend đã live tại: `https://your-app.vercel.app`

---

## 🎯 Next Steps

### 1. Custom Domain (Optional)

Vercel Dashboard → **Settings → Domains**:

- Thêm domain: `plant-health.yourdomain.com`
- Follow DNS setup instructions

### 2. Cập nhật CORS trên Backend

Sửa `server/app.py`:

```python
CORS(app, origins=[
    "https://your-app.vercel.app",
    "http://localhost:5174"
])
```

Commit và push để Railway auto-deploy.

### 3. Performance Optimization

- ✅ Enable Vercel Analytics
- ✅ Configure caching headers
- ✅ Optimize images

---

## 📊 Monitoring

Vercel cung cấp:

- ✅ Real-time Analytics
- ✅ Web Vitals
- ✅ Deployment logs
- ✅ Error tracking

---

## 💰 Chi phí

**Vercel Free Tier:**

- ✅ Unlimited deployments
- ✅ 100GB bandwidth/month
- ✅ Automatic HTTPS
- ✅ Global CDN

**Pro ($20/month):**

- Advanced analytics
- More bandwidth
- Serverless function hours

---

## 🐛 Troubleshooting

### Lỗi: Cannot connect to backend

```typescript
// Check API URL
console.log("API URL:", import.meta.env.VITE_API_URL);

// Test backend directly
fetch("https://your-app.railway.app/health")
  .then((res) => res.json())
  .then(console.log);
```

### Lỗi: CORS

```python
# Backend (app.py)
CORS(app, origins=["https://your-app.vercel.app"])
```

### Lỗi: 404 on refresh

Vercel tự động xử lý SPA routing. Nếu vẫn lỗi, tạo `vercel.json`:

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

---

## ✅ Testing Checklist

- [ ] Health check hoạt động
- [ ] Upload ảnh hoạt động
- [ ] Prediction trả về kết quả
- [ ] GradCAM + Contour hiển thị
- [ ] Pie chart hiển thị chính xác
- [ ] Responsive trên mobile

---

**Live URLs:**

- Frontend: `https://your-app.vercel.app`
- Backend: `https://your-app.railway.app`

🎉 **Hoàn tất deployment!**
