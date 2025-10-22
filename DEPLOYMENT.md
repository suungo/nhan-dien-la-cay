# 🚀 Hướng dẫn Deploy Plant Health Detection System

Hướng dẫn deploy hệ thống nhận diện sức khỏe cây trồng lên các nền tảng hosting phổ biến.

## 📋 Tổng quan

Hệ thống gồm 2 phần:

- **React Client**: Frontend web app
- **Flask Server**: Backend API với AI model

## 🎯 Các phương án hosting

### 1. **Vercel + Railway** (Khuyến nghị - Dễ nhất)

- React Client → Vercel
- Flask Server → Railway

### 2. **Netlify + Heroku**

- React Client → Netlify
- Flask Server → Heroku

### 3. **AWS**

- React Client → S3 + CloudFront
- Flask Server → EC2 hoặc Lambda

### 4. **DigitalOcean**

- Cả 2 → Droplet

### 5. **VPS tự quản lý**

- Cả 2 → VPS (Ubuntu/CentOS)

---

## 🟢 Phương án 1: Vercel + Railway (Khuyến nghị)

### A. Deploy React Client lên Vercel

#### Bước 1: Chuẩn bị code

```bash
# Build production
npm run build

# Kiểm tra thư mục dist/ được tạo
ls dist/
```

#### Bước 2: Tạo file cấu hình Vercel

Tạo file `vercel.json`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

#### Bước 3: Deploy lên Vercel

1. Truy cập [vercel.com](https://vercel.com)
2. Đăng nhập với GitHub
3. Import project từ GitHub
4. Vercel tự động detect React và build
5. Deploy thành công!

#### Bước 4: Cấu hình Environment Variables

Trong Vercel Dashboard:

- Settings → Environment Variables
- Thêm: `VITE_API_URL=https://your-railway-app.railway.app`

### B. Deploy Flask Server lên Railway

#### Bước 1: Chuẩn bị server code

Tạo file `Procfile`:

```
web: python app.py
```

Tạo file `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### Bước 2: Tạo Railway project

1. Truy cập [railway.app](https://railway.app)
2. Đăng nhập với GitHub
3. New Project → Deploy from GitHub repo
4. Chọn repository và folder `server/`

#### Bước 3: Cấu hình Environment Variables

Trong Railway Dashboard:

- Variables tab
- Thêm: `PORT=5000`
- Thêm: `MODEL_PATH=/app/model.h5`

#### Bước 4: Upload model file

1. Railway → Deployments → Files
2. Upload file `vegetable_classifier_resnet50_final (1).h5`
3. Đặt tên: `model.h5`

#### Bước 5: Cập nhật code để sử dụng model từ Railway

Sửa `server/app.py`:

```python
# Thay đổi đường dẫn model
model_path = os.path.join(os.getcwd(), "model.h5")
```

### C. Cấu hình CORS

Sửa `server/app.py`:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "https://your-vercel-app.vercel.app",
    "http://localhost:5173"  # For development
])
```

---

## 🟡 Phương án 2: Netlify + Heroku

### A. Deploy React Client lên Netlify

#### Bước 1: Build và deploy

```bash
# Build production
npm run build

# Deploy lên Netlify
npx netlify deploy --prod --dir=dist
```

#### Bước 2: Cấu hình redirects

Tạo file `public/_redirects`:

```
/*    /index.html   200
```

### B. Deploy Flask Server lên Heroku

#### Bước 1: Chuẩn bị Heroku

```bash
# Install Heroku CLI
# Tạo Heroku app
heroku create your-app-name

# Login
heroku login
```

#### Bước 2: Tạo file cấu hình

Tạo file `server/Procfile`:

```
web: gunicorn app:app
```

Tạo file `server/runtime.txt`:

```
python-3.9.18
```

#### Bước 3: Deploy

```bash
cd server
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

---

## 🔵 Phương án 3: AWS

### A. Deploy React Client lên S3 + CloudFront

#### Bước 1: Tạo S3 bucket

```bash
# Install AWS CLI
aws configure

# Tạo bucket
aws s3 mb s3://your-bucket-name

# Upload files
aws s3 sync dist/ s3://your-bucket-name --delete
```

#### Bước 2: Cấu hình CloudFront

1. AWS Console → CloudFront
2. Create Distribution
3. Origin: S3 bucket
4. Default Root Object: index.html
5. Error Pages: 404 → /index.html (200)

### B. Deploy Flask Server lên EC2

#### Bước 1: Tạo EC2 instance

- Instance type: t3.medium (2 vCPU, 4GB RAM)
- OS: Ubuntu 20.04 LTS
- Security Group: Port 22 (SSH), Port 80 (HTTP), Port 5000 (Flask)

#### Bước 2: Cài đặt dependencies

```bash
# SSH vào EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, pip, nginx
sudo apt install python3 python3-pip nginx -y

# Install Python dependencies
pip3 install -r requirements.txt
```

#### Bước 3: Cấu hình Nginx

Tạo file `/etc/nginx/sites-available/plant-health`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Bước 4: Chạy ứng dụng

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/plant-health /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Chạy Flask app
cd /path/to/your/app
python3 app.py
```

---

## 🟠 Phương án 4: DigitalOcean Droplet

### Tạo Droplet và cài đặt

#### Bước 1: Tạo Droplet

- Image: Ubuntu 20.04 LTS
- Size: 2GB RAM, 1 vCPU (tối thiểu)
- Region: Gần nhất với users

#### Bước 2: Cài đặt system

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python
sudo apt install python3 python3-pip python3-venv nginx -y

# Install PM2 for process management
sudo npm install -g pm2
```

#### Bước 3: Deploy code

```bash
# Clone repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Setup React client
npm install
npm run build

# Setup Flask server
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Bước 4: Cấu hình Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve React app
    location / {
        root /path/to/your/repo/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Bước 5: Chạy với PM2

```bash
# Start Flask server
cd server
pm2 start app.py --name "plant-health-api"

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Save PM2 configuration
pm2 save
pm2 startup
```

---

## 🔧 Cấu hình chung

### Environment Variables

#### React Client (.env.production):

```env
VITE_API_URL=https://your-api-domain.com
```

#### Flask Server:

```env
FLASK_ENV=production
MODEL_PATH=/path/to/model.h5
CORS_ORIGINS=https://your-client-domain.com
```

### SSL/HTTPS

#### Vercel + Railway:

- Tự động có SSL
- Không cần cấu hình thêm

#### Netlify + Heroku:

- Netlify: Tự động SSL
- Heroku: Cần add-on SSL hoặc Cloudflare

#### AWS:

- CloudFront: Tự động SSL
- EC2: Cần Certificate Manager + Load Balancer

#### DigitalOcean:

- Cần cấu hình Let's Encrypt:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Monitoring và Logs

#### Vercel:

- Dashboard → Functions → Logs

#### Railway:

- Dashboard → Deployments → Logs

#### Heroku:

```bash
heroku logs --tail
```

#### AWS:

- CloudWatch Logs

#### DigitalOcean:

```bash
pm2 logs
pm2 monit
```

---

## 📊 So sánh các phương án

| Phương án        | Chi phí      | Độ khó   | Performance | Scalability |
| ---------------- | ------------ | -------- | ----------- | ----------- |
| Vercel + Railway | $0-20/tháng  | ⭐⭐     | ⭐⭐⭐⭐    | ⭐⭐⭐⭐    |
| Netlify + Heroku | $0-25/tháng  | ⭐⭐⭐   | ⭐⭐⭐      | ⭐⭐⭐      |
| AWS              | $10-50/tháng | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐  |
| DigitalOcean     | $5-20/tháng  | ⭐⭐⭐⭐ | ⭐⭐⭐⭐    | ⭐⭐⭐      |

---

## 🚀 Quick Start (Vercel + Railway)

### 1. Deploy React Client (5 phút)

```bash
# Build
npm run build

# Deploy to Vercel
npx vercel --prod
```

### 2. Deploy Flask Server (10 phút)

```bash
# Push to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# Deploy to Railway
# - Connect GitHub repo
# - Select server/ folder
# - Add environment variables
# - Upload model file
```

### 3. Cấu hình CORS

Sửa `server/app.py`:

```python
CORS(app, origins=["https://your-vercel-app.vercel.app"])
```

### 4. Test

- Client: `https://your-vercel-app.vercel.app`
- API: `https://your-railway-app.railway.app/health`

---

## 🔍 Troubleshooting

### Lỗi CORS

```python
# Thêm domain client vào CORS
CORS(app, origins=["https://your-client-domain.com"])
```

### Lỗi Model không load

```python
# Kiểm tra đường dẫn model
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")
```

### Lỗi Memory

- Tăng RAM cho server
- Optimize model size
- Sử dụng model quantization

### Lỗi Timeout

- Tăng timeout cho API calls
- Optimize model inference
- Sử dụng caching

---

## 📝 Checklist Deploy

### Trước khi deploy:

- [ ] Code đã test local
- [ ] Environment variables đã cấu hình
- [ ] Model file đã upload
- [ ] CORS đã cấu hình đúng
- [ ] SSL/HTTPS đã setup

### Sau khi deploy:

- [ ] Test API endpoints
- [ ] Test client kết nối server
- [ ] Test upload và prediction
- [ ] Kiểm tra logs
- [ ] Test performance

---

**Chúc bạn deploy thành công! 🎉**
