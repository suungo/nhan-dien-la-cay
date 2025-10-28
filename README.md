Hướng dẫn các cài đặt và chạy đồ án nhận diện lá healthy và unHealthy

## File tranin

Để tranin với mô hình đã chọn là resnet50 và mobilenetv2 (khuyến khích tranin bằng google colap)
Ở file DA-KPDL.ipynb chứa toàn bộ source code của quá trinh tranin được chia thành các select cụ thể để dễ phân biệt
Lưu ý trước khi tranin bạn cần truy cập và folder data trong đó đường dẫn google driver trỏ để tập dữ liệu sao đó bạn càn download về và sửa lại đường dẫn để không bị lỗi .
Ở các phần sau sẽ là phần web để hiển thị bạn cần làm theo từng bước theo chỉ dẫn để có thể chạy thành công dự án này

lưu ý : nếu trong file .h5 trên web đang lấy kêt quả tốt nhánh của mô hình nếu như bạn có thể tinh chỉnh sao cho độ chính xác cao hơn thì trong folder server ở file app.py hãy thay đổi lại đường dẫn .h5 chứa kq tốt nhất của bạn . Chúc bạn thành công !

## Server (Python / Flask)

1. Tạo virtualenv và cài dependencies (Windows - cmd.exe):

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r server\requirements.txt
```

2. Chạy server (Windows):

```cmd
cd server
venv\Scripts\activate
python app.py
# hoặc dùng run_server.bat nếu có
```

3. Health check:

```cmd
curl http://localhost:5000/health
```

4. Gửi ảnh để dự đoán (predict) và nhận GradCAM + contour:

```cmd
curl -X POST -F "file=@C:\path\to\leaf.jpg" http://localhost:5000/predict
```

Ví dụ response (JSON):

```json
{
  "success": true,
  "result": {
    "status": "unhealthy",
    "confidence": 0.99,
    "healthy_probability": 0.01,
    "unhealthy_probability": 0.99,
    "gradcam_image": "data:image/jpeg;base64,...",
    "contour_image": "data:image/jpeg;base64,..."
  }
}
```

5. Reload model (sau khi cập nhật file .h5):

```cmd
curl -X POST http://localhost:5000/model/reload
```

6. Lấy thông tin model (layer, shapes) — dùng để debug GradCAM:

```cmd
curl http://localhost:5000/model/info
```

7. Lưu GradCAM base64 về file (PowerShell example):

```powershell
$json = Invoke-RestMethod -Method Post -Form @{ file = Get-Item 'C:\path\to\leaf.jpg' } -Uri 'http://localhost:5000/predict'
$b64 = $json.result.gradcam_image -replace '^data:image\/jpeg;base64,',''
[System.IO.File]::WriteAllBytes('gradcam.jpg', [Convert]::FromBase64String($b64))
```

---

## Client (React + Vite)

1. Cài dependencies và chạy dev server (trong `view-kp`):

```cmd
cd view-kp
npm install
npm run dev
```

2. Build production và preview:

```cmd
npm run build
npm run preview
```
