import { Button, Form, Image, message, Modal, Spin, Tag } from "antd";
import {
  Camera,
  Download,
  Images,
  RefreshCw,
  Server,
  Video,
} from "lucide-react";
import React, {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { apiService } from "./services/apiService";

// Define InferenceResult type locally (compatible with server response)
type InferenceResult = {
  status: "healthy" | "unhealthy" | "unreliable";
  confidence?: number;
  healthy_probability?: number;
  unhealthy_probability?: number;
  timestamp?: number;
  gradcam_image?: string;
  contour_image?: string;
};

// Danh sách MIME types ảnh chấp nhận
const ACCEPTED_IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/bmp",
  "image/tif",
  "image/tiff",
  "image/webp",
  "image/heic",
  "image/heif",
  "image/svg+xml",
] as const;

// Đuôi file hợp lệ
const ACCEPTED_IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".bmp",
  ".tif",
  ".tiff",
  ".webp",
  ".heic",
  ".heif",
  ".svg",
] as const;

// Hàm kiểm tra file ảnh có hợp lệ không
const isValidImageFile = (file: File): boolean => {
  const typeValid = ACCEPTED_IMAGE_MIME_TYPES.includes(
    file.type.toLowerCase() as (typeof ACCEPTED_IMAGE_MIME_TYPES)[number]
  );
  const extValid = ACCEPTED_IMAGE_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );
  return typeValid || extValid;
};

function App() {
  const [logo, setLogo] = useState<string | File | null>(null);
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [isWebcamOpen, setIsWebcamOpen] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null); // Ảnh đã chụp từ webcam
  const [isVideoReady, setIsVideoReady] = useState(false); // Trạng thái video sẵn sàng
  const [serverConnected, setServerConnected] = useState(false); // Trạng thái kết nối server
  const [serverStatus, setServerStatus] = useState<string>(""); // Thông tin server
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Kiểm tra kết nối server khi component mount
  const checkServerConnection = useCallback(async () => {
    try {
      console.log("🔍 Checking server connection...");
      const healthData = await apiService.healthCheck();
      setServerConnected(true);
      setServerStatus(
        `Server: ${healthData.message} | Model: ${
          healthData.model_loaded ? "Loaded" : "Not loaded"
        }`
      );
      console.log("✅ Server connection successful:", healthData);
    } catch (error) {
      console.error("❌ Server connection failed:", error);
      setServerConnected(false);
      setServerStatus("Server không khả dụng");
    }
  }, []);

  // Effect để kiểm tra kết nối server khi component mount
  useEffect(() => {
    checkServerConnection();
  }, [checkServerConnection]);

  // Xử lý sự kiện kéo thả file vào vùng drop
  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  // Xử lý sự kiện kéo thả file ra khỏi vùng drop
  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  // Xử lý upload ảnh khi người dùng chọn file từ máy tính
  const handleFile = (file: File) => {
    const maxSize = 2 * 1024 * 1024; // 2MB

    if (!isValidImageFile(file)) {
      return message.error("Vui lòng đăng tải ảnh với định dạng thích hợp");
    }
    if (file.size > maxSize) {
      return message.error("Vui lòng nén ảnh hoặc chọn ảnh khác");
    }

    setLogo(file); // Đây là file dùng để gửi qua FormData
    setLogoPreview(URL.createObjectURL(file)); // Preview ảnh
    message.success("Đăng ảnh thành công - Nhấn 'Kiểm tra' để nhận diện");
  };

  // Xử lý sự kiện khi người dùng chọn file từ máy tính
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  };

  // Xử lý sự kiện kéo thả file vào vùng drop
  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];

    if (isValidImageFile(file)) {
      handleFile(file);
    } else {
      return message.error("Vui lòng đăng tải ảnh với định dạng thích hợp");
    }
  }, []);

  // Hàm gửi ảnh lên Flask server để xử lý
  const predictWithServer = async (file: File): Promise<InferenceResult> => {
    try {
      console.log("🚀 Sending image to Flask server for prediction...");

      // Kiểm tra kết nối server trước
      if (!serverConnected) {
        throw new Error(
          "Không thể kết nối đến server. Vui lòng kiểm tra server Flask."
        );
      }

      // Gửi ảnh lên server
      const serverResult = await apiService.predictPlantHealth(file);

      console.log("✅ Server prediction successful:", serverResult);

      // Convert server result sang format InferenceResult
      const result: InferenceResult = {
        status: serverResult.status,
        confidence: serverResult.confidence,
        healthy_probability: serverResult.healthy_probability,
        unhealthy_probability: serverResult.unhealthy_probability,
        timestamp: serverResult.timestamp,
        gradcam_image: serverResult.gradcam_image, // ✅ THÊM GRADCAM
        contour_image: serverResult.contour_image, // ✅ THÊM CONTOUR
      };

      console.log("🔍 Converted result:", result);
      console.log("🖼️ Has GradCAM after convert:", !!result.gradcam_image);
      console.log("🎯 Has Contour after convert:", !!result.contour_image);

      return result;
    } catch (error) {
      console.error("❌ Server prediction failed:", error);
      throw error;
    }
  };

  // Xử lý sự kiện người dùng nhấn kiểm tra
  const handleClick = async () => {
    if (!logo) {
      return message.warning("Vui lòng chọn ảnh trước khi kiểm tra");
    }
    if (typeof logo === "string") {
      return message.error("Ảnh không hợp lệ, vui lòng chọn lại");
    }

    console.log("🚀 Bắt đầu kiểm tra ảnh:", logo.name, logo.size);
    setIsLoading(true);
    setResult(null);

    try {
      // Kiểm tra kết nối server trước
      if (!serverConnected) {
        throw new Error(
          "Không thể kết nối đến server Flask. Vui lòng kiểm tra server có đang chạy không."
        );
      }

      console.log("🔄 Sử dụng Flask server để xử lý ảnh...");

      // Gửi ảnh lên server để xử lý
      const result = await predictWithServer(logo);
      console.log("✅ Server prediction thành công:", result);
      console.log(
        "🖼️ GradCAM image:",
        result.gradcam_image ? "CÓ" : "KHÔNG CÓ"
      );
      console.log(
        "🎯 Contour image:",
        result.contour_image ? "CÓ" : "KHÔNG CÓ"
      );

      setResult(result);
      console.log("📊 State updated, result set to:", result);
      message.success("Đã kiểm tra thành công");
    } catch (error) {
      console.error("❌ Lỗi khi kiểm tra:", error);
      const msg = error instanceof Error ? error.message : "Lỗi không xác định";
      message.error(`Lỗi: ${msg}`);

      // Hiển thị thông tin debug
      console.log("🔍 Debug info:");
      console.log("- File name:", logo.name);
      console.log("- File size:", logo.size);
      console.log("- File type:", logo.type);
      console.log("- Server connected:", serverConnected);
      console.log("- Server status:", serverStatus);
    } finally {
      console.log("🏁 Hoàn thành kiểm tra");
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setLogo(null);
    if (logoPreview) URL.revokeObjectURL(logoPreview);
    setLogoPreview(null);
    setResult(null); // Xóa luôn kết quả dự đoán
    message.info("Đã xoá ảnh và kết quả");
  };

  // Hàm mở webcam
  const openWebcam = async () => {
    try {
      console.log("🎥 Đang mở webcam...");
      setIsVideoReady(false); // Reset trạng thái

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920, min: 1280 },
          height: { ideal: 1080, min: 720 },
          facingMode: "environment", // Camera sau chất lượng cao hơn
          frameRate: { ideal: 30, max: 60 },
        },
      });
      console.log("✅ Webcam stream:", mediaStream);
      setStream(mediaStream);
      setIsWebcamOpen(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        console.log("📹 Video element srcObject set");

        // Đợi video metadata load
        videoRef.current.onloadedmetadata = () => {
          console.log("📹 Video metadata loaded:", {
            videoWidth: videoRef.current?.videoWidth,
            videoHeight: videoRef.current?.videoHeight,
            readyState: videoRef.current?.readyState,
          });

          // Kiểm tra xem video có dimensions hợp lệ không
          if (videoRef.current && videoRef.current.videoWidth > 0) {
            setIsVideoReady(true);
            console.log("✅ Video sẵn sàng chụp!");
          }
        };

        videoRef.current.oncanplay = () => {
          console.log("📹 Video can play:", {
            videoWidth: videoRef.current?.videoWidth,
            videoHeight: videoRef.current?.videoHeight,
            readyState: videoRef.current?.readyState,
          });

          // Double-check video ready
          if (videoRef.current && videoRef.current.videoWidth > 0) {
            setIsVideoReady(true);
            console.log("✅ Video đã sẵn sàng (oncanplay)");
          }
        };

        // Event khi video bắt đầu play
        videoRef.current.onplay = () => {
          console.log("📹 Video started playing");
          // Đợi thêm một chút để đảm bảo dimensions đã load
          setTimeout(() => {
            if (videoRef.current && videoRef.current.videoWidth > 0) {
              setIsVideoReady(true);
              console.log("✅ Video dimensions ready");
            }
          }, 500);
        };

        // Thử play video để kích hoạt
        videoRef.current.play().catch((error) => {
          console.error("❌ Lỗi play video:", error);
        });
      }
      message.success("Webcam đã được bật. Đang tải camera...");
    } catch (error) {
      message.error(
        "Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập."
      );
      console.error("❌ Error accessing webcam:", error);
    }
  };

  // Hàm đóng webcam
  const closeWebcam = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setIsWebcamOpen(false);
    setIsVideoReady(false); // Reset trạng thái video
    setCapturedImage(null); // Xóa ảnh đã chụp
    message.info("Webcam đã được tắt");
  };

  // Hàm chụp ảnh từ webcam
  const capturePhoto = async () => {
    if (!videoRef.current || !canvasRef.current) {
      console.error("❌ Video hoặc Canvas không tồn tại");
      message.error("Lỗi: Không tìm thấy video hoặc canvas");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    console.log("📸 Bắt đầu chụp ảnh...");
    console.log("📹 Video info:", {
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      readyState: video.readyState,
      currentTime: video.currentTime,
      paused: video.paused,
    });

    // Kiểm tra video đã sẵn sàng chưa
    if (video.readyState < 2) {
      console.error("❌ Video chưa sẵn sàng (readyState < 2)");
      message.error("Video chưa sẵn sàng, vui lòng đợi vài giây");
      return;
    }

    // Kiểm tra video có đang play không
    if (video.paused || video.ended) {
      console.error("❌ Video không đang phát");
      try {
        await video.play();
        console.log("✅ Đã khởi động lại video");
      } catch (error) {
        console.error("❌ Không thể khởi động video:", error);
        message.error("Không thể chụp ảnh, vui lòng thử lại");
        return;
      }
    }

    // Kiểm tra video dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.error("❌ Video dimensions = 0");
      message.error("Camera chưa sẵn sàng, vui lòng đợi và thử lại");
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      console.error("❌ Không thể lấy canvas context");
      message.error("Lỗi canvas context");
      return;
    }

    // Set canvas size theo video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    console.log("📐 Canvas size:", {
      width: canvas.width,
      height: canvas.height,
    });

    // Draw video frame to canvas
    try {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      console.log("✅ Đã vẽ video lên canvas");
    } catch (error) {
      console.error("❌ Lỗi vẽ video lên canvas:", error);
      message.error("Lỗi chụp ảnh, vui lòng thử lại");
      return;
    }

    // Convert to data URL
    try {
      const imageDataURL = canvas.toDataURL("image/jpeg", 0.9);

      // Kiểm tra data URL hợp lệ
      if (
        !imageDataURL ||
        imageDataURL === "data:," ||
        imageDataURL.length < 100
      ) {
        console.error(
          "❌ Data URL không hợp lệ:",
          imageDataURL.substring(0, 100)
        );
        message.error("Lỗi tạo ảnh, vui lòng thử lại");
        return;
      }

      setCapturedImage(imageDataURL);

      console.log("✅ Ảnh đã chụp thành công!");
      console.log("📸 Data URL length:", imageDataURL.length);
      console.log(
        "📸 Data URL preview:",
        imageDataURL.substring(0, 100) + "..."
      );

      message.success("Đã chụp ảnh thành công! Nhấn 'Xác nhận' để sử dụng.");
    } catch (error) {
      console.error("❌ Lỗi tạo data URL:", error);
      message.error("Lỗi tạo ảnh, vui lòng thử lại");
      return;
    }
  };

  return (
    <>
      <div className="min-h-screen bg-slate-50 p-4">
        <div className="w-full max-w-7xl mx-auto">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 md:p-8">
            <div className="text-center mb-6">
              <h1 className="text-2xl md:text-3xl font-semibold text-slate-800">
                Hệ thống nhận diện sức khỏe cây trồng
              </h1>
              <p className="text-slate-500 mt-1">
                Sử dụng AI để kiểm tra sức khỏe cây trồng qua hình ảnh
              </p>
            </div>

            {/* Chỉ giữ lại upload ảnh, bỏ realtime */}
            <div className="space-y-6">
              {/* Server Status */}
              <div
                className={`p-4 rounded-lg ${
                  serverConnected ? "bg-green-50" : "bg-red-50"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4
                    className={`font-medium ${
                      serverConnected ? "text-green-800" : "text-red-800"
                    }`}
                  >
                    Trạng thái Server
                  </h4>
                  <Button
                    size="small"
                    icon={<RefreshCw size={14} />}
                    onClick={checkServerConnection}
                    loading={isLoading}
                  >
                    Kiểm tra lại
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <Server
                    size={16}
                    className={
                      serverConnected ? "text-green-600" : "text-red-600"
                    }
                  />
                  <span
                    className={`text-sm ${
                      serverConnected ? "text-green-700" : "text-red-700"
                    }`}
                  >
                    {serverStatus}
                  </span>
                </div>
                {!serverConnected && (
                  <p className="text-xs text-red-600 mt-2">
                    ⚠️ Vui lòng khởi động Flask server trước khi sử dụng
                  </p>
                )}
              </div>

              {/* AI Processing Info */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-800 mb-2">
                  Xử lý AI trên Server
                </h4>
                <div className="p-3 bg-white rounded-lg">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🤖</span>
                    <div>
                      <h5 className="font-medium text-blue-800">
                        Flask Server + TensorFlow
                      </h5>
                      <p className="text-sm text-blue-600">
                        Model .h5 được xử lý trên server để tăng hiệu suất
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-blue-600 mt-2">
                  Ảnh sẽ được gửi lên server để AI phân tích và trả về kết quả
                </p>
              </div>

              <Form.Item name="formImages" className="mb-6">
                <div
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={() => inputRef.current?.click()}
                  className="relative rounded-xl border-2 border-dashed border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-100/50 transition-colors cursor-pointer min-h-[200px] flex items-center justify-center"
                >
                  <div className="flex flex-col items-center justify-center text-center gap-4 px-4 sm:px-6 py-8 sm:py-10 w-full">
                    {logoPreview ? (
                      <div className="w-full max-w-md mx-auto">
                        <Image
                          preview={false}
                          loading="lazy"
                          className="w-full h-auto max-h-80 sm:max-h-96 object-contain rounded-lg shadow-sm border border-gray-200"
                          src={logoPreview}
                          alt="Ảnh cây cần kiểm tra"
                          style={{
                            width: "100%",
                            height: "auto",
                            maxHeight: "320px",
                            objectFit: "contain",
                          }}
                        />
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-2 text-slate-500 w-full">
                        <Images
                          size={32}
                          className="text-slate-400 sm:w-9 sm:h-9"
                        />
                        <p className="text-sm sm:text-base px-2">
                          Kéo thả ảnh vào đây hoặc bấm để chọn
                        </p>
                        <p className="text-xs text-slate-400 px-2">
                          Hỗ trợ JPG, PNG, GIF... tối đa 2MB
                        </p>
                      </div>
                    )}
                    <input
                      ref={inputRef}
                      onChange={handleFileChange}
                      type="file"
                      accept="image/*"
                      style={{ display: "none" }}
                    />
                  </div>
                </div>
              </Form.Item>

              {isLoading && (
                <div className="flex justify-center mb-4">
                  <Spin />
                </div>
              )}

              {result && (
                <div className="flex flex-col items-center mb-4 space-y-2 max-w-full overflow-hidden px-2">
                  <div className="flex justify-center w-full">
                    {result.status === "healthy" ? (
                      <Tag
                        color="green"
                        className="text-sm sm:text-base px-3 py-1 text-center"
                      >
                        🌱 Cây khỏe mạnh
                        {result.confidence
                          ? ` (${((result.confidence || 0) * 100).toFixed(4)}%)`
                          : ""}
                      </Tag>
                    ) : result.status === "unreliable" ? (
                      <Tag
                        color="orange"
                        className="text-sm sm:text-base px-3 py-1 text-center"
                      >
                        ⚠️ Model không tin cậy
                      </Tag>
                    ) : (
                      <Tag
                        color="red"
                        className="text-sm sm:text-base px-3 py-1 text-center"
                      >
                        🍂 Cây có dấu hiệu bệnh
                        {result.confidence
                          ? ` (${((result.confidence || 0) * 100).toFixed(4)}%)`
                          : ""}
                      </Tag>
                    )}
                  </div>

                  {/* Thông tin chi tiết về độ tin cậy */}
                  <div className="text-center">
                    <div
                      className={`text-sm px-3 py-1 rounded-full ${
                        (result.confidence || 0) > 0.8
                          ? "bg-green-100 text-green-800"
                          : (result.confidence || 0) > 0.6
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {result.confidence && result.confidence > 0.8
                        ? "✅ Độ tin cậy cao"
                        : result.confidence && result.confidence > 0.6
                        ? "⚠️ Độ tin cậy trung bình"
                        : "❌ Độ tin cậy thấp - Cần kiểm tra lại"}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Xử lý bởi: Flask Server + TensorFlow
                    </p>
                    {result.healthy_probability !== undefined &&
                      result.unhealthy_probability !== undefined && (
                        <div className="mt-4 w-full max-w-sm mx-auto px-2">
                          {/* Biểu đồ tròn (Pie Chart) */}
                          <div className="flex flex-col items-center">
                            <div className="relative w-48 h-48">
                              <svg
                                viewBox="0 0 100 100"
                                className="transform -rotate-90"
                              >
                                {/* Background circle */}
                                <circle
                                  cx="50"
                                  cy="50"
                                  r="40"
                                  fill="none"
                                  stroke="#e5e7eb"
                                  strokeWidth="20"
                                />

                                {/* Healthy segment (green) */}
                                <circle
                                  cx="50"
                                  cy="50"
                                  r="40"
                                  fill="none"
                                  stroke="url(#greenGradient)"
                                  strokeWidth="20"
                                  strokeDasharray={`${
                                    result.healthy_probability * 251.2
                                  } 251.2`}
                                  className="transition-all duration-700"
                                  strokeLinecap="round"
                                />

                                {/* Unhealthy segment (red) */}
                                <circle
                                  cx="50"
                                  cy="50"
                                  r="40"
                                  fill="none"
                                  stroke="url(#redGradient)"
                                  strokeWidth="20"
                                  strokeDasharray={`${
                                    result.unhealthy_probability * 251.2
                                  } 251.2`}
                                  strokeDashoffset={`-${
                                    result.healthy_probability * 251.2
                                  }`}
                                  className="transition-all duration-700"
                                  strokeLinecap="round"
                                />

                                {/* Gradients */}
                                <defs>
                                  <linearGradient
                                    id="greenGradient"
                                    x1="0%"
                                    y1="0%"
                                    x2="100%"
                                    y2="100%"
                                  >
                                    <stop offset="0%" stopColor="#4ade80" />
                                    <stop offset="100%" stopColor="#16a34a" />
                                  </linearGradient>
                                  <linearGradient
                                    id="redGradient"
                                    x1="0%"
                                    y1="0%"
                                    x2="100%"
                                    y2="100%"
                                  >
                                    <stop offset="0%" stopColor="#f87171" />
                                    <stop offset="100%" stopColor="#dc2626" />
                                  </linearGradient>
                                </defs>
                              </svg>

                              {/* Center text */}
                              <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-center">
                                  <div className="text-2xl font-bold text-gray-800">
                                    {((result.confidence || 0) * 100).toFixed(
                                      4
                                    )}
                                    %
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    Độ tin cậy
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Legend */}
                            <div className="mt-4 space-y-2 w-full">
                              <div className="flex items-center justify-between p-2 bg-green-50 rounded-lg">
                                <div className="flex items-center gap-2">
                                  <div className="w-4 h-4 rounded-full bg-gradient-to-br from-green-400 to-green-600"></div>
                                  <span className="text-sm font-medium text-green-800">
                                    🌱 Khỏe mạnh
                                  </span>
                                </div>
                                <span className="text-sm font-bold text-green-700">
                                  {(result.healthy_probability * 100).toFixed(
                                    4
                                  )}
                                  %
                                </span>
                              </div>

                              <div className="flex items-center justify-between p-2 bg-red-50 rounded-lg">
                                <div className="flex items-center gap-2">
                                  <div className="w-4 h-4 rounded-full bg-gradient-to-br from-red-400 to-red-600"></div>
                                  <span className="text-sm font-medium text-red-800">
                                    🍂 Có bệnh
                                  </span>
                                </div>
                                <span className="text-sm font-bold text-red-700">
                                  {(result.unhealthy_probability * 100).toFixed(
                                    4
                                  )}
                                  %
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                    {/* GradCAM + Contour Visualization */}
                    {(() => {
                      console.log(
                        "🎨 Rendering image section, result:",
                        result
                      );
                      console.log("🎨 Has GradCAM:", !!result?.gradcam_image);
                      console.log("🎨 Has Contour:", !!result?.contour_image);
                      return null;
                    })()}
                    {(result?.gradcam_image || result?.contour_image) && (
                      <div className="mt-4 w-full max-w-4xl mx-auto px-2">
                        <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-sm font-semibold text-blue-800 mb-3 flex items-center gap-2">
                            🔍 Phân tích AI chi tiết
                          </h4>

                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {/* Original Image */}
                            {logoPreview && (
                              <div>
                                <p className="text-xs text-gray-600 mb-2 font-medium text-center">
                                  Ảnh gốc
                                </p>
                                <div className="relative rounded-lg overflow-hidden border-2 border-gray-300 shadow-md">
                                  <Image
                                    src={logoPreview}
                                    alt="Ảnh gốc"
                                    preview={true}
                                    className="w-full h-auto"
                                    style={{
                                      width: "100%",
                                      height: "auto",
                                      maxHeight: "250px",
                                      objectFit: "contain",
                                    }}
                                  />
                                </div>
                              </div>
                            )}

                            {/* GradCAM Heatmap */}
                            {result?.gradcam_image && (
                              <div>
                                <p className="text-xs text-red-600 mb-2 font-medium text-center">
                                  Bản đồ vùng bệnh (GradCAM)
                                </p>
                                <div className="relative rounded-lg overflow-hidden border-2 border-red-300 shadow-md">
                                  <img
                                    src={result.gradcam_image}
                                    alt="GradCAM Heatmap"
                                    className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                                    style={{
                                      width: "100%",
                                      height: "auto",
                                      maxHeight: "250px",
                                      objectFit: "contain",
                                    }}
                                    onClick={() =>
                                      window.open(
                                        result.gradcam_image,
                                        "_blank"
                                      )
                                    }
                                  />
                                </div>
                              </div>
                            )}

                            {/* Contour Image */}
                            {result?.contour_image && (
                              <div>
                                <p className="text-xs text-orange-600 mb-2 font-medium text-center">
                                  Khoanh vùng bệnh (Contour)
                                </p>
                                <div className="relative rounded-lg overflow-hidden border-2 border-orange-300 shadow-md">
                                  <img
                                    src={result.contour_image}
                                    alt="Contour Detection"
                                    className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                                    style={{
                                      width: "100%",
                                      height: "auto",
                                      maxHeight: "250px",
                                      objectFit: "contain",
                                    }}
                                    onClick={() =>
                                      window.open(
                                        result.contour_image,
                                        "_blank"
                                      )
                                    }
                                  />
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Legend */}
                          <div className="mt-3 flex flex-wrap items-center justify-center gap-3 text-xs">
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 bg-blue-500 rounded"></div>
                              <span className="text-blue-700">Khỏe mạnh</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 bg-green-400 rounded"></div>
                              <span className="text-green-700">Trung bình</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 bg-yellow-400 rounded"></div>
                              <span className="text-yellow-700">Cảnh báo</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 bg-red-500 rounded"></div>
                              <span className="text-red-700">Bệnh nặng</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Thông báo model không tin cậy */}
                    {result.status === "unreliable" && (
                      <div className="mt-3 p-3 sm:p-4 bg-orange-50 rounded-lg mx-2">
                        <h6 className="font-medium text-orange-800 mb-2 text-sm">
                          ⚠️ Model AI không tin cậy
                        </h6>
                        <p className="text-sm text-orange-700 mb-2">
                          Model hiện tại có thể không chính xác. Vui lòng:
                        </p>
                        <ul className="text-xs text-orange-600 space-y-1">
                          <li>• 🔄 Thử model khác trong danh sách</li>
                          <li>• 📸 Chụp ảnh rõ nét hơn</li>
                          <li>• 👨‍🌾 Tham khảo ý kiến chuyên gia</li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-3 px-2">
                <Button
                  size="middle"
                  type="default"
                  onClick={() => inputRef.current?.click()}
                  className="px-3 sm:px-4 w-full sm:w-auto"
                >
                  <Download size={16} className="mr-1 sm:mr-2 text-slate-600" />
                  <span className="text-sm">Chọn ảnh</span>
                </Button>
                <Button
                  size="middle"
                  type="default"
                  onClick={openWebcam}
                  className="px-3 sm:px-4 w-full sm:w-auto"
                  disabled={isWebcamOpen}
                >
                  <Camera size={16} className="mr-1 sm:mr-2 text-slate-600" />
                  <span className="text-sm">Chụp ảnh</span>
                </Button>
                {logoPreview && (
                  <Button
                    danger
                    onClick={handleClear}
                    className="px-3 sm:px-4 w-full sm:w-auto"
                  >
                    <span className="text-sm">Xoá ảnh</span>
                  </Button>
                )}
                <Button
                  type="primary"
                  htmlType="button"
                  onClick={handleClick}
                  className="px-3 sm:px-4 w-full sm:w-auto"
                  disabled={isLoading || !serverConnected}
                >
                  {!serverConnected ? "Server không khả dụng" : "Kiểm tra"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal Webcam */}
      <Modal
        title={
          <div className="flex items-center gap-2">
            <Video size={20} className="text-blue-600" />
            <span>Chụp ảnh từ webcam</span>
          </div>
        }
        open={isWebcamOpen}
        onCancel={closeWebcam}
        footer={[
          <Button key="cancel" onClick={closeWebcam}>
            Hủy
          </Button>,
          capturedImage ? (
            <Fragment key="captured-buttons">
              <Button key="retake" onClick={() => setCapturedImage(null)}>
                <Camera size={16} className="mr-2" />
                Chụp lại
              </Button>
              <Button
                key="confirm"
                type="primary"
                onClick={() => {
                  // Convert captured image to file and use it
                  const canvas = canvasRef.current;
                  if (canvas) {
                    canvas.toBlob(
                      (blob) => {
                        if (blob) {
                          const file = new File([blob], "webcam-capture.jpg", {
                            type: "image/jpeg",
                          });
                          handleFile(file);
                          closeWebcam();
                          message.success("Đã sử dụng ảnh từ webcam");
                        }
                      },
                      "image/jpeg",
                      0.8
                    );
                  }
                }}
              >
                Xác nhận
              </Button>
            </Fragment>
          ) : (
            <Button
              key="capture"
              type="primary"
              onClick={capturePhoto}
              disabled={!isVideoReady}
              loading={!isVideoReady}
            >
              <Camera size={16} className="mr-2" />
              {isVideoReady ? "Chụp ảnh" : "Đang tải camera..."}
            </Button>
          ),
        ]}
        width="90vw"
        style={{ maxWidth: "800px" }}
        centered
        className="webcam-modal"
      >
        <div className="text-center">
          <div
            className="relative bg-slate-100 rounded-lg overflow-hidden mb-4 mx-auto"
            style={{
              width: "100%",
              maxWidth: "100%",
              aspectRatio: "4/3",
              minHeight: "300px",
              maxHeight: "70vh",
            }}
          >
            {capturedImage ? (
              <div className="relative w-full h-full flex items-center justify-center">
                <img
                  src={capturedImage}
                  alt="Ảnh đã chụp"
                  className="max-w-full max-h-full object-contain rounded-lg"
                  style={{
                    width: "auto",
                    height: "auto",
                    maxWidth: "100%",
                    maxHeight: "100%",
                  }}
                  onLoad={() => console.log("✅ Ảnh đã load thành công")}
                  onError={(e) => {
                    console.error("❌ Lỗi load ảnh:", e);
                    console.log(
                      "📸 CapturedImage data:",
                      capturedImage.substring(0, 100)
                    );
                  }}
                />
                <div className="absolute top-2 right-2 bg-green-500 text-white px-2 py-1 rounded text-xs z-10">
                  ✅ Đã chụp
                </div>
              </div>
            ) : (
              <div className="relative w-full h-full">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover rounded-lg"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
                {!isVideoReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-800 bg-opacity-50 rounded-lg">
                    <div className="text-center text-white">
                      <Spin size="large" />
                      <p className="mt-3 text-sm">Đang khởi động camera...</p>
                      <p className="mt-1 text-xs text-slate-300">
                        Vui lòng đợi vài giây
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>
          <p className="text-slate-600 text-sm px-2">
            {capturedImage
              ? "Ảnh đã được chụp thành công! Nhấn 'Xác nhận' để sử dụng ảnh này."
              : "Đặt cây cần kiểm tra vào khung hình và nhấn 'Chụp ảnh'"}
          </p>
        </div>
      </Modal>
    </>
  );
}

export default App;
