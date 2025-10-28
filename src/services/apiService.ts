// API service để giao tiếp với Flask server
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export interface PredictionResult {
  status: 'healthy' | 'unhealthy' | 'unreliable';
  confidence: number;
  healthy_probability: number;
  unhealthy_probability: number;
  timestamp?: number;
  gradcam_image?: string; // Base64 encoded GradCAM heatmap
  contour_image?: string; // Base64 encoded contour image
}

export interface ApiResponse {
  success: boolean;
  result?: PredictionResult;
  error?: string;
  message?: string;
}

export interface ModelInfo {
  current_model: string;
  current_model_name: string;
  model_path: string;
  model_loaded: boolean;
  input_shape: string;
  output_shape: string;
  classes: string[];
  img_size: number;
}

export interface AvailableModel {
  id: string;
  name: string;
  filename: string;
  is_current: boolean;
}

export interface ModelsListResponse {
  available_models: AvailableModel[];
  current_model: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Kiểm tra server có hoạt động không
  async healthCheck(): Promise<{ status: string; model_loaded: boolean; message: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error(`Không thể kết nối đến server: ${error}`);
    }
  }

  // Gửi ảnh lên server để dự đoán
  async predictPlantHealth(imageFile: File): Promise<PredictionResult> {
    try {
      console.log('🚀 Sending image to Flask server for prediction...');
      console.log('📁 File info:', {
        name: imageFile.name,
        size: imageFile.size,
        type: imageFile.type
      });

      // Tạo FormData để gửi file
      const formData = new FormData();
      formData.append('image', imageFile);

      // Gửi request đến Flask server
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        body: formData,
        // Không set Content-Type header, để browser tự động set với boundary
      });

      console.log('📡 Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      console.log('📊 Server response:', data);

      if (!data.success || !data.result) {
        throw new Error(data.message || 'Server returned unsuccessful response');
      }

      console.log('✅ Prediction successful:', data.result);
      return data.result;

    } catch (error) {
      console.error('❌ Prediction failed:', error);
      throw new Error(`Lỗi khi gửi ảnh lên server: ${error}`);
    }
  }

  // Lấy thông tin model
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/model/info`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Get model info failed:', error);
      throw new Error(`Không thể lấy thông tin model: ${error}`);
    }
  }

  // Lấy danh sách model có sẵn
  async getAvailableModels(): Promise<ModelsListResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/models/list`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Get available models failed:', error);
      throw new Error(`Không thể lấy danh sách model: ${error}`);
    }
  }

  // Chuyển đổi model
  async switchModel(modelType: string): Promise<{ success: boolean; message: string; current_model?: string; model_name?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/model/switch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_type: modelType }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Switch model failed:', error);
      throw new Error(`Không thể chuyển đổi model: ${error}`);
    }
  }

  // Reload model
  async reloadModel(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/model/reload`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Reload model failed:', error);
      throw new Error(`Không thể reload model: ${error}`);
    }
  }

  // Kiểm tra kết nối server
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('Server connection test failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export class để có thể tạo instance mới nếu cần
export default ApiService;

// Re-export interfaces for convenience
// export type { ApiResponse, AvailableModel, ModelInfo, ModelsListResponse, PredictionResult };

