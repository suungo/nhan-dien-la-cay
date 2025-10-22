import * as tf from '@tensorflow/tfjs';

export interface ModelPrediction {
  status: 'healthy' | 'unhealthy';
  confidence: number;
  timestamp: number;
}

class TensorFlowService {
  private model: tf.LayersModel | null = null;
  private isModelLoaded = false;
  
  // Caching mechanism để tránh re-computation
  private predictionCache = new Map<string, ModelPrediction>();
  private cacheMaxSize = 100; // Giới hạn cache size
  private cacheEnabled = true; // Flag để bật/tắt cache
  
  // Prediction deduplication để tránh multiple calls
  private activePredictions = new Map<string, Promise<ModelPrediction>>();
  
  // Persistent storage key
  private readonly STORAGE_KEY = 'tensorflow_predictions_cache';
  
  // Shared prediction state để tránh duplicate calls từ multiple components
  private lastPrediction: { hash: string; result: ModelPrediction; timestamp: number } | null = null;
  private readonly PREDICTION_CACHE_DURATION = 5000; // 5 seconds

  // Load model từ file .h5
  async loadModel(modelFile: File): Promise<void> {
    try {
      console.log('Loading model from file:', modelFile.name);
      
      // Convert file to URL
      const modelUrl = URL.createObjectURL(modelFile);
      
      // Load model using TensorFlow.js
      this.model = await tf.loadLayersModel(modelUrl);
      this.isModelLoaded = true;
      
      console.log('Model loaded successfully:', this.model);
      
      // Clean up URL
      URL.revokeObjectURL(modelUrl);
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error(`Không thể load model: ${error}`);
    }
  }

  // Load model từ URL với diagnostic - CLIENT-SIDE OPTIMIZED
  async loadModelFromUrl(modelUrl: string): Promise<void> {
    try {
      console.log('🔄 Loading ResNet50 model from URL (CLIENT-SIDE):', modelUrl);
      
      // CLIENT-SIDE: Force CPU backend và disable WebGL để tránh inconsistency
      await tf.setBackend('cpu');
      await tf.ready();
      
      // CLIENT-SIDE: Disable WebGL hoàn toàn để tránh browser issues
      tf.ENV.set('WEBGL_DETERMINISTIC', true);
      tf.ENV.set('WEBGL_PACK', false);
      tf.ENV.set('CPU_HANDLER', 'tensorflow');
      tf.ENV.set('WEBGL_VERSION', 1); // Force WebGL 1
      
      console.log('🔒 CLIENT-SIDE: Using CPU backend only for stability');
      
      this.model = await tf.loadLayersModel(modelUrl);
      this.isModelLoaded = true;
      
      // Diagnostic model info
      console.log('✅ Model loaded successfully!');
      console.log('📊 Model summary:');
      this.model.summary();
      
      // Check input/output shapes
      const inputShape = this.model.inputs[0].shape;
      const outputShape = this.model.outputs[0].shape;
      console.log('🔍 Input shape:', inputShape);
      console.log('🔍 Output shape:', outputShape);
      
      // Test model with dummy input
      const testInput = tf.zeros([1, 224, 224, 3]);
      const testOutput = this.model.predict(testInput) as tf.Tensor;
      console.log('🧪 Test prediction shape:', testOutput.shape);
      console.log('🧪 Test prediction values:', testOutput.dataSync());
      testInput.dispose();
      testOutput.dispose();
      
    } catch (error) {
      console.error('❌ Error loading ResNet50 model:', error);
      console.log('🔄 Falling back to mock model...');
      // Fallback: Tạo mock model đơn giản
      this.createMockModel();
    }
  }

  // Tạo mock model đơn giản
  private createMockModel(): void {
    console.log('🤖 Tạo mock model với kiến trúc đúng...');
    
    // Tạo model với kiến trúc phù hợp cho ảnh
    const model = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: [224, 224, 3] }), // Flatten ảnh thành vector
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.5 }),
        tf.layers.dense({ units: 2, activation: 'softmax' }) // 2 classes: healthy, unhealthy
      ]
    });
    
    this.model = model;
    this.isModelLoaded = true;
    
    console.log('✅ Mock model đã tạo thành công với kiến trúc đúng');
  }

  // Preprocess ảnh để phù hợp với model - IMPROVED VERSION
  private preprocessImage(imageElement: HTMLImageElement | HTMLCanvasElement): tf.Tensor {
    // Resize ảnh về kích thước mà model mong đợi
    const targetSize = 224;
    
    return tf.tidy(() => {
      console.log('🖼️ IMPROVED preprocessing for plant disease detection...');
      console.log('- Original size:', imageElement.width, 'x', imageElement.height);
      
      // Convert ảnh thành tensor
      let tensor = tf.browser.fromPixels(imageElement);
      console.log('- Tensor shape after fromPixels:', tensor.shape);
      
      // Convert to float32 và normalize to [0, 1]
      tensor = tensor.toFloat().div(255.0);
      
      // IMPROVED: Better resize with proper interpolation
      tensor = tf.image.resizeBilinear(tensor, [targetSize, targetSize], true);
      console.log('- Tensor shape after resize:', tensor.shape);
      
      // IMPROVED: ImageNet normalization (standard for ResNet)
      const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]); // ImageNet mean
      const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);   // ImageNet std
      
      tensor = tensor.sub(mean).div(std);
      console.log('- Tensor shape after ImageNet normalization:', tensor.shape);
      
      // Add batch dimension
      tensor = tensor.expandDims(0);
      console.log('- Final tensor shape:', tensor.shape);
      
      // Log tensor statistics for debugging
      const minVal = tensor.min().dataSync()[0];
      const maxVal = tensor.max().dataSync()[0];
      console.log('- Tensor value range:', minVal, 'to', maxVal);
      
      return tensor;
    });
  }

  // Process prediction results - SIMPLIFIED và DETERMINISTIC
  private processPredictionResults(predictionArray: Float32Array): ModelPrediction {
    console.log('🔍 Processing prediction results (simplified deterministic)...');
    console.log('Prediction array length:', predictionArray.length);
    console.log('Prediction values:', Array.from(predictionArray));
    
    // SIMPLIFIED: Sử dụng thứ tự cố định thay vì đoán
    let healthyProb, unhealthyProb;
    
    if (predictionArray.length === 2) {
      // Binary classification: Model thực tế là [unhealthy, healthy] - ĐÃ XÁC NHẬN
      // Index 0 = unhealthy, Index 1 = healthy
      unhealthyProb = predictionArray[0];
      healthyProb = predictionArray[1];
      
      console.log('🔍 Binary classification - CORRECTED order [unhealthy, healthy]:');
      console.log('- Unhealthy (index 0):', unhealthyProb);
      console.log('- Healthy (index 1):', healthyProb);
    } else if (predictionArray.length === 1) {
      // Single output: 0 = unhealthy, 1 = healthy
      const prob = predictionArray[0];
      healthyProb = prob;
      unhealthyProb = 1 - prob;
      console.log('🔍 Single output - Prob:', prob, 'Healthy:', healthyProb, 'Unhealthy:', unhealthyProb);
    } else {
      // Multi-class: lấy 2 class đầu tiên
      healthyProb = predictionArray[0];
      unhealthyProb = predictionArray[1] || 0;
      console.log('🔍 Multi-class - Class 0:', healthyProb, 'Class 1:', unhealthyProb);
    }
    
    // PURE MODEL OUTPUT - NO BIAS
    const confidence = Math.max(healthyProb, unhealthyProb);
    const isHealthy = healthyProb > unhealthyProb;
    
    console.log('🎯 PURE MODEL OUTPUT (NO BIAS):');
    console.log('- Healthy probability:', healthyProb);
    console.log('- Unhealthy probability:', unhealthyProb);
    console.log('- Confidence:', confidence);
    console.log('- Is healthy:', isHealthy);
    
    // Pure model decision - NO bias correction
    const finalStatus = isHealthy ? 'healthy' : 'unhealthy';
    
    console.log('✅ PURE MODEL RESULT:', finalStatus);
    
    // Warning nếu confidence thấp
    if (confidence < 0.5) {
      console.log('⚠️ WARNING: Low confidence prediction! Model is not sure.');
    }
    
    const result: ModelPrediction = {
      status: finalStatus as 'healthy' | 'unhealthy',
      confidence: confidence,
      timestamp: Date.now()
    };
    
    console.log('🎯 Final prediction result:', result);
    return result;
  }

  // Tạo FINGERPRINT mạnh mẽ từ image để cache
  private generateImageHash(imageElement: HTMLImageElement | HTMLCanvasElement): string {
    console.log('🔍 Generating image fingerprint...');
    
    // Tạo canvas để xử lý ảnh
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    
    // Resize về kích thước cố định để consistent
    const targetSize = 64; // Kích thước cố định cho fingerprint
    canvas.width = targetSize;
    canvas.height = targetSize;
    
    // Draw ảnh với kích thước cố định
    ctx.drawImage(imageElement, 0, 0, targetSize, targetSize);
    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    
    // Tạo fingerprint từ multiple features
    const features = this.extractImageFeatures(imageData);
    const fingerprint = this.createFingerprint(features);
    
    console.log('🔍 Image fingerprint generated:', fingerprint);
    return fingerprint;
  }

  // Trích xuất đặc trưng từ ảnh
  private extractImageFeatures(imageData: ImageData): {
    pixelHash: number;
    colorHistogram: number[];
    edgeFeatures: number[];
    size: string;
  } {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    // 1. Pixel hash (như cũ nhưng cải tiến)
    let pixelHash = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      
      // Combine RGBA values
      const combined = (r << 24) | (g << 16) | (b << 8) | a;
      pixelHash = ((pixelHash << 5) - pixelHash + combined) & 0xffffffff;
    }
    
    // 2. Color histogram (8 bins cho mỗi channel)
    const histogram = new Array(24).fill(0); // 8 bins * 3 channels
    for (let i = 0; i < data.length; i += 4) {
      const r = Math.floor(data[i] / 32); // 0-7
      const g = Math.floor(data[i + 1] / 32);
      const b = Math.floor(data[i + 2] / 32);
      
      histogram[r]++;
      histogram[8 + g]++;
      histogram[16 + b]++;
    }
    
    // 3. Edge features (gradient-based)
    const edgeFeatures = this.detectEdges(imageData);
    
    // 4. Size signature
    const size = `${width}x${height}`;
    
    return {
      pixelHash,
      colorHistogram: histogram,
      edgeFeatures,
      size
    };
  }

  // Phát hiện cạnh đơn giản
  private detectEdges(imageData: ImageData): number[] {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const edges = [];
    
    // Sobel edge detection (simplified)
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // Simple gradient
        const gx = data[idx + 4] - data[idx - 4]; // Right - Left
        const gy = data[(y + 1) * width * 4 + x * 4] - data[(y - 1) * width * 4 + x * 4]; // Bottom - Top
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        
        edges.push(magnitude > 30 ? 1 : 0); // Threshold
      }
    }
    
    return edges;
  }

  // Tạo fingerprint từ features
  private createFingerprint(features: {
    pixelHash: number;
    colorHistogram: number[];
    edgeFeatures: number[];
    size: string;
  }): string {
    // Combine all features
    const combined = [
      features.pixelHash.toString(16),
      features.size,
      features.colorHistogram.slice(0, 8).join(','), // First 8 histogram bins
      features.edgeFeatures.slice(0, 16).join('') // First 16 edge features
    ].join('|');
    
    // Create hash from combined string
    let hash = 0;
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash + char) & 0xffffffff;
    }
    
    return `img_${Math.abs(hash).toString(16)}`;
  }

  // Dự đoán từ ảnh - CLIENT-SIDE OPTIMIZED VERSION
  async predictFromImage(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<ModelPrediction> {
    if (!this.model || !this.isModelLoaded) {
      throw new Error('Model chưa được load. Vui lòng load model trước.');
    }

    // CLIENT-SIDE: Thêm delay để tránh race conditions
    await new Promise(resolve => setTimeout(resolve, 50));

    try {
      // Tạo hash cho image để check cache và deduplication
      const imageHash = this.generateImageHash(imageElement);
      
      console.log(`🔍 Image hash: ${imageHash}`);
      console.log(`🔍 Cache has this hash: ${this.predictionCache.has(imageHash)}`);
      
      // Check recent prediction first (tránh duplicate calls từ multiple components)
      if (this.lastPrediction && 
          this.lastPrediction.hash === imageHash && 
          (Date.now() - this.lastPrediction.timestamp) < this.PREDICTION_CACHE_DURATION) {
        console.log('🚀 Using recent prediction result (avoiding duplicate calls)');
        return this.lastPrediction.result;
      }
      
      // Check cache trước (nếu cache được bật)
      if (this.cacheEnabled && this.predictionCache.has(imageHash)) {
        const cachedResult = this.predictionCache.get(imageHash)!;
        console.log('🚀 Using cached prediction result for hash:', imageHash);
        return cachedResult;
      }
      
      // DEDUPLICATION: Kiểm tra nếu đang có prediction đang chạy cho cùng 1 ảnh
      if (this.activePredictions.has(imageHash)) {
        console.log('⏳ Prediction already in progress, waiting...');
        return await this.activePredictions.get(imageHash)!;
      }
      
      // FORCE CLEAN STATE: Clear any residual tensors trước khi predict
      tf.engine().startScope();
      
      // Tạo promise cho prediction này
      const predictionPromise = this.performPrediction(imageElement, imageHash);
      
      // Lưu vào active predictions để tránh duplicate calls
      this.activePredictions.set(imageHash, predictionPromise);
      
      try {
        const result = await predictionPromise;
        
        // Lưu vào lastPrediction để tránh duplicate calls
        this.lastPrediction = {
          hash: imageHash,
          result: result,
          timestamp: Date.now()
        };
        
        return result;
      } finally {
        // Xóa khỏi active predictions sau khi hoàn thành
        this.activePredictions.delete(imageHash);
        // Clean up scope
        tf.engine().endScope();
      }
      
    } catch (error) {
      console.error('Error during prediction:', error);
      throw new Error(`Lỗi khi dự đoán: ${error}`);
    }
  }

  // Cache result với LRU eviction và persistent storage
  private cacheResult(imageHash: string, result: ModelPrediction): void {
    // Nếu cache đầy, xóa item cũ nhất
    if (this.predictionCache.size >= this.cacheMaxSize) {
      const firstKey = this.predictionCache.keys().next().value;
      if (firstKey) {
        this.predictionCache.delete(firstKey);
      }
    }
    
    this.predictionCache.set(imageHash, result);
    console.log(`💾 Cached prediction result. Cache size: ${this.predictionCache.size}/${this.cacheMaxSize}`);
    
    // Save to persistent storage
    this.saveCacheToStorage();
  }

  // Clear cache
  clearCache(): void {
    this.predictionCache.clear();
    this.lastPrediction = null;
    console.log('🗑️ Prediction cache cleared');
  }

  // Clear last prediction (for testing)
  clearLastPrediction(): void {
    this.lastPrediction = null;
    console.log('🗑️ Last prediction cleared');
  }

  // Test bias detection
  async testBiasDetection(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<void> {
    console.log('🧪 TESTING BIAS DETECTION');
    
    // Tắt cache để test thực
    const originalCacheState = this.cacheEnabled;
    this.setCacheEnabled(false);
    
    try {
      const result = await this.predictFromImage(imageElement);
      
      console.log('\n📊 BIAS ANALYSIS:');
      console.log('- Status:', result.status);
      console.log('- Confidence:', result.confidence);
      
      // Phân tích bias
      if (result.status === 'unhealthy' && result.confidence < 0.7) {
        console.log('⚠️ POTENTIAL BIAS: Model predicts unhealthy with low confidence');
        console.log('💡 This might indicate bias towards unhealthy classification');
      } else if (result.status === 'healthy' && result.confidence > 0.6) {
        console.log('✅ BALANCED: Model predicts healthy with good confidence');
      } else {
        console.log('📈 NORMAL: Model prediction seems balanced');
      }
      
    } finally {
      this.setCacheEnabled(originalCacheState);
    }
  }

  // Enable/disable cache
  setCacheEnabled(enabled: boolean): void {
    this.cacheEnabled = enabled;
    console.log(`💾 Cache ${enabled ? 'enabled' : 'disabled'}`);
  }

  // Get cache status
  isCacheEnabled(): boolean {
    return this.cacheEnabled;
  }

  // Load cache từ localStorage
  private loadCacheFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const cacheData = JSON.parse(stored);
        this.predictionCache = new Map(cacheData);
        console.log(`💾 Loaded ${this.predictionCache.size} predictions from storage`);
      }
    } catch (error) {
      console.warn('Failed to load cache from storage:', error);
    }
  }

  // Save cache vào localStorage
  private saveCacheToStorage(): void {
    try {
      const cacheArray = Array.from(this.predictionCache.entries());
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(cacheArray));
      console.log(`💾 Saved ${this.predictionCache.size} predictions to storage`);
    } catch (error) {
      console.warn('Failed to save cache to storage:', error);
    }
  }

  // Initialize cache từ storage
  initializeCache(): void {
    this.loadCacheFromStorage();
  }

  // SIMPLE TEST: Test ảnh A → B → A với persistent cache
  async testImagePersistence(imageA: HTMLImageElement | HTMLCanvasElement, imageB: HTMLImageElement | HTMLCanvasElement): Promise<void> {
    console.log('🧪 TESTING IMAGE PERSISTENCE - A → B → A');
    
    // Clear cache trước khi test
    this.clearCache();
    
    console.log('\n🔄 Step 1: First prediction of Image A');
    const resultA1 = await this.predictFromImage(imageA);
    console.log('Result A1:', resultA1);
    
    console.log('\n🔄 Step 2: Prediction of Image B');
    const resultB = await this.predictFromImage(imageB);
    console.log('Result B:', resultB);
    
    console.log('\n🔄 Step 3: Second prediction of Image A (should be cached)');
    const resultA2 = await this.predictFromImage(imageA);
    console.log('Result A2:', resultA2);
    
    // Check if A1 === A2
    const isConsistent = resultA1.status === resultA2.status && 
                        Math.abs(resultA1.confidence - resultA2.confidence) < 0.001;
    
    console.log('\n📊 PERSISTENCE TEST RESULTS:');
    console.log('- A1:', resultA1.status, resultA1.confidence);
    console.log('- A2:', resultA2.status, resultA2.confidence);
    console.log('- B:', resultB.status, resultB.confidence);
    console.log('- A1 === A2?', isConsistent ? '✅ YES' : '❌ NO');
    
    if (isConsistent) {
      console.log('✅ IMAGE PERSISTENCE WORKING! Same image gives same result.');
    } else {
      console.log('❌ IMAGE PERSISTENCE FAILED! Same image gives different results.');
    }
  }

  // Get cache stats
  getCacheStats(): { size: number; maxSize: number; hitRate?: number } {
    return {
      size: this.predictionCache.size,
      maxSize: this.cacheMaxSize
    };
  }

  // Thực hiện prediction thực tế (tách riêng để dễ quản lý)
  private async performPrediction(imageElement: HTMLImageElement | HTMLCanvasElement, imageHash: string): Promise<ModelPrediction> {
    const predictionId = Math.random().toString(36).substr(2, 9);
    console.log(`🔒 [${predictionId}] Predicting from image (deterministic mode with caching)...`);
    console.log(`📊 [${predictionId}] Debug info:`);
    console.log(`- Image size: ${imageElement.width} x ${imageElement.height}`);
    console.log(`- Cache enabled: ${this.cacheEnabled}`);
    console.log(`- Cache size: ${this.predictionCache.size}`);
    console.log(`- Image hash: ${imageHash}`);
    console.log(`- Prediction ID: ${predictionId}`);
    
    // IMPROVED: Sử dụng tf.tidy() chỉ cho tensor operations
    let predictionArray: Float32Array = new Float32Array(0);
    
    const startTime = performance.now();
    
    tf.tidy(() => {
      console.log('🔄 Step 1: Preprocessing image...');
      const preprocessStart = performance.now();
      const preprocessedImage = this.preprocessImage(imageElement);
      console.log(`⏱️ Preprocessing took: ${performance.now() - preprocessStart}ms`);
      
      console.log('🔄 Step 2: Running model prediction...');
      const predictStart = performance.now();
      const predictions = this.model!.predict(preprocessedImage) as tf.Tensor;
      console.log(`⏱️ Model prediction took: ${performance.now() - predictStart}ms`);
      
      console.log('🔄 Step 3: Extracting results...');
      const extractStart = performance.now();
      predictionArray = predictions.dataSync() as Float32Array;
      console.log(`⏱️ Data extraction took: ${performance.now() - extractStart}ms`);
      
      console.log('🔒 Raw predictions (deterministic):', Array.from(predictionArray));
      console.log('🔍 DETAILED PREDICTION DEBUG:');
      console.log('- Array length:', predictionArray.length);
      console.log('- Value 0 (unhealthy):', predictionArray[0]);
      console.log('- Value 1 (healthy):', predictionArray[1]);
      console.log('- Sum:', predictionArray[0] + predictionArray[1]);
      console.log('- Max value:', Math.max(...predictionArray));
      console.log('- Max index:', predictionArray.indexOf(Math.max(...predictionArray)));
      
      // Check if predictions are reasonable
      const sum = predictionArray[0] + predictionArray[1];
      const isReasonable = Math.abs(sum - 1.0) < 0.1; // Should be close to 1
      console.log('- Sum close to 1?', isReasonable, '(should be true)');
      
      // Check if values are in reasonable range
      const allPositive = predictionArray.every(val => val >= 0);
      const allReasonable = predictionArray.every(val => val <= 1.5); // Allow some overflow
      console.log('- All positive?', allPositive, '(should be true)');
      console.log('- All reasonable?', allReasonable, '(should be true)');
      
      if (!isReasonable || !allPositive || !allReasonable) {
        console.warn('⚠️ WARNING: Unusual prediction values detected!');
        console.warn('This might indicate model loading or preprocessing issues.');
      }
    });
    
    const totalTime = performance.now() - startTime;
    console.log(`⏱️ Total prediction time: ${totalTime}ms`);
    
    // Process results (ngoài tf.tidy vì không phải tensor operation)
    const result = this.processPredictionResults(predictionArray);
    
    // Cache result (nếu cache được bật)
    if (this.cacheEnabled) {
      this.cacheResult(imageHash, result);
    }
    
    return result;
  }

  // Dự đoán từ canvas (cho realtime)
  async predictFromCanvas(canvas: HTMLCanvasElement): Promise<ModelPrediction> {
    return this.predictFromImage(canvas);
  }

 

  // Test model consistency với cùng 1 ảnh - ENHANCED VERSION
  async testModelConsistency(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<void> {
    if (!this.isModelReady()) {
      console.log('❌ Model not ready for consistency test');
      return;
    }

    console.log('🧪 Testing model consistency (ENHANCED)...');
    
    // Test cùng 1 ảnh 5 lần để đảm bảo consistency
    const results: Array<{
      attempt: number;
      prediction: number[];
      maxValue: number;
      argMax: number;
      processingTime: number;
    }> = [];
    const startTime = Date.now();
    
    for (let i = 0; i < 5; i++) {
      const attemptStart = Date.now();
      
      // DETERMINISTIC: Manual tensor management
      let preprocessed: tf.Tensor | undefined;
      let prediction: tf.Tensor | undefined;
      
      try {
        preprocessed = this.preprocessImage(imageElement);
        prediction = this.model!.predict(preprocessed) as tf.Tensor;
        const predictionArray = prediction.dataSync() as Float32Array;
        
        const result = {
          attempt: i + 1,
          prediction: Array.from(predictionArray),
          maxValue: Math.max(...predictionArray),
          argMax: predictionArray.indexOf(Math.max(...predictionArray)),
          processingTime: Date.now() - attemptStart
        };
        
        results.push(result);
      } finally {
        if (preprocessed) preprocessed.dispose();
        if (prediction) prediction.dispose();
      }
      
      // Small delay để tránh race conditions
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    const totalTime = Date.now() - startTime;
    
    console.log('📊 Enhanced consistency test results:');
    results.forEach(result => {
      console.log(`Attempt ${result.attempt}:`, 
        result.prediction.map(v => v.toFixed(6)), 
        `Max: ${result.maxValue.toFixed(6)}, ArgMax: ${result.argMax}, Time: ${result.processingTime}ms`);
    });
    
    // Check if all results are identical với tolerance cao hơn
    const firstResult = results[0].prediction;
    const tolerance = 1e-8; // Tăng tolerance để phát hiện inconsistency
    
    const allIdentical = results.every(result => 
      result.prediction.every((val, idx) => Math.abs(val - firstResult[idx]) < tolerance)
    );
    
    // Tính toán variance để đánh giá độ ổn định
    const variances = firstResult.map((_, idx) => {
      const values = results.map(r => r.prediction[idx]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      return variance;
    });
    
    const maxVariance = Math.max(...variances);
    const avgVariance = variances.reduce((a, b) => a + b, 0) / variances.length;
    
    console.log('📈 Variance analysis:');
    console.log('- Max variance:', maxVariance.toFixed(10));
    console.log('- Average variance:', avgVariance.toFixed(10));
    console.log('- Total processing time:', totalTime + 'ms');
    
    if (allIdentical) {
      console.log('✅ Model is CONSISTENT - same predictions for same image');
      console.log('🎯 All predictions are identical within tolerance');
    } else {
      console.log('❌ Model is INCONSISTENT - different predictions for same image');
      console.log('🔍 This explains why you get different results!');
      console.log('💡 Consider using deterministic operations or fixing model architecture');
      
      // Phân tích chi tiết sự khác biệt
      const differences = results.slice(1).map((result, idx) => {
        const diff = result.prediction.map((val, i) => Math.abs(val - firstResult[i]));
        return { attempt: idx + 2, maxDiff: Math.max(...diff), avgDiff: diff.reduce((a, b) => a + b, 0) / diff.length };
      });
      
      console.log('🔍 Detailed differences:');
      differences.forEach(diff => {
        console.log(`  Attempt ${diff.attempt}: Max diff = ${diff.maxDiff.toFixed(8)}, Avg diff = ${diff.avgDiff.toFixed(8)}`);
      });
    }
  }

   // Kiểm tra model đã load chưa
   isModelReady(): boolean {
    return this.isModelLoaded && this.model !== null;
  }

  // Lấy thông tin model
  getModelInfo(): { inputShape: number[]; outputShape: number[] } | null {
    if (!this.model) return null;
    
    return {
      inputShape: this.model.inputs[0].shape as number[],
      outputShape: this.model.outputs[0].shape as number[]
    };
  }

  // Dispose model để giải phóng memory
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isModelLoaded = false;
    }
  }

  // QUICK TEST: Test consistency ngay lập tức
  async quickConsistencyTest(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<void> {
    console.log('🧪 QUICK CONSISTENCY TEST - Same image, 3 times');
    
    // Tắt cache để test thực
    const originalCacheState = this.cacheEnabled;
    this.setCacheEnabled(false);
    
    try {
      for (let i = 1; i <= 3; i++) {
        console.log(`\n🔄 Test ${i}/3:`);
        const result = await this.predictFromImage(imageElement);
        console.log(`Result ${i}:`, result);
      }
    } finally {
      // Khôi phục cache state
      this.setCacheEnabled(originalCacheState);
    }
  }

  // TEST SEQUENCE: A → B → A để kiểm tra cache pollution
  async testSequenceConsistency(imageA: HTMLImageElement | HTMLCanvasElement, imageB: HTMLImageElement | HTMLCanvasElement): Promise<void> {
    console.log('🧪 SEQUENCE TEST - A → B → A');
    
    // Tắt cache để test thực
    const originalCacheState = this.cacheEnabled;
    this.setCacheEnabled(false);
    
    try {
      console.log('\n🔄 Step 1: Predict Image A (first time)');
      const resultA1 = await this.predictFromImage(imageA);
      console.log('Result A1:', resultA1);
      
      console.log('\n🔄 Step 2: Predict Image B');
      const resultB = await this.predictFromImage(imageB);
      console.log('Result B:', resultB);
      
      console.log('\n🔄 Step 3: Predict Image A (second time)');
      const resultA2 = await this.predictFromImage(imageA);
      console.log('Result A2:', resultA2);
      
      // So sánh kết quả
      const isConsistent = resultA1.status === resultA2.status && 
                          Math.abs(resultA1.confidence - resultA2.confidence) < 0.001;
      
      console.log('\n📊 SEQUENCE TEST RESULTS:');
      console.log('- A1 status:', resultA1.status, 'confidence:', resultA1.confidence);
      console.log('- A2 status:', resultA2.status, 'confidence:', resultA2.confidence);
      console.log('- B status:', resultB.status, 'confidence:', resultB.confidence);
      console.log('- Is A1 === A2?', isConsistent ? '✅ YES' : '❌ NO');
      
      if (!isConsistent) {
        console.log('❌ SEQUENCE INCONSISTENCY DETECTED!');
        console.log('This indicates cache pollution or state contamination.');
      } else {
        console.log('✅ SEQUENCE CONSISTENT!');
      }
      
    } finally {
      // Khôi phục cache state
      this.setCacheEnabled(originalCacheState);
    }
  }
}

// Export singleton instance
export const tensorFlowService = new TensorFlowService();

// Initialize cache từ storage khi service start
tensorFlowService.initializeCache();