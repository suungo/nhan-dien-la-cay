"""
Flask API for Plant Health Detection using ResNet50
Model: best_resnet50_finetune.h5
Classes: ['healthy', 'unhealthy']
Model được tải từ Google Drive tự động khi server khởi động
"""

import os
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            'http://localhost:5173',
            'http://localhost:5174',
            'http://localhost:5175',
            'http://localhost:3000',
            'https://laptrinhvienweb.netlify.app',
            # Optionally allow the production API origin as well
            'https://web-production-15417.up.railway.app'
        ]
    }
})
# Global variables
model = None
current_model_type = "resnet50"  # Default model
IMG_SIZE = 224
CLASS_NAMES = ['healthy', 'unhealthy']  # Class 0 = healthy, Class 1 = unhealthy

# Model configurations
MODEL_CONFIGS = {
    "resnet50": {
        "name": "ResNet50",
        "file_id": "16meeyGZ57rg8KxfJYqAsgrfQrJRppPFI",
        "filename": "best_resnet50_finetune.h5",
        "preprocess_func": "resnet50"
    },
    "mobilenetv2": {
        "name": "MobileNetV2", 
        "file_id": "1Jb5rhT20js-PbaZFQlbHyzn_BeBUl8Xc",
        "filename": "best_mobilenetv2_finetune.h5",
        "preprocess_func": "mobilenetv2"
    }
}

def get_model_path(model_type):
    """Get model path for specific model type"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', MODEL_CONFIGS[model_type]["filename"])

def get_google_drive_url(model_type):
    """Get Google Drive download URL for specific model type"""
    file_id = MODEL_CONFIGS[model_type]["file_id"]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_model_from_gdrive(model_type="resnet50"):
    """
    Tải model từ Google Drive nếu chưa có local
    Args:
        model_type: Type of model to download ("resnet50" or "mobilenetv2")
    Returns: True nếu thành công hoặc file đã tồn tại, False nếu thất bại
    """
    try:
        model_path = get_model_path(model_type)
        google_drive_url = get_google_drive_url(model_type)
        file_id = MODEL_CONFIGS[model_type]["file_id"]
        
        # Kiểm tra xem model đã tồn tại chưa
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            # Kiểm tra xem file có phải HTML không (file bị lỗi)
            with open(model_path, 'rb') as f:
                header = f.read(15)
                if header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                    logger.warning(f"⚠️ Existing file is corrupted (HTML file), will re-download")
                    os.remove(model_path)
                else:
                    logger.info(f"✅ Model file already exists: {model_path} ({file_size:.2f} MB)")
                    return True
        
        # Tạo thư mục models nếu chưa có
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info(f"📥 Downloading {MODEL_CONFIGS[model_type]['name']} model from Google Drive...")
        logger.info(f"   File ID: {file_id}")
        logger.info(f"   Destination: {model_path}")
        
        # Download file với session để handle large files
        session = requests.Session()
        
        # Bước 1: Get initial URL để lấy confirm token (cho file lớn)
        response = session.get(google_drive_url, stream=True)
        
        # Kiểm tra xem có cần confirm download không (file lớn >100MB)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Nếu có token, download với confirm
        if token:
            logger.info("   Large file detected, confirming download...")
            params = {'export': 'download', 'id': file_id, 'confirm': token}
            response = session.get("https://drive.google.com/uc", params=params, stream=True)
        
        # Kiểm tra status code
        if response.status_code != 200:
            logger.error(f"❌ Failed to download: HTTP {response.status_code}")
            return False
        
        # Lưu file với progress
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress mỗi 10MB
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < block_size:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"   Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        
        # Verify file size
        final_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"✅ {MODEL_CONFIGS[model_type]['name']} model downloaded successfully! Size: {final_size:.2f} MB")
        
        # Verify file is not HTML
        with open(model_path, 'rb') as f:
            header = f.read(15)
            if header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                logger.error("❌ Downloaded file is HTML, not a valid model file!")
                logger.error("   Please check if the Google Drive link is public and accessible")
                os.remove(model_path)
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading model from Google Drive: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Xóa file không hoàn chỉnh nếu có
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info("   Removed incomplete file")
            except:
                pass
        
        return False

def load_model(model_type="resnet50"):
    """Load the Keras model (tải từ Google Drive nếu cần)"""
    global model, current_model_type
    
    try:
        model_path = get_model_path(model_type)
        
        # Tải model từ Google Drive nếu chưa có
        if not download_model_from_gdrive(model_type):
            logger.error(f"Failed to download {MODEL_CONFIGS[model_type]['name']} model from Google Drive")
            return False
        
        logger.info(f"Loading {MODEL_CONFIGS[model_type]['name']} model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        # Try loading with TensorFlow Keras first (most compatible)
        import tensorflow as tf
        logger.info(f"Attempting to load with TensorFlow {tf.__version__}")
        
        # Monkey patch InputLayer to ignore batch_shape (Keras 3 compatibility)
        original_input_layer_init = tf.keras.layers.InputLayer.__init__
        
        def patched_input_layer_init(self, *args, batch_shape=None, **kwargs):
            # Convert batch_shape to shape if needed
            if batch_shape is not None and 'shape' not in kwargs:
                kwargs['shape'] = batch_shape[1:]  # Skip batch dimension
            # Remove batch_shape from kwargs
            kwargs.pop('batch_shape', None)
            original_input_layer_init(self, *args, **kwargs)
        
        # Apply patch
        tf.keras.layers.InputLayer.__init__ = patched_input_layer_init
        
        # Try multiple loading strategies
        loading_strategies = [
            ("Patched InputLayer", lambda: tf.keras.models.load_model(model_path, compile=False)),
            ("Standard load", lambda: tf.keras.models.load_model(model_path, compile=False)),
            ("Legacy load", lambda: tf.keras.models.load_model(model_path, compile=False, custom_objects={})),
        ]
        
        last_error = None
        for strategy_name, load_func in loading_strategies:
            try:
                logger.info(f"   Trying: {strategy_name}")
                model = load_func()

                # Some saved objects may be plain Layer/Sequential instances without defined
                # input tensors (this manifests as "The layer <name> has never been called...").
                # Ensure the loaded object has defined input tensors by attempting to build
                # it or wrapping it into a Functional Model with a new Input if necessary.
                try:
                    _ = model.input  # access to check if inputs exist
                except Exception:
                    logger.info("Loaded model has no defined input tensors — attempting to build or wrap it.")
                    try:
                        # Try to build in-place (works for Sequential/layer with build method)
                        model.build((None, IMG_SIZE, IMG_SIZE, 3))
                        logger.info("Built model in-place via model.build((None, IMG_SIZE, IMG_SIZE, 3)).")
                    except Exception as build_exc:
                        logger.info("model.build failed, attempting to wrap the layer into a Functional Model...")
                        try:
                            inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                            outputs = model(inp)
                            model = tf.keras.Model(inputs=inp, outputs=outputs)
                            logger.info("Wrapped loaded Layer/Sequential into a Functional Model.")
                        except Exception as wrap_exc:
                            logger.warning("Failed to wrap/build the loaded object to expose inputs.")
                            # Re-raise the original exception so the loading strategy can continue
                            raise

                # Restore original InputLayer
                tf.keras.layers.InputLayer.__init__ = original_input_layer_init

                logger.info(f"✅ {MODEL_CONFIGS[model_type]['name']} model loaded successfully with {strategy_name}")
                try:
                    logger.info(f"   Input shape: {model.input_shape}")
                except Exception:
                    logger.info("   Input shape: (unknown)")
                try:
                    logger.info(f"   Output shape: {model.output_shape}")
                except Exception:
                    logger.info("   Output shape: (unknown)")
                current_model_type = model_type
                return True
            except Exception as e:
                logger.warning(f"   {strategy_name} failed: {str(e)[:100]}")
                last_error = e
                continue
        
        # Restore original InputLayer if all failed
        tf.keras.layers.InputLayer.__init__ = original_input_layer_init
        
        # If TensorFlow methods failed, try standalone Keras
        try:
            import keras
            logger.info(f"Attempting to load with standalone Keras {keras.__version__}")
            
            # Try with safe_mode=False for Keras 3
            try:
                model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                logger.info("✅ Model loaded successfully with Keras 3 (safe_mode=False)")
                current_model_type = model_type
                return True
            except TypeError:
                # Keras 2 doesn't have safe_mode parameter
                model = keras.models.load_model(model_path, compile=False)
                logger.info("✅ Model loaded successfully with standalone Keras")
                current_model_type = model_type
                return True
                        
        except Exception as e2:
            logger.error(f"Failed to load with standalone Keras: {e2}")
            logger.error(f"All loading methods failed. Last TF error: {last_error}")
            return False
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image_data, model_type="resnet50"):
    """
    Preprocess image theo đúng cách train model với ResNet50 hoặc MobileNetV2 ImageNet preprocessing
    """
    try:
        # Import preprocess_input từ TensorFlow (giống training)
        if model_type == "resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif model_type == "mobilenetv2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply preprocessing (CHÍNH XÁC như training)
        img_array = preprocess_input(img_array)
        
        logger.info(f"Image preprocessed for {MODEL_CONFIGS[model_type]['name']}: shape={img_array.shape}, dtype={img_array.dtype}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for {model_type}: {str(e)}")
        raise

def generate_gradcam(img_array, class_idx=1, model_type=None):
    """
    Generate GradCAM heatmap for visualization
    class_idx: 0 = healthy, 1 = unhealthy
    model_type: "resnet50" or "mobilenetv2"
    """
    try:
        import tensorflow as tf
        
        if model_type is None:
            model_type = current_model_type
            
        logger.info(f"🔍 Generating GradCAM for {MODEL_CONFIGS[model_type]['name']} model, class {class_idx}...")
        
        # Tìm layer convolutional cuối cùng
        last_conv_layer_name = None
        
        # Quét tất cả layers và tìm layer tích chập cuối cùng (bao gồm Depthwise/Separable cho MobileNetV2)
        conv_like_types = (
            tf.keras.layers.Conv2D,
            getattr(tf.keras.layers, 'DepthwiseConv2D', tuple()),
            getattr(tf.keras.layers, 'SeparableConv2D', tuple())
        )
        for layer in reversed(model.layers):
            if isinstance(layer, conv_like_types):
                last_conv_layer_name = layer.name
                logger.info(f"✅ Found conv-like layer: {last_conv_layer_name} ({type(layer).__name__})")
                break
        
        # Nếu không tìm thấy, thử tìm theo tên với logic khác nhau cho từng model
        if not last_conv_layer_name:
            logger.info("Trying to find conv layer by name...")
            
            if model_type == "mobilenetv2":
                # Ưu tiên các layer phổ biến của MobileNetV2 nếu tồn tại
                preferred_layer_names = [
                    'Conv_1',          # Last 1x1 conv before global pooling
                    'out_relu',        # Activation after Conv_1
                    'block_16_project' # Cuối backbone
                ]
                for name in preferred_layer_names:
                    try:
                        _ = model.get_layer(name)
                        last_conv_layer_name = name
                        logger.info(f"✅ Found preferred MobileNetV2 layer: {last_conv_layer_name}")
                        break
                    except Exception:
                        pass

                if not last_conv_layer_name:
                    # MobileNetV2 specific layer names - tìm layer có 'conv' và không phải activation/bn
                    for layer in reversed(model.layers):
                        layer_name_lower = layer.name.lower()
                        if 'conv' in layer_name_lower:
                            # Skip batch norm, activation, pooling layers
                            if not any(skip in layer_name_lower for skip in ['bn', 'relu', 'pool', 'activation', 'dropout']):
                                last_conv_layer_name = layer.name
                                logger.info(f"✅ Found MobileNetV2 conv layer: {last_conv_layer_name}")
                                break
                
                # Nếu vẫn không tìm thấy, thử tìm layer có 'block' hoặc 'expanded'
                if not last_conv_layer_name:
                    for layer in reversed(model.layers):
                        layer_name_lower = layer.name.lower()
                        if any(keyword in layer_name_lower for keyword in ['block', 'expanded', 'depthwise']):
                            if not any(skip in layer_name_lower for skip in ['bn', 'relu', 'pool', 'activation']):
                                last_conv_layer_name = layer.name
                                logger.info(f"✅ Found MobileNetV2 block layer: {last_conv_layer_name}")
                                break
            else:
                # ResNet50 specific layer names
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower() and not 'bn' in layer.name.lower():
                        last_conv_layer_name = layer.name
                        logger.info(f"✅ Found ResNet50 conv layer: {last_conv_layer_name}")
                        break
        
        if not last_conv_layer_name:
            logger.warning(f"⚠️ No specific conv layer found, trying fallback method...")
            
            # Fallback: tìm bất kỳ layer nào có output shape phù hợp
            for layer in reversed(model.layers):
                try:
                    # Kiểm tra xem layer có output shape không
                    if hasattr(layer, 'output_shape') and layer.output_shape:
                        output_shape = layer.output_shape
                        # Tìm layer có 4D output (batch, height, width, channels)
                        if len(output_shape) == 4 and output_shape[1] is not None and output_shape[2] is not None:
                            last_conv_layer_name = layer.name
                            logger.info(f"✅ Found fallback layer: {last_conv_layer_name} with shape {output_shape}")
                            break
                except Exception as e:
                    continue
        
        if not last_conv_layer_name:
            logger.error(f"❌ No suitable layer found for GradCAM in {MODEL_CONFIGS[model_type]['name']}")
            # Log all layer names for debugging
            logger.info("Available layers:")
            for i, layer in enumerate(model.layers):
                try:
                    output_shape = getattr(layer, 'output_shape', 'No output shape')
                    logger.info(f"  {i}: {layer.name} ({type(layer).__name__}) - {output_shape}")
                except:
                    logger.info(f"  {i}: {layer.name} ({type(layer).__name__}) - Error getting shape")
            return None
        
        # Tạo grad model - sử dụng cách tiếp cận đơn giản và an toàn
        try:
            # Bảo đảm có thể truy cập input/output: nếu model là Sequential chưa build, wrap lại thành Functional
            working_model = model
            try:
                _ = working_model.input  # thử truy cập input
            except Exception:
                logger.info("Top model has no input — wrapping into Functional Model for Grad-CAM...")
                inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                out = working_model(inp)
                working_model = tf.keras.Model(inputs=inp, outputs=out)

            # Build model graph bằng cách gọi 1 lần
            logger.info("Building model by calling it with input...")
            _ = working_model(img_array)
            logger.info("Model built successfully")

            # Lấy layer đặc trưng
            conv_layer = working_model.get_layer(last_conv_layer_name)

            # Tạo model mới từ input đến conv output và prediction output
            grad_model = tf.keras.Model(
                inputs=working_model.input,
                outputs=[conv_layer.output, working_model.output]
            )
            
            logger.info(f"✅ Grad model created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating grad model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Compute the gradient of the class output value with respect to the feature map
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_output = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Kiểm tra gradients
        if grads is None:
            logger.error("Gradients are None - cannot compute GradCAM")
            return None
        
        # Pool the gradients across the feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature map by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        logger.info(f"GradCAM computation: conv_outputs.shape={conv_outputs.shape}, pooled_grads.shape={pooled_grads.shape}")
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create the heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # Normalize
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
        else:
            logger.warning("Heatmap max is 0, cannot normalize")
        
        logger.info(f"Heatmap generated: shape={heatmap.shape}, min={np.min(heatmap):.4f}, max={np.max(heatmap):.4f}")
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error generating GradCAM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def apply_jet_colormap(value):
    """Apply JET colormap to a value between 0 and 1"""
    # JET colormap approximation
    if value < 0.125:
        r, g, b = 0, 0, int(255 * (0.5 + value / 0.125 * 0.5))
    elif value < 0.375:
        r, g, b = 0, int(255 * (value - 0.125) / 0.25), 255
    elif value < 0.625:
        r, g, b = int(255 * (value - 0.375) / 0.25), 255, int(255 * (1 - (value - 0.375) / 0.25))
    elif value < 0.875:
        r, g, b = 255, int(255 * (1 - (value - 0.625) / 0.25)), 0
    else:
        r, g, b = int(255 * (1 - (value - 0.875) / 0.125 * 0.5)), 0, 0
    return (r, g, b)

def create_gradcam_image(original_image_data, heatmap):
    """Create GradCAM overlay image using PIL"""
    try:
        logger.info("Creating GradCAM overlay image...")
        
        # Load original image
        img = Image.open(io.BytesIO(original_image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        
        # Resize heatmap using PIL
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heatmap_resized = np.array(heatmap_pil) / 255.0
        
        logger.info(f"Applying JET colormap to heatmap...")
        
        # Apply JET colormap manually
        heatmap_colored = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                heatmap_colored[i, j] = apply_jet_colormap(heatmap_resized[i, j])
        
        # Overlay heatmap on original image
        superimposed_img = heatmap_colored * 0.4 + img_array * 0.6
        superimposed_img = np.uint8(superimposed_img)
        
        # Convert to base64
        pil_img = Image.fromarray(superimposed_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info(f"✅ GradCAM image created successfully (size: {len(img_base64)} chars)")
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error creating GradCAM image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_contour_image(original_image_data, heatmap, threshold=0.5):
    """Create image with disease contour highlighted"""
    try:
        logger.info("Creating contour image...")
        
        # Load original image
        img = Image.open(io.BytesIO(original_image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        
        # Resize heatmap
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heatmap_resized = np.array(heatmap_pil) / 255.0
        
        # Threshold to find disease regions
        binary_mask = (heatmap_resized > threshold).astype(np.uint8) * 255

        # Create PIL image from array
        result_img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(result_img)

        # Try to use OpenCV for robust contour extraction (best quality)
        try:
            import cv2

            logger.info("Using OpenCV to find contours for precise disease outline.")

            mask = binary_mask.astype(np.uint8)

            # Find external contours
            contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Compatibility: OpenCV returns (contours, hierarchy) or (image, contours, hierarchy)
            if len(contours_info) == 3:
                _, contours, _ = contours_info
            else:
                contours, _ = contours_info

            if contours and len(contours) > 0:
                # Choose the largest contour by area
                largest = max(contours, key=cv2.contourArea)

                # Approximate polygon to simplify shape (epsilon proportional to perimeter)
                peri = cv2.arcLength(largest, True)
                epsilon = max(1.0, 0.01 * peri)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                # Convert contour points to list of (x, y) tuples for PIL
                poly = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

                if len(poly) >= 3:
                    # Draw polygon outline (red) and a semi-transparent fill
                    draw.polygon(poly, outline=(255, 0, 0))

                    # For thicker outline, draw lines between consecutive points
                    for i in range(len(poly)):
                        p1 = poly[i]
                        p2 = poly[(i + 1) % len(poly)]
                        draw.line([p1, p2], fill=(255, 0, 0), width=3)
                else:
                    logger.warning("Detected contour has too few points, falling back to bounding box")
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0:
                        y_min, y_max = ys.min(), ys.max()
                        x_min, x_max = xs.min(), xs.max()
                        padding = 10
                        bbox = [
                            max(0, x_min - padding),
                            max(0, y_min - padding),
                            min(IMG_SIZE, x_max + padding),
                            min(IMG_SIZE, y_max + padding)
                        ]
                        draw.rectangle(bbox, outline=(255, 0, 0), width=3)
            else:
                logger.warning("No contours found by OpenCV, falling back to bounding box")
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()
                    padding = 10
                    bbox = [
                        max(0, x_min - padding),
                        max(0, y_min - padding),
                        min(IMG_SIZE, x_max + padding),
                        min(IMG_SIZE, y_max + padding)
                    ]
                    draw.rectangle(bbox, outline=(255, 0, 0), width=3)

        except Exception as e:
            # If OpenCV isn't installed or fails, fallback to bounding box but inform user
            logger.warning("OpenCV not available or failed to run contour detection: %s", str(e))
            logger.info("Falling back to simple bounding box. For precise polygonal contours, install opencv-python: pip install opencv-python")

            disease_pixels = np.where(binary_mask > 0)

            if len(disease_pixels[0]) > 0:
                # Get bounding box of disease region
                y_min, y_max = disease_pixels[0].min(), disease_pixels[0].max()
                x_min, x_max = disease_pixels[1].min(), disease_pixels[1].max()

                logger.info(f"Disease region found: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

                # Draw polygon-like rectangle (thick outline) as fallback
                padding = 10
                bbox = [
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    min(IMG_SIZE, x_max + padding),
                    min(IMG_SIZE, y_max + padding)
                ]

                # Draw thick red outline rectangle
                draw.rectangle(bbox, outline=(255, 0, 0), width=3)
            else:
                logger.warning("No disease pixels found above threshold")
        
        # Convert to base64
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info(f"✅ Contour image created successfully (size: {len(img_base64)} chars)")
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error creating contour image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_plant_health(image_data, include_gradcam=False, model_type=None):
    """Predict plant health from image"""
    global model, current_model_type
    
    # Đảm bảo model luôn được load nếu chạy production lazy hoặc restart worker
    if model is None:
        logger.warning("Model not loaded in memory, attempting to load now...")
        load_model(current_model_type)
        if model is None:
            raise Exception("Model loading failed!")
    
    # Use provided model_type or current loaded model
    if model_type is None:
        model_type = current_model_type
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_data, model_type)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)
        
        logger.info(f"Raw predictions: {predictions[0]}")
        
        # Parse results
        # Model output: [healthy_prob, unhealthy_prob] với softmax
        healthy_prob = float(predictions[0][0])      # Class 0 = healthy
        unhealthy_prob = float(predictions[0][1])    # Class 1 = unhealthy
        
        # Determine status
        is_healthy = healthy_prob > unhealthy_prob
        confidence = max(healthy_prob, unhealthy_prob)
        
        result = {
            "status": "healthy" if is_healthy else "unhealthy",
            "confidence": float(confidence),  # Không làm tròn
            "healthy_probability": float(healthy_prob),  # Không làm tròn
            "unhealthy_probability": float(unhealthy_prob),  # Không làm tròn
            "class_probabilities": {
                "healthy": float(healthy_prob),
                "unhealthy": float(unhealthy_prob)
            }
        }
        
        # Generate GradCAM if requested
        if include_gradcam:
            logger.info("=" * 60)
            logger.info("STARTING GRADCAM GENERATION")
            logger.info("=" * 60)
            try:
                # Always use unhealthy class for better visualization
                target_class = 1  # Focus on unhealthy class
                logger.info(f"Target class for GradCAM: {target_class}")
                
                heatmap = generate_gradcam(processed_image, class_idx=target_class, model_type=model_type)
                logger.info(f"Heatmap result: {heatmap is not None}")
                
                if heatmap is not None:
                    logger.info("Creating GradCAM visualization images...")
                    
                    # Create GradCAM heatmap
                    gradcam_image = create_gradcam_image(image_data, heatmap)
                    logger.info(f"GradCAM image created: {gradcam_image is not None}")
                    if gradcam_image:
                        result["gradcam_image"] = gradcam_image
                        logger.info("✅ gradcam_image added to result")
                    else:
                        logger.error("❌ gradcam_image creation returned None")
                    
                    # Create contour image
                    contour_image = create_contour_image(image_data, heatmap)
                    logger.info(f"Contour image created: {contour_image is not None}")
                    if contour_image:
                        result["contour_image"] = contour_image
                        logger.info("✅ contour_image added to result")
                    else:
                        logger.error("❌ contour_image creation returned None")
                    
                    logger.info("✅✅✅ GradCAM and contour generated successfully")
                else:
                    logger.error("❌❌❌ GradCAM heatmap generation failed - returned None")
            except Exception as e:
                logger.error(f"❌ ERROR generating GradCAM: {str(e)}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole request if GradCAM fails
            
            logger.info("=" * 60)
            logger.info(f"Final result keys: {list(result.keys())}")
            logger.info("=" * 60)
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Đảm bảo model luôn được load nếu chạy production lazy hoặc restart worker
    if model is None:
        logger.warning("Model not loaded in memory, attempting to load now...")
        load_model(current_model_type)
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Plant Health Detection API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint để dự đoán sức khỏe cây từ ảnh"""
    try:
        # Check if model is loaded
        if model is None:
                return jsonify({
                "success": False,
                    "error": "Model not loaded",
                "message": "Please wait for the model to load or check server logs"
                }), 500
        
        # Check if image file is present (accept both 'file' and 'image' keys)
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'image' in request.files:
            file = request.files['image']
        else:
            # Log what keys are available for debugging
            logger.error(f"No file found. Available keys: {list(request.files.keys())}")
            logger.error(f"Request form data: {list(request.form.keys())}")
            return jsonify({
                "success": False,
                "error": "No file provided",
                "message": "Please upload an image file with key 'file' or 'image'"
            }), 400
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename",
                "message": "Please select a valid image file"
            }), 400
        
        # Read image data
        image_data = file.read()
        
        # Predict with GradCAM enabled
        result = predict_plant_health(image_data, include_gradcam=True)
        
        # Wrap response theo format frontend expect
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    # Đảm bảo model luôn được load nếu chạy production lazy hoặc restart worker
    if model is None:
        logger.warning("Model not loaded in memory, attempting to load now...")
        load_model(current_model_type)
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Model is not available"
            }), 500
    
    # Get layer information for debugging
    import tensorflow as tf
    layer_info = []
    for i, layer in enumerate(model.layers):
        layer_info.append({
            "index": i,
            "name": layer.name,
            "type": type(layer).__name__,
            "is_conv": isinstance(layer, tf.keras.layers.Conv2D)
        })
    
    return jsonify({
        "current_model": current_model_type,
        "current_model_name": MODEL_CONFIGS[current_model_type]["name"],
        "model_path": get_model_path(current_model_type),
        "model_loaded": True,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "layers": layer_info
    })

@app.route('/models/list', methods=['GET'])
def list_models():
    """Get list of available models"""
    return jsonify({
        "available_models": [
            {
                "id": model_id,
                "name": config["name"],
                "filename": config["filename"],
                "is_current": model_id == current_model_type
            }
            for model_id, config in MODEL_CONFIGS.items()
        ],
        "current_model": current_model_type
    })

@app.route('/model/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global model, current_model_type
    
    try:
        data = request.get_json()
        if not data or 'model_type' not in data:
            return jsonify({
                "success": False,
                "message": "model_type is required"
            }), 400
        
        model_type = data['model_type']
        if model_type not in MODEL_CONFIGS:
            return jsonify({
                "success": False,
                "message": f"Invalid model type. Available: {list(MODEL_CONFIGS.keys())}"
            }), 400
        
        if model_type == current_model_type:
            return jsonify({
                "success": True,
                "message": f"{MODEL_CONFIGS[model_type]['name']} is already loaded"
            })
        
        logger.info(f"Switching from {MODEL_CONFIGS[current_model_type]['name']} to {MODEL_CONFIGS[model_type]['name']}")
        
        # Unload current model
        model = None
        
        # Load new model
        success = load_model(model_type)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully switched to {MODEL_CONFIGS[model_type]['name']}",
                "current_model": current_model_type,
                "model_name": MODEL_CONFIGS[model_type]["name"]
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to load {MODEL_CONFIGS[model_type]['name']}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload current model"""
    global model
    try:
        model = None
        success = load_model(current_model_type)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"{MODEL_CONFIGS[current_model_type]['name']} model reloaded successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to reload {MODEL_CONFIGS[current_model_type]['name']} model"
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

def _startup_load_model():
    """Attempt to load model at process start. If it fails due to timeout,
    app still serves and model will be lazily loaded on first request."""
    try:
        os.makedirs('models', exist_ok=True)
        logger.info("Starting Plant Health Detection API...")
        logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")
        logger.info(f"Default model: {current_model_type}")
        logger.info(f"Classes: {CLASS_NAMES}")
        if load_model(current_model_type):
            logger.info(f"✅ {MODEL_CONFIGS[current_model_type]['name']} model loaded successfully!")
            try:
                logger.info(f"   Input shape: {model.input_shape}")
                logger.info(f"   Output shape: {model.output_shape}")
            except Exception:
                pass
        else:
            logger.error(f"❌ Failed to load {MODEL_CONFIGS[current_model_type]['name']} model!")
            logger.error("   API will start but predictions will fail until /model/reload")
    except Exception as e:
        logger.error(f"Startup load error: {e}")

if __name__ == '__main__':
    _startup_load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
