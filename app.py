import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, render_template, flash, redirect, url_for, Response
import warnings
import uuid
from io import BytesIO

print("Starting Flask app...")

# Suppress PyTorch FutureWarning for backward hooks
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define Hook class for Grad-CAM
class Hook:
    def __init__(self):
        self.forward_out = None
        self.backward_out = None

    def register_hook(self, module):
        self.hook_f = module.register_forward_hook(self.forward_hook)
        self.hook_b = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.forward_out = output

    def backward_hook(self, module, grad_in, grad_out):
        self.backward_out = grad_out[0]

    def unregister_hook(self):
        self.hook_f.remove()
        self.hook_b.remove()

# Define SEBlock
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# Define ResEmoteNet
class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se = SEBlock(256)
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.se(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

# Define emotion labels
class_labels = ['Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Neutral']

# Load the model
model = ResEmoteNet(num_classes=7).to(device)
model_path = "best_model_resemotenet_73.pth"
if not os.path.exists(model_path):
    model_path = "final_model_resemotenet_73.pth"
    print("Best model not found, using final model.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Register Grad-CAM hook
final_layer = model.conv3
hook = Hook()
hook.register_hook(final_layer)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Text settings for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (154, 1, 254)  # Neon pink in BGR
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4

def detect_emotion(pil_crop_img):
    try:
        if not isinstance(pil_crop_img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(pil_crop_img)}")
        vid_fr_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
        logits = model(vid_fr_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        predicted_class_idx = predicted_class.item()

        # Backward pass for Grad-CAM
        one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_().to(device)
        one_hot_output[0][predicted_class_idx] = 1
        logits.backward(gradient=one_hot_output, retain_graph=True)

        gradients = hook.backward_out
        feature_maps = hook.forward_out

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = cam.clamp(min=0).squeeze()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().detach().numpy()

        scores = probabilities.cpu().detach().numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        return rounded_scores, cam
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return None, None

def plot_heatmap(x, y, w, h, cam, pil_crop_img, image):
    try:
        cam = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        roi = image[y:y + h, x:x + w, :]
        overlay = heatmap * transparency + roi / 255 * (1 - transparency)
        overlay = np.clip(overlay, 0, 1)
        image[y:y + h, x:x + w, :] = np.uint8(255 * overlay)
    except Exception as e:
        print(f"Error in plot_heatmap: {e}")

def update_max_emotion(rounded_scores):
    max_index = np.argmax(rounded_scores)
    return class_labels[max_index]

def print_max_emotion(x, y, max_emotion, image):
    org = (x, y - 15)
    cv2.putText(image, max_emotion, org, font, font_scale, font_color, thickness, line_type)

def print_all_emotion(x, y, w, rounded_scores, image):
    org = (x + w + 10, y)
    for index, value in enumerate(class_labels):
        emotion_str = f'{value}: {rounded_scores[index]:.2f}'
        y_offset = org[1] + (index * 30)
        cv2.putText(image, emotion_str, (org[0], y_offset), font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(image, use_mediapipe=True):
    faces = {}
    try:
        if use_mediapipe:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image)
                if not results.detections:
                    return faces
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    faces[f'face_{i}'] = {'facial_area': [x1, y1, x2, y2]}
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pil_crop_img = Image.fromarray(image[y1:y2, x1:x2])
                    rounded_scores, cam = detect_emotion(pil_crop_img)
                    if rounded_scores is None or cam is None:
                        continue
                    max_emotion = update_max_emotion(rounded_scores)
                    plot_heatmap(x1, y1, x2 - x1, y2 - y1, cam, pil_crop_img, image)
                    print_max_emotion(x1, y1, max_emotion, image)
                    print_all_emotion(x1, y1, x2 - x1, rounded_scores, image)
        return faces
    except Exception as e:
        print(f"Error in detect_bounding_box: {e}")
        return {}

def process_frame(frame, frame_id, results):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_bounding_box(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Collect results for this frame
    frame_results = {'frame_id': frame_id, 'faces': []}
    for face_id, data in faces.items():
        x1, y1, x2, y2 = data['facial_area']
        pil_crop_img = Image.fromarray(frame_rgb[y1:y2, x1:x2])
        scores, _ = detect_emotion(pil_crop_img)
        if scores:
            emotion_scores = dict(zip(class_labels, [round(score, 2) for score in scores]))
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            frame_results['faces'].append({
                'face_id': face_id,
                'emotion_scores': emotion_scores,
                'max_emotion': max_emotion
            })
    if frame_results['faces']:
        results.append(frame_results)
    return frame_bgr

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            image = cv2.imread(filepath)
            if image is None:
                flash('Invalid image file')
                return redirect(request.url)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detect_bounding_box(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            output_filename = f'processed_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, image_bgr)
            
            # Prepare results
            results = []
            for face_id, data in faces.items():
                x1, y1, x2, y2 = data['facial_area']
                pil_crop_img = Image.fromarray(image_rgb[y1:y2, x1:x2])
                scores, _ = detect_emotion(pil_crop_img)
                if scores:
                    emotion_scores = dict(zip(class_labels, [round(score, 2) for score in scores]))
                    max_emotion = max(emotion_scores, key=emotion_scores.get)
                    results.append({
                        'face_id': face_id,
                        'emotion_scores': emotion_scores,
                        'max_emotion': max_emotion
                    })
            
            hook.unregister_hook()
            return render_template('result.html', output_image=output_filename, results=results)
    
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.mp4'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                flash('Invalid video file')
                return redirect(request.url)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize output video
            output_filename = f'processed_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames (sample every 30th frame for results to reduce output size)
            results = []
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, frame_id, results if frame_id % 30 == 0 else [])
                out.write(processed_frame)
                frame_id += 1
            
            cap.release()
            out.release()
            
            hook.unregister_hook()
            return render_template('video_result.html', output_video=output_filename, results=results)
    
    return render_template('video_upload.html')

@app.route('/camera')
def camera_feed():
    return render_template('camera.html')

def generate_camera_feed():
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, 0, [])  # Process without storing results
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/video_stream')
def video_stream():
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)