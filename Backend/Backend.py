import os
import io
import json
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from neo4j import GraphDatabase

from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model_utils import build_model, DiseaseOnlyWrapper
import sys
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import Mobile_V3
MiniCNNGuard = Mobile_V3.MiniCNNGuard

app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'Frontend', 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'Frontend', 'static'))
app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODELS_DIR = os.path.join(ROOT_DIR, 'models')

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print(f"Neo4j driver initialized for {NEO4J_URI}.")
except Exception as e:
    print("Error initializing Neo4j driver:", e)
    neo4j_driver = None

try:
    with open(os.path.join(MODELS_DIR, "class_names.json"), 'r') as f:
        class_names = json.load(f)
    with open(os.path.join(MODELS_DIR, "species_names.json"), 'r') as f:
        species_names = json.load(f)
except FileNotFoundError:
    class_names = []
    species_names = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading dual models to {device}...")

# 1. ResNet50 + CBAM Model
model_resnet = build_model('resnet50', num_disease=len(class_names), num_species=len(species_names))
RESNET_MODEL_PATH = os.path.join(ROOT_DIR, 'models/hoya_multitask_resnet50_final.pth')
try:
    model_resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device), strict=False)
    model_resnet.to(device)
    model_resnet.eval()
    print("Main ResNet50 model loaded successfully.")
except Exception as e:
    print(f"Error loading ResNet50 model: {e}")

GUARD_MODEL_PATH = os.path.join(ROOT_DIR, 'models/MobileNetV3 Small - Guard Final Model.pth')
guard_model = MiniCNNGuard(pretrained=False).to(device)
if os.path.exists(GUARD_MODEL_PATH):
    guard_model.load_state_dict(torch.load(GUARD_MODEL_PATH, map_location=device))
    guard_model.eval()
    print("Mini-CNN Guard model loaded successfully.")
else:
    print(f"Warning: Mini-CNN Guard model not found at {GUARD_MODEL_PATH}")

resnet_wrapper = DiseaseOnlyWrapper(model_resnet)
resnet_target_layers = [model_resnet.feature_extractor[-1]]

cam_methods_resnet = {
    'gradcam': GradCAM(model=resnet_wrapper, target_layers=resnet_target_layers),
    'gradcam_pp': GradCAMPlusPlus(model=resnet_wrapper, target_layers=resnet_target_layers),
    'eigencam': EigenCAM(model=resnet_wrapper, target_layers=resnet_target_layers),
    'layercam': LayerCAM(model=resnet_wrapper, target_layers=resnet_target_layers),
}


def get_cbam_attention_map(model, input_tensor):
    """Extract CBAM spatial attention map via forward hook."""
    spatial_attn = {}
    def hook_fn(module, inp, out):
        with torch.no_grad():
            x = inp[0]
            avg_out_ch = model.attention.fc2(model.attention.relu(model.attention.fc1(F.adaptive_avg_pool2d(x, 1))))
            max_out_ch = model.attention.fc2(model.attention.relu(model.attention.fc1(F.adaptive_max_pool2d(x, 1))))
            channel_att = model.attention.sigmoid_channel(avg_out_ch + max_out_ch)
            x_ch = x * channel_att
            avg_s = torch.mean(x_ch, dim=1, keepdim=True)
            max_s, _ = torch.max(x_ch, dim=1, keepdim=True)
            spatial_attn['map'] = model.attention.sigmoid_spatial(
                model.attention.conv_spatial(torch.cat([avg_s, max_s], dim=1))
            )
    handle = model.attention.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    attn_map = spatial_attn['map'][0, 0].cpu().numpy()
    return attn_map

class AdaptiveSharpen:
    def __init__(self, blur_threshold=100.0, sharpen_strength=1.2):
        self.blur_threshold = blur_threshold
        self.sharpen_strength = sharpen_strength
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
            sharpened = cv2.addWeighted(img_np, 1 + self.sharpen_strength, blurred, -self.sharpen_strength, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            return Image.fromarray(sharpened)
        return img

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_eq)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

class LetterboxPad:
    def __init__(self, target_size=(224, 224), fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        img_w, img_h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", self.target_size, self.fill_color)
        new_img.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return new_img

eval_transform = transforms.Compose([
    LetterboxPad(target_size=(IMG_SIZE, IMG_SIZE)),
    AdaptiveSharpen(blur_threshold=100.0, sharpen_strength=1.2),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

import rembg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        input_image = Image.open(file.stream).convert('RGB')
        input_image.thumbnail((1200, 1200))
        
        # Remove background using rembg (U2-Net)
        output_rgba = rembg.remove(input_image)
        
        # Create white background version for clean AI classification & display
        white_bg = Image.new("RGB", output_rgba.size, (255, 255, 255))
        white_bg.paste(output_rgba, mask=output_rgba.split()[3])
        
        buffered = io.BytesIO()
        white_bg.save(buffered, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_b64}"
        })
    except Exception as e:
        import traceback
        print(f"Error in /remove-bg: {e}", flush=True)
        traceback.print_exc()
        return jsonify({'error': f"Gagal menghapus background: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')

        width, height = image.size
        if width < 100 or height < 100:
            return jsonify({
                'error': 'Gambar Terdeteksi Bukan Daun Hoya. Ukuran gambar terlalu kecil, harap gunakan foto yang lebih besar.',
                'error_type': 'noise'
            })

        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 3.0:
            return jsonify({
                'error': 'Gambar Terdeteksi Bukan Daun Hoya. Foto terlalu buram atau tidak fokus, harap foto ulang daun dengan lebih jelas.',
                'error_type': 'noise'
            })

        input_tensor = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            guard_logit = guard_model(input_tensor)
            is_hoya_prob = torch.sigmoid(guard_logit).item()
            print(f"Mini-CNN Guard (MobileNetV3) Hoya probability: {is_hoya_prob*100:.2f}%", flush=True)

            if is_hoya_prob < 0.50:
                return jsonify({
                    'error': 'Gambar Terdeteksi Bukan Daun Hoya. Sistem mendeteksi objek ini bukan daun Hoya (tanaman/objek lain). Harap pastikan foto memperlihatkan daun Hoya.',
                    'error_type': 'noise'
                })

            # 1. ResNet50 + CBAM Inference
            _, disease_out_res, species_out_res = model_resnet(input_tensor)
            conf_d_res_all = F.softmax(disease_out_res, 1)[0]
            conf_s_res_all = F.softmax(species_out_res, 1)[0]
            conf_d_res, pred_d_res = torch.max(conf_d_res_all, 0)
            conf_s_res, pred_s_res = torch.max(conf_s_res_all, 0)

            disease_res = class_names[pred_d_res.item()]
            species_res = species_names[pred_s_res.item()]
            conf_d_res_pct = round(conf_d_res.item() * 100, 1)
            conf_s_res_pct = round(conf_s_res.item() * 100, 1)

            top3_d_res = torch.topk(conf_d_res_all, 3)
            top3_diseases_res = [{'nama': class_names[i], 'confidence': round(v.item()*100, 2)} for v, i in zip(top3_d_res.values, top3_d_res.indices)]

        # Generate all heatmaps
        letterboxed_img = LetterboxPad(target_size=(IMG_SIZE, IMG_SIZE))(image)
        img_np = np.array(letterboxed_img, dtype=np.float32) / 255.0
        targets_res = [ClassifierOutputTarget(pred_d_res.item())]

        # Calculate letterbox crop region (to remove black bars from output)
        orig_w, orig_h = image.size
        scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (IMG_SIZE - new_w) // 2
        pad_y = (IMG_SIZE - new_h) // 2
        crop_box = (pad_x, pad_y, pad_x + new_w, pad_y + new_h)

        def overlay_and_crop(cam_map, img_np_base, crop_region):
            """Generate heatmap overlay and crop out letterbox padding."""
            overlay = show_cam_on_image(img_np_base, cam_map, use_rgb=True)
            cropped = Image.fromarray(overlay).crop(crop_region)
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # Generate ResNet50 Heatmaps (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM)
        heatmaps_resnet = {}
        method_labels = {
            'gradcam': 'Grad-CAM',
            'gradcam_pp': 'Grad-CAM++',
            'layercam': 'Layer-CAM',
            'eigencam': 'Eigen-CAM'
        }
        for method_key, cam_obj in cam_methods_resnet.items():
            raw = cam_obj(input_tensor=input_tensor, targets=targets_res)[0]
            heatmaps_resnet[method_key] = overlay_and_crop(raw, img_np, crop_box)

        # Main ResNet heatmap is Grad-CAM++
        cam_res_b64 = heatmaps_resnet['gradcam_pp']

        # Original image (also cropped, no padding)
        orig_cropped = letterboxed_img.crop(crop_box)
        original_buffered = io.BytesIO()
        orig_cropped.save(original_buffered, format="JPEG")
        orig_b64 = base64.b64encode(original_buffered.getvalue()).decode('utf-8')

        knowledge_data = []
        target_disease_name = disease_res
        if neo4j_driver and target_disease_name.lower() != 'sehat':
            try:
                session_kwargs = {'database': NEO4J_DATABASE} if NEO4J_DATABASE else {}
                with neo4j_driver.session(**session_kwargs) as session:
                    # Alur Query Ontologi Konsisten (Sesuai Revisi Dosen):
                    # Predicted Symptom -> Symptom Node -> Disease/Pest -> CausalFactor/Treatment
                    query = """
                    MATCH (s:Symptom)
                    WHERE s.category = $category OR s.name_id CONTAINS $category
                    MATCH (n)-[:HAS_SYMPTOM]->(s)
                    WHERE n:Disease OR n:Pest
                    OPTIONAL MATCH (n)-[:FAVORED_BY]->(c:CausalFactor)
                    OPTIONAL MATCH (n)-[:TREATED_WITH]->(t:Treatment)
                    RETURN labels(n)[0] AS type, n.name_id AS d_id, n.name_en AS d_en,
                           collect(DISTINCT {id: s.name_id, en: s.name_en}) AS symptoms,
                           collect(DISTINCT {id: c.name_id, en: c.name_en}) AS causes,
                           collect(DISTINCT {id: t.name_id, en: t.name_en}) AS treatments
                    """
                    result_records = session.run(query, category=target_disease_name)
                    for record in result_records:
                        symptoms = [s for s in record["symptoms"] if s.get("id")]
                        causes = [c for c in record["causes"] if c.get("id")]
                        treatments = [t for t in record["treatments"] if t.get("id")]

                        knowledge_data.append({
                            "type": record["type"],
                            "disease_id": record["d_id"],
                            "disease_en": record["d_en"],
                            "symptoms": symptoms,
                            "causes": causes,
                            "treatments": treatments
                        })
            except Exception as e:
                print("Neo4j query error:", e)

        result = {
            'success': True,
            'disease': {
                'name': disease_res,
                'confidence': conf_d_res_pct,
                'top3': top3_diseases_res,
                'status': 'sehat' if disease_res.lower() == 'sehat' else 'sakit'
            },
            'species': {
                'name': species_res,
                'confidence': conf_s_res_pct
            },
            'images': {
                'original': f"data:image/jpeg;base64,{orig_b64}",
                'gradcam': f"data:image/jpeg;base64,{cam_res_b64}"
            },
            'models': {
                'resnet50': {
                    'name': 'ResNet50 + CBAM',
                    'disease': {
                        'name': disease_res,
                        'confidence': conf_d_res_pct,
                        'top3': top3_diseases_res,
                        'status': 'sehat' if disease_res.lower() == 'sehat' else 'sakit'
                    },
                    'species': {
                        'name': species_res,
                        'confidence': conf_s_res_pct
                    },
                    'gradcam': f"data:image/jpeg;base64,{cam_res_b64}"
                }
            },
            'heatmap_methods': {k: {'label': method_labels[k], 'image': f"data:image/jpeg;base64,{v}"} for k, v in heatmaps_resnet.items()},
            'knowledge': knowledge_data
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"Error in /predict: {e}", flush=True)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask App on port 5000...", flush=True)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

