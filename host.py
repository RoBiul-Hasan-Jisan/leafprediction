from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import os


# CONFIG

LEAF_MODEL_PATH = "models/leaf_detector.pth"
DISEASE_MODEL_PATH = "models/best_cpu_model.pth"
SPECIFIC_DISEASE_MODEL_PATH = "models/disease_stage2_best_model.pth"

DEVICE = torch.device("cpu")
LEAF_CLASS_NAMES = ["Leaf", "Not Leaf"]
DISEASE_CLASS_NAMES = ["Dry", "Healthy", "Unhealthy"]

SPECIFIC_DISEASE_CLASSES = [
    "Anthracnose","Anthrax_Leaf","Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Bacterial_Blight","Bacterial_Canker","Bituminous_Leaf","Black_Spot","Cherry_including_sour___Powdery_mildew",
    "Curl_Leaf","Curl_Virus","Cutting_Weevil","Deficiency_Leaf","Die_Back",
    "Entomosporium_Leaf_Spot_on_woody_ornamentals","Felt_Leaf","Fungal_Leaf_Spot","Gall_Midge","Leaf_Blight",
    "Leaf_Gall","Leaf_Holes","Leaf_blight_Litchi_leaf_diseases","Litchi_algal_spot_in_non-direct_sunlight",
    "Litchi_anthracnose_on_cloudy_day","Litchi_leaf_mites_in_direct_sunlight","Litchi_mayetiola_after_raining",
    "Pepper__bell___Bacterial_spot","Potato___Early_blight","Potato___Late_blight","Powdery_Mildew",
    "Sooty_Mould","Spider_Mites","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot","Tomato___Tomato_mosaic_virus"
]

THRESHOLD_LEAF = 0.8
THRESHOLD_UNHEALTHY = 0.6


# LOAD MODELS

def load_leaf_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LEAF_CLASS_NAMES))
    model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(DISEASE_CLASS_NAMES)))
    model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_specific_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(SPECIFIC_DISEASE_CLASSES)))
    model.load_state_dict(torch.load(SPECIFIC_DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

leaf_model = load_leaf_model()
disease_model = load_disease_model()
specific_disease_model = load_specific_disease_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# FLASK APP

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    response = {"leaf": None, "health": None, "health_conf": None,
                "specific_disease": [], "disease_conf": []}

    # Step 1: Leaf
    with torch.no_grad():
        out_leaf = leaf_model(img_tensor)
        probs = F.softmax(out_leaf, dim=1)[0]
        max_prob, pred_idx = torch.max(probs,0)

    if LEAF_CLASS_NAMES[pred_idx.item()] == "Not Leaf" and max_prob.item() > THRESHOLD_LEAF:
        response["leaf"] = False
        return jsonify(response)

    response["leaf"] = True

    # Step 2: Health
    with torch.no_grad():
        out_health = disease_model(img_tensor)
        probs2 = F.softmax(out_health, dim=1)[0]
        max_prob2, pred_idx2 = torch.max(probs2,0)

    pred_class = DISEASE_CLASS_NAMES[pred_idx2.item()]
    response["health"] = pred_class
    response["health_conf"] = float(max_prob2.item())

    # Step 3: Specific Disease (only top 1)
    if pred_class == "Unhealthy" and max_prob2.item() > THRESHOLD_UNHEALTHY:
        with torch.no_grad():
            out_spec = specific_disease_model(img_tensor)
            probs3 = F.softmax(out_spec, dim=1)[0]
            top_prob, top_idx = torch.max(probs3, 0)
            response["specific_disease"].append(SPECIFIC_DISEASE_CLASSES[top_idx.item()])
            response["disease_conf"].append(float(top_prob.item()))

    return jsonify(response)


# RUN FLASK 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

