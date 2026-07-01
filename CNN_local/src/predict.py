import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations identiques à la validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Charger ResNet18 SANS téléchargement
model = models.resnet18(weights=None)

# Charger les poids ImageNet localement
state_dict = torch.load("models/resnet18-f37072fd.pth", map_location=device)
model.load_state_dict(state_dict)

# Adapter la dernière couche
model.fc = nn.Linear(model.fc.in_features, 2)

# Charger TON modèle entraîné
model.load_state_dict(torch.load("resnet18_normal_opacity_optimized.pth", map_location=device))

model = model.to(device)
model.eval()

def predict_image(image_path, threshold=0.75):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Classe incertaine
    if confidence.item() < threshold:
        return "uncertain", confidence.item()

    classes = ["normal", "opacity"]
    return classes[predicted.item()], confidence.item()

label, conf = predict_image(
"D:/Efrei/3e_annee_semestre_2/Mastercamp/Projet_MC/Assistant-Radiologue-IA/assistant-radiologue-virtuel-main/data/chexpert_eval/_cache/ashery_chexpert/train/patient00008/study1/view1_frontal.jpg"
)
print(f"Prediction: {label} (confidence: {conf:.2f})")
