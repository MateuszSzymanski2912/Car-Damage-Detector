import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import cv2
import base64


#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
torch.serialization.add_safe_globals([models.resnet.ResNet])
model = torch.load("resnet50.pth", map_location=DEVICE, weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_gradcam(model, img_tensor):
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    last_conv_layer.register_forward_hook(forward_hook)
    last_conv_layer.register_backward_hook(backward_hook)

    # Forward
    output = torch.sigmoid(model(img_tensor))
    class_idx = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    return cam

def process_image(image_bytes, generate_heatmap=False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).item()

    label = "uszkodzony" if probs > 0.5 else "nieuszkodzony"

    heatmap_b64 = None
    if generate_heatmap:
        cam = generate_gradcam(model, img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig = cv2.resize(np.array(image), (224, 224))
        overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
        _, buffer = cv2.imencode(".jpg", overlay)
        heatmap_b64 = base64.b64encode(buffer).decode()

    return {
        "label": label,
        "probability": f"{round(probs * 100, 2)}%",
        "original_image": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(),
        "heatmap": "data:image/jpeg;base64," + heatmap_b64 if generate_heatmap else None
    }