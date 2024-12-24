import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from U2Net.model.u2net import U2NET

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    return model

def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
    return (predicted_map - mi) / (ma - mi)

def apply_mask(file_path, model_path="U2Net/saved_models/u2net/u2net.pth", device="cpu"):
    model = U2NET(in_ch=3, out_ch=1)
    model = load_model(model, model_path, device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    resize_shape = (320, 320)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    image = Image.open(file_path).convert("RGB")
    image_resized = image.resize(resize_shape, resample=Image.BILINEAR)
    image_tensor = transforms(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    prediction_map = torch.squeeze(prediction[0].cpu(), dim=(0, 1)).numpy()

    normalized_prediction = normPRED(prediction_map)

    return normalized_prediction

if __name__ == "__main__":
    prediction = apply_mask("cropped.png")
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    plt.show()
