import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from U2Net.model.u2net import U2NET


model = U2NET(in_ch=3, out_ch=1)
model.load_state_dict(torch.load("preprocessing/U2Net/saved_models/u2net/u2net.pth", map_location="cpu", weights_only=True))
model = model.to("cpu")
model.eval()


def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
    return (predicted_map - mi) / (ma - mi)

def apply_mask(image: Image.Image, model_path="preprocessing/U2Net/saved_models/u2net/u2net.pth", device="cpu"):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    resize_shape = (320, 320)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

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
