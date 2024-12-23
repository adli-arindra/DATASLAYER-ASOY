import os
from PIL import Image
import numpy as np
from torchinfo import summary
import torch
import torchvision.transforms as T
from U2Net.model.u2net import U2NET, U2NETP
import torchvision.transforms.functional as F

u2net = U2NET(in_ch=3,out_ch=1)

def load_model(model, model_path, device):
     
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
 
    return model

u2net = load_model(model=u2net, model_path="U2Net/saved_models/u2net/u2net.pth", device="cpu")
	
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
 
resize_shape = (320,320)
 
transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

def prepare_image_batch(image_dir, resize, transforms, device):
 
    image_batch = []
 
    for image_file in os.listdir(image_dir):
        image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
        image_resize = image.resize(resize, resample = Image.BILINEAR)
     
        image_trans = transforms(image_resize)
        image_batch.append(image_trans)
     
     
    image_batch = torch.stack(image_batch).to(device)
 
    return image_batch
 
image_batch = prepare_image_batch(image_dir="coba",
                                 resize=resize_shape,
                                 transforms=transforms,
                                 device="cpu")

def denorm_image(image):
    image_denorm = torch.addcmul(mean[:,None,None], image, std[:,None, None])
    image = torch.clamp(image_denorm*255., min=0., max=255.)
    image = torch.permute(image, dims=(1,2,0)).numpy().astype("uint8")
     
    return image

def prepare_predictions(model, image_batch):
 
    model.eval()
     
    all_results = []
 
    for image in image_batch:
        with torch.no_grad():
            results = model(image.unsqueeze(dim=0))
     
        all_results.append(torch.squeeze(results[0].cpu(), dim=(0,1)).numpy())
 
    return all_results

predictions_u2net = prepare_predictions(u2net, image_batch)

def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
 
    map_normalize = (predicted_map - mi) / (ma-mi)
 
    return map_normalize

import matplotlib.pyplot as plt
import numpy as np

# Example 2D array (grayscale image)

# Display the image
plt.imshow(predictions_u2net[0], cmap='gray')  # cmap='gray' for grayscale
plt.axis('off')  # Hide axes for cleaner display
plt.show()


print(predictions_u2net[0])