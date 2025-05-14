import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

# ============ Argument Parser ============
parser = argparse.ArgumentParser(description="Spatial Transformer for Skew Correction")
parser.add_argument("--filename", type=str, required=True, help="Path to input image file")
args = parser.parse_args()

# ============ Load and Preprocess Image ============
transform = transforms.Compose([
    transforms.Grayscale(),              # Convert to grayscale
    transforms.Resize((100, 100)),      # Resize for simplicity
    transforms.ToTensor()               # Convert to tensor [C, H, W]
])

if not os.path.exists(args.filename):
    raise FileNotFoundError(f"Image not found: {args.filename}")

image = Image.open(args.filename)
input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 100, 100]

# ============ Define Spatial Transformer Network ============
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 2x3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 21 * 21, 32),  # Use actual flattened size here
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize to identity affine transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)  # Flatten dynamically
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = nn.functional.grid_sample(x, grid, align_corners=False)
        return x_transformed, theta

# ============ Apply STN ============
model = STN()
model.eval()

with torch.no_grad():
    corrected_image, theta = model(input_tensor)

# ============ Save Corrected Image ============
corrected_pil = transforms.ToPILImage()(corrected_image.squeeze(0))
basename = os.path.basename(args.filename)
corrected_filename = f"corrected_{basename}"
corrected_pil.save(corrected_filename)
print(f" Corrected image saved to {corrected_filename}")

# ============ Optional Display ============
def show_tensor_image(tensor_img, title):
    img = tensor_img.squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
show_tensor_image(input_tensor, "Original Skewed")
plt.subplot(1, 2, 2)
show_tensor_image(corrected_image, "Corrected by STN")
plt.show()

# ============ Print Affine Matrix ============
print("\nEstimated affine transformation matrix (theta):")
print(theta.squeeze().numpy())
