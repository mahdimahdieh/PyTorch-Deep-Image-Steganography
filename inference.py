import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
import collections.abc
import io
collections.Iterable = collections.abc.Iterable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PATH_H = 'checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth'
PATH_R = 'checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth'
SECRET_PATH = 'example_pics/ILSVRC2012_val_00049018.JPEG'
COVER_PATH = 'example_pics/ILSVRC2012_val_00048985.JPEG'

hide_net = UnetGenerator(input_nc=6, output_nc=3, num_downs=7).to(device)
reveal_net = RevealNet().to(device)


hide_net.load_state_dict(torch.load(PATH_H, map_location=device), strict=False)
reveal_net.load_state_dict(torch.load(PATH_R, map_location=device), strict=False)

hide_net.eval()
reveal_net.eval()


def load_img(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    """Helper to convert a GPU tensor back to a PIL image"""
    img = tensor.squeeze().cpu().detach()
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)


# ADVANCED ATTACKS

def apply_noise(tensor, mode='gaussian', strength=0.1):
    """
    Handles generic noise addition.
    strength: Variance for Gaussian, or Probability for Salt & Pepper.
    """
    if mode == 'gaussian':
        noise = torch.randn_like(tensor) * strength
        return (tensor + noise).clamp(0, 1)

    elif mode == 'salt_pepper':
        mask = torch.rand_like(tensor)
        out = tensor.clone()
        out[mask < (strength / 2)] = 0.0  # Pepper
        out[mask > (1 - strength / 2)] = 1.0  # Salt
        return out
    return tensor


def apply_color_light(tensor, mode='brightness', factor=1.0):
    """
    Handles color and lighting adjustments.
    """
    if mode == 'brightness':
        return F.adjust_brightness(tensor, factor)
    elif mode == 'hue':
        return F.adjust_hue(tensor, factor)  # Factor is -0.5 to 0.5
    elif mode == 'saturation':
        return F.adjust_saturation(tensor, factor)
    return tensor


def apply_geometry(tensor, mode='rotate', value=0):
    """
    Handles geometric transformations.
    """
    if mode == 'rotate':
        return F.rotate(tensor, angle=value)
    elif mode == 'crop':
        # Value is the size of the blacked-out square
        out = tensor.clone()
        _, _, h, w = out.shape
        size = int(value)
        y = np.random.randint(0, h - size)
        x = np.random.randint(0, w - size)
        out[:, :, y:y + size, x:x + size] = 0.0
        return out
    elif mode == 'blur':
        # Value is kernel size
        k_size = int(value) if int(value) % 2 != 0 else int(value) + 1
        return transforms.GaussianBlur(kernel_size=k_size, sigma=1.3)(tensor)
    return tensor


def apply_format_degrade(tensor, mode='jpeg', quality=80):
    """
    Handles format-based degradations (JPEG, Bit Depth, Resizing).
    """
    if mode == 'bit_depth':
        # Quality represents number of bits (e.g., 4)
        levels = 2 ** quality
        return (tensor * levels).floor() / levels

    # Helper for PIL conversions
    def to_pil(t):
        return transforms.ToPILImage()(t.squeeze().cpu().detach().clamp(0, 1))

    def to_tensor(p):
        return transforms.ToTensor()(p).unsqueeze(0).to(tensor.device)

    pil_img = to_pil(tensor)

    if mode == 'jpeg':
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return to_tensor(Image.open(buffer))

    elif mode == 'social':
        # Simulate generic Social Media (Resize -> Low Quality JPEG -> Resize Back)
        original_size = pil_img.size
        pil_img = pil_img.resize((128, 128), Image.Resampling.BILINEAR)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)  # usually 60-70 for social
        buffer.seek(0)

        final_img = Image.open(buffer).resize(original_size, Image.Resampling.BILINEAR)
        return to_tensor(final_img)

    return tensor

# --- DEFINE ATTACK LIST ---
# --- THE DICTIONARY ---

attacks = {
    "No Attack":          lambda x: x,

    "Gaussian Noise":     lambda x: apply_noise(x, mode='gaussian', strength=0.05),
    "Salt & Pepper":      lambda x: apply_noise(x, mode='salt_pepper', strength=0.02),

    "Rotate (10°)":       lambda x: apply_geometry(x, mode='rotate', value=10),
    "Blur (k=5)":         lambda x: apply_geometry(x, mode='blur', value=5),
    "Brightness (+20%)":  lambda x: apply_color_light(x, mode='brightness', factor=1.2),
    "Hue (+0.1)":         lambda x: apply_color_light(x, mode='hue', factor=0.1),
    "Saturation (x1.5)":  lambda x: apply_color_light(x, mode='saturation', factor=1.5),

    "JPEG Compression (Q=90)":   lambda x: apply_format_degrade(x, mode='jpeg', quality=90),
    "Social Network Sim": lambda x: apply_format_degrade(x, mode='social', quality=60),
    "Bit Depth (6-bit)":  lambda x: apply_format_degrade(x, mode='bit_depth', quality=6),
    "Crop / Cutout":      lambda x: apply_geometry(x, mode='crop', value=60),
}

cover = load_img(COVER_PATH)
secret = load_img(SECRET_PATH)

print("Running attacks...")

with torch.no_grad():
    # Hide
    input_img = torch.cat([cover, secret], dim=1)
    container = hide_net(input_img)

    # Plot setup
    num_attacks = len(attacks)
    fig, axes = plt.subplots(num_attacks, 3, figsize=(12, 3.5 * num_attacks))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    for i, (name, attack_func) in enumerate(attacks.items()):
        # 1. Attack
        attacked_container = attack_func(container.clone())

        # 2. Reveal
        revealed = reveal_net(attacked_container)

        # 3. Calculate Error (Residual)
        residual = torch.abs(secret - revealed)

        # 4. Display
        # Container
        ax_cont = axes[i, 0]
        ax_cont.imshow(tensor_to_pil(attacked_container))
        ax_cont.set_title(f"Container: {name}", fontsize=10)
        ax_cont.axis('off')

        # Revealed
        ax_rev = axes[i, 1]
        ax_rev.imshow(tensor_to_pil(revealed))
        ax_rev.set_title("Revealed Secret", fontsize=10)
        ax_rev.axis('off')

        # Error Map
        ax_res = axes[i, 2]
        # We boost the error map brightness so you can see it better
        ax_res.imshow(tensor_to_pil(residual * 3), cmap='hot')
        ax_res.set_title("Error", fontsize=10)
        ax_res.axis('off')

plt.show()

