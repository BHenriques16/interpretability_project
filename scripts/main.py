import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from methods import *
from metrics import *
from model import PretrainedModel
from data_transform import create_data_transforms
import cv2
from PIL import Image
import os
import glob

# Configurations
IMG_SIZE = 128
BASE_DIR = "scripts/validation_data" 
IMG_DIR = os.path.join(BASE_DIR, "CelebA-HQ-img")
MASK_DIR = os.path.join(BASE_DIR, "CelebAMask-HQ-mask-anno")

# Mapping based on CelebA attributes to relevant CelebAMask parts
MASK_MAPPING = {
    # Mouth related attributes
    21: ['l_lip', 'u_lip', 'mouth'], # Mouth_Slightly_Open
    31: ['l_lip', 'u_lip', 'mouth'], # Smiling
    36: ['l_lip', 'u_lip'],          # Wearing_Lipstick
    
    # Eyes/Goggles related attributes
    1:  ['l_brow', 'r_brow'],        # Arched_Eyebrows
    15: ['eye_g'],                   # Eyeglasses
    23: ['l_eye', 'r_eye'],          # Narrow_Eyes
    
    # Hair related attributes
    8:  ['hair'], # Black_Hair
    9:  ['hair'], # Blond_Hair
    11: ['hair'], # Brown_Hair
    17: ['hair'], # Gray_Hair
    33: ['hair'], # Wavy_Hair
    
    # Nose related attributes
    7:  ['nose'], # Big_Nose
    27: ['nose'], # Pointy_Nose
    
    # Skin and general facial features
    'default': ['skin', 'nose', 'l_eye', 'r_eye', 'l_lip', 'u_lip', 'l_brow', 'r_brow'] 
}

# Model loading and Utilities
def load_model(model_path='models/best_model.pth', num_classes=40):
    model = PretrainedModel(num_classes=num_classes, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def remove_inplace_relu(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU): module.inplace = False

def get_mask_filepath(base_mask_dir, image_id_int, part_name):
    folder_num = image_id_int // 2000
    filename_str = f"{image_id_int:05d}_{part_name}.png"
    return os.path.join(base_mask_dir, str(folder_num), filename_str)

def construct_mask_from_parts(base_mask_dir, image_id_int, class_idx, image_shape=(128, 128)):
    parts_needed = MASK_MAPPING.get(class_idx, MASK_MAPPING['default'])
    final_mask = np.zeros(image_shape)
    found_any = False
    
    for part in parts_needed:
        mask_path = get_mask_filepath(base_mask_dir, image_id_int, part)
        
        if os.path.exists(mask_path):
            try:
                part_img = Image.open(mask_path).convert('L')
                part_img = part_img.resize(image_shape)
                part_np = np.array(part_img)
                final_mask = np.maximum(final_mask, part_np)
                found_any = True
            except Exception as e:
                print(f"Erro ao ler {mask_path}: {e}")
    
    if not found_any:
        return np.zeros(image_shape)
        
    return (final_mask > 0).astype(float)

# Visualization Functions
def visualize_result(image_pil, mask_binary, methods_dict, filename, class_name):
    image_vis = np.array(image_pil.resize((128, 128))) / 255.0
    num_methods = len(methods_dict)
    fig, axes = plt.subplots(1, num_methods + 2, figsize=(3 * (num_methods + 2), 3))
    
    axes[0].imshow(image_vis)
    axes[0].set_title(f"ID: {filename}\nPred: {class_name}")
    axes[0].axis('off')
    
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title("CelebAMask GT")
    axes[1].axis('off')
    
    for idx, (name, exp) in enumerate(methods_dict.items()):
        ax = axes[idx + 2]
        if isinstance(exp, torch.Tensor): exp = exp.cpu().detach().numpy()
        if exp.ndim == 4: exp = exp[0]
        if exp.ndim == 3: exp = np.mean(np.abs(exp), axis=0)
        
        v_min, v_max = np.percentile(exp, [1, 99])
        exp_norm = np.clip((exp - v_min) / (v_max - v_min + 1e-8), 0, 1)
        exp_norm = cv2.resize(exp_norm.astype(np.float32), (128, 128))
        
        ax.imshow(image_vis)
        ax.imshow(exp_norm, cmap='jet', alpha=0.5)
        ax.set_title(name)
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/result_{filename}.png', dpi=100)
    plt.close()

def plot_metrics_comparison(metrics_results):
    if not metrics_results: return

    names = [item[0] for item in metrics_results]
    completeness_scores = [item[1] for item in metrics_results]
    attribution_scores = [item[2] for item in metrics_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Completeness Graphic
    bars1 = ax1.bar(names, completeness_scores, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Average Completeness', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(max(completeness_scores)*1.2, 0.1)) 
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Attribution Localization Graphic
    bars2 = ax2.bar(names, attribution_scores, color='coral', alpha=0.7)
    ax2.set_ylabel('IoU Score')
    ax2.set_title('Average Attribution Localization (IoU)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/final_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# MAIN
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, val_transforms = create_data_transforms(IMG_SIZE)
    model, device = load_model()
    remove_inplace_relu(model)
    
    # Search for images in the CelebA-HQ-img folder
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    
    # Loads only first 5 images for quick testing
    img_files = img_files[:5] 
    
    if not img_files:
        print(f"ERRO: Nenhuma imagem encontrada em {IMG_DIR}")
        return

    global_results = {k: {'comp': [], 'loc': []} for k in ['LIME', 'Integrated Gradients', 'Grad-CAM', 'Occlusion', 'Saliency']}

    print(f"Starting Evaluation on {len(img_files)} images...")

    for img_path in img_files:
        filename = os.path.basename(img_path)
        # Extract numeric ID (e.g., "0.jpg" -> 0)
        try:
            image_id_int = int(filename.split('.')[0])
        except ValueError:
            continue
        
        print(f"\nProcessing ID: {image_id_int}")
        
        # Load Image and Preprocess
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = val_transforms(pil_img).unsqueeze(0).to(device)
        
        # Predict classes
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            target_class = torch.argmax(probs, dim=1).item()
        
        print(f"  -> Predicted Class: {target_class}")
            
        # Build mask based on predicted class
        gt_mask = construct_mask_from_parts(MASK_DIR, image_id_int, target_class)
        
        if np.sum(gt_mask) == 0:
            print("  -> Skipping metrics (Empty Mask / Parts not found)")
            continue

        # Methods and metrics
        methods = {
            'LIME': lime_method(model, input_tensor),
            'Integrated Gradients': integrated_gradients_method(model, input_tensor, target_class),
            'Grad-CAM': grad_cam_method(model, input_tensor, target_class),
            'Occlusion': occlusion_method(model, input_tensor, target_class),
            'Saliency': saliency_method(model, input_tensor, target_class)
        }
        
        img_np_intensity = np.mean(input_tensor.cpu().numpy()[0], axis=0)
        
        for name, exp in methods.items():
            if isinstance(exp, torch.Tensor): exp = exp.cpu().detach().numpy()
            comp = completeness(exp, img_np_intensity, threshold=0.2)
            loc = attribution_localization(exp, gt_mask, threshold=0.2)
            global_results[name]['comp'].append(comp)
            global_results[name]['loc'].append(loc)
            
        visualize_result(pil_img, gt_mask, methods, str(image_id_int), f"Class {target_class}")

    # Final Results
    print("\n" + "="*60)
    print("FINAL RESULTS (Average over Validation Subset)")
    print(f"{'METHOD':<25} | {'AVG COMPLETENESS':<20} | {'AVG ATTR LOC (IoU)':<20}")
    print("-" * 70)
    
    final_metrics = []
    for name, m in global_results.items():
        if m['comp']:
            avg_c, avg_l = np.mean(m['comp']), np.mean(m['loc'])
            print(f"{name:<25} | {avg_c:.4f}               | {avg_l:.4f}")
            final_metrics.append((name, avg_c, avg_l))
    
    if final_metrics:
        plot_metrics_comparison(final_metrics)
    
    print("\nDone. Check 'images/' folder for visualizations.")

if __name__ == '__main__':
    main()
