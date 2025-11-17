import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from methods import *
from metrics import *
from model import PretrainedModel
from data_transform import create_data_transforms
import cv2


def load_model(model_path='models/best_model.pth', num_classes=40, pretrained=False):
    model = PretrainedModel(num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def remove_inplace_relu(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

def load_data(test_loader):
    data_iter = iter(test_loader)
    image, ground_truth = next(data_iter)
    image = image[0].unsqueeze(0)
    ground_truth = ground_truth[0].unsqueeze(0)
    return image, ground_truth

def normalize_explanation(explanation):
    exp_min = np.min(explanation)
    exp_max = np.max(explanation)
    if exp_max - exp_min == 0:
        return np.zeros_like(explanation)
    return (explanation - exp_min) / (exp_max - exp_min)


def visualize_explanations(image, ground_truth, methods_dict):    
    # Converts image for visualization
    image_vis = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min() + 1e-8)
    
    # Create the figure wihth subplots
    num_methods = len(methods_dict)
    fig, axes = plt.subplots(2, (num_methods + 1) // 2 + 1, figsize=(20, 10))
    fig.suptitle('Comparison of Interpretability Methods', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    axes[0].imshow(image_vis)
    axes[0].set_title('Imagem Original', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth as heatmap
    gt_vis = ground_truth.squeeze(0).cpu().numpy()
    gt_heatmap = np.zeros((len(gt_vis), 128))
    gt_heatmap[:, :] = gt_vis.reshape(-1, 1)
    im = axes[1].imshow(gt_heatmap, cmap='RdYlGn', aspect='auto')
    axes[1].set_title('Ground Truth ', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Atributte Index', fontsize=10)
    plt.colorbar(im, ax=axes[1], label='Present')
    
    # Visualize each method
    for idx, (name, explanation) in enumerate(methods_dict.items(), start=2):
        if idx >= len(axes):
            break
        
        try:
            # Converte to numpy if is a tensor
            if isinstance(explanation, torch.Tensor):
                explanation = explanation.cpu().numpy()
            
            if explanation.ndim >= 3:
                exp_norm = np.abs(explanation).squeeze().mean(axis=0)
            elif explanation.ndim == 2:
                exp_norm = np.abs(explanation)
            else:
                exp_norm = np.abs(explanation)
            
            if exp_norm.ndim == 1:
                size = int(np.sqrt(len(exp_norm)))
                if size * size != len(exp_norm):
                    target_size = (size + 1) ** 2
                    exp_padded = np.zeros(target_size)
                    exp_padded[:len(exp_norm)] = exp_norm.flatten()
                    exp_norm = exp_padded.reshape(size + 1, size + 1)
                else:
                    exp_norm = exp_norm.reshape(size, size)
            
            # Normalize [0,1]
            exp_min = np.min(exp_norm)
            exp_max = np.max(exp_norm)
            if exp_max - exp_min > 1e-8:
                exp_norm = (exp_norm - exp_min) / (exp_max - exp_min)
            else:
                exp_norm = np.zeros_like(exp_norm, dtype=float)
            
            exp_norm = cv2.resize(exp_norm.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR)
            
            axes[idx].imshow(image_vis)
            im = axes[idx].imshow(exp_norm, cmap='jet', alpha=0.6)
            axes[idx].set_title(name, fontweight='bold', fontsize=12)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx])
            
        except Exception as e:
            print(f"Error visualizing {name}: {str(e)}")
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            axes[idx].axis('off')
    
    # Remove eixos vazios
    for idx in range(len(methods_dict) + 2, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('images/interpretability_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(metrics_results):
    names = [item[0] for item in metrics_results]
    completeness_scores = [item[1] for item in metrics_results]
    attribution_scores = [item[2] for item in metrics_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparison of Interpretability Metrics', fontsize=14, fontweight='bold')
    
    # Gráfico de Completeness
    bars1 = ax1.bar(names, completeness_scores, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Completeness Score', fontsize=12, fontweight='bold')
    ax1.set_title('Completeness', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Gráfico de Attribution Localization
    bars2 = ax2.bar(names, attribution_scores, color='coral', alpha=0.7)
    ax2.set_ylabel('Attribution Localization Score', fontsize=12, fontweight='bold')
    ax2.set_title('Attribution Localization', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    img_size = 128
    batch_size = 16
    num_classes = 40
    model_path = "models/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Criação do DataLoader
    train_transforms, val_transforms = create_data_transforms(img_size)
    celeba_dataset = datasets.CelebA(root="./data", split="all", target_type="attr", 
                                    download=False, transform=val_transforms)
    total_size = len(celeba_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    _, _, test_data = random_split(celeba_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Carrega o modelo
    model = load_model(model_path, num_classes=num_classes, pretrained=False)
    model.to(device)
    remove_inplace_relu(model)

    # Carrega os dados
    image, ground_truth = load_data(test_loader)
    image_device = image.to(device)

    print("\nApplying interpretability methods...")
    
    # Aplica os métodos de interpretabilidade
    methods_explanations = {}
    
    print("LIME...")
    lime_explanation = lime_method(model, image)
    methods_explanations['LIME'] = lime_explanation
    
    print("Integrated Gradients...")
    integrated_gradients_explanation = integrated_gradients_method(model, image_device, 0)
    methods_explanations['Integrated Gradients'] = integrated_gradients_explanation
    
    print("Grad-CAM...")
    grad_cam_explanation = grad_cam_method(model, image_device, 0)
    methods_explanations['Grad-CAM'] = grad_cam_explanation
    
    print("Occlusion...")
    occlusion_explanation = occlusion_method(model, image_device, 0)
    methods_explanations['Occlusion'] = occlusion_explanation
    
    print("Saliency...")
    saliency_explanation = saliency_method(model, image_device, 0)
    methods_explanations['Saliency'] = saliency_explanation
    
    # Calcula as métricas para cada método
    metrics_results = []
    
    print("\nCalculating metrics...")
    print("=" * 80)
    
    for name, explanation in methods_explanations.items():
        comp = completeness(explanation, image.numpy(), threshold=0.5)
        attr_loc = attribution_localization(explanation, ground_truth.numpy(), threshold=0.5)
        metrics_results.append((name, comp, attr_loc))
        print(f'{name:25} | Completeness: {comp:.4f} | Attribution Localization: {attr_loc:.4f}')
    
    print("=" * 80)
    
    # Gera visualizações
    print("\nGenerating visualizations...")
    visualize_explanations(image, ground_truth, methods_explanations)
    plot_metrics_comparison(metrics_results)

if __name__ == '__main__':
    main()