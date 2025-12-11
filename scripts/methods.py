import numpy as np
import torch
from lime import lime_image
from captum.attr import IntegratedGradients, Saliency, GuidedGradCam, Occlusion

def lime_method(model, image, num_samples=1000):
    # Transforma imagem para formato numpy (H, W, C) para o LIME
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Função de predição wrapper para o LIME
    def predict_fn(x):
        # O LIME passa um batch de imagens numpy
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Envia para o device do modelo
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(x_tensor)
        return torch.sigmoid(outputs).cpu().numpy()
    
    explainer = lime_image.LimeImageExplainer()
    # top_labels=1 foca na classe predita. num_features controla a complexidade da explicação
    explanation = explainer.explain_instance(
        image_np.astype('double'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Gera a máscara da explicação (apenas para a classe principal)
    # positive_only=True mostra apenas o que contribui positivamente
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    
    # Substitui valores None por 0 (superpixels não relevantes)
    heatmap = np.nan_to_num(heatmap)
    
    return heatmap

def integrated_gradients_method(model, image, target_class):
    ig = IntegratedGradients(model)
    # attribute retorna (Batch, Channels, H, W)
    attributions = ig.attribute(image, target=target_class)
    return attributions.detach().cpu().numpy()

def grad_cam_method(model, image, target_class):
    # Nota: layer4 é específico para ResNet. Se mudares o modelo, verifica a camada.
    guided_gc = GuidedGradCam(model, model.model.layer4)
    attributions = guided_gc.attribute(image, target=target_class)
    return attributions.detach().cpu().numpy()

def occlusion_method(model, image, target_class):
    occlusion = Occlusion(model)
    # Strides e sliding_window definem a granularidade. (3, 15, 15) é razoável para 128x128
    attributions = occlusion.attribute(
        image, 
        strides=(3, 8, 8), 
        target=target_class, 
        sliding_window_shapes=(3, 16, 16)
    )
    return attributions.detach().cpu().numpy()

def saliency_method(model, image, target_class):
    saliency = Saliency(model)
    attributions = saliency.attribute(image, target=target_class)
    return attributions.detach().cpu().numpy()