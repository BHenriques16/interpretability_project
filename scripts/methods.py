import numpy as np
import torch
from lime import lime_image
from captum.attr import IntegratedGradients, Saliency, GuidedGradCam, Occlusion

def lime_method(model, image, num_samples=1000):
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(x_tensor)
        return torch.sigmoid(outputs).cpu().numpy()
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np.astype('double'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=num_samples
    )

    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    
    heatmap = np.nan_to_num(heatmap)
    
    return heatmap

def integrated_gradients_method(model, image, target_class):
    ig = IntegratedGradients(model)
    attributions = ig.attribute(image, target=target_class)
    return attributions.detach().cpu().numpy()

def grad_cam_method(model, image, target_class):
    guided_gc = GuidedGradCam(model, model.model.layer4)
    attributions = guided_gc.attribute(image, target=target_class)
    return attributions.detach().cpu().numpy()

def occlusion_method(model, image, target_class):
    occlusion = Occlusion(model)
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
