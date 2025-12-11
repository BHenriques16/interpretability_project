import numpy as np

def normalize_map(m):
    """Normaliza um mapa para o intervalo [0, 1]."""
    if m.max() - m.min() == 0:
        return np.zeros_like(m)
    return (m - m.min()) / (m.max() - m.min())

def preprocess_explanation(explanation):
    """Garante que a explicação é um mapa 2D (H, W)."""
    # Se for tensor numpy (1, 3, 128, 128) ou (3, 128, 128)
    if explanation.ndim == 4: 
        explanation = explanation[0]
    if explanation.ndim == 3: 
        # Faz a média dos canais de cor para obter intensidade
        explanation = np.mean(np.abs(explanation), axis=0)
    return explanation

def completeness(explanation, ground_truth_region, threshold=0.2):
    """
    Calcula a 'Completeness' (neste contexto, similar a Coverage/Sensitivity).
    Threshold ajustado para 0.2 para capturar gradientes mais suaves.
    """
    explanation = preprocess_explanation(explanation)
    explanation = normalize_map(explanation)
    
    # Binariza a explicação
    explanation_binary = (explanation > threshold).astype(float)
    
    # Se ground_truth_region for imagem colorida, converte para intensidade
    if ground_truth_region.ndim == 3:
        ground_truth_region = np.mean(np.abs(ground_truth_region), axis=0)
    
    # Normaliza a região de interesse
    if ground_truth_region.max() > 1.0:
        ground_truth_region = normalize_map(ground_truth_region)
        
    total_region = np.sum(ground_truth_region)
    
    if total_region == 0:
        return 0.0
    
    # Calcula quanto da intensidade da imagem original está coberta pela explicação
    completeness_score = np.sum(explanation_binary * ground_truth_region) / total_region
    return completeness_score

def attribution_localization(explanation, ground_truth_mask, threshold=0.2):
    """
    Calcula IoU. Retorna 0.0 se não houver máscara espacial compatível.
    """
    explanation = preprocess_explanation(explanation)
    explanation = normalize_map(explanation)
    
    # Verificação de segurança: Se ground_truth for vetor (ex: labels do CelebA), aborta.
    if ground_truth_mask.ndim < 2:
        return 0.0

    # Ajustes de dimensão do Ground Truth
    if ground_truth_mask.ndim > 2:
        ground_truth_mask = ground_truth_mask.squeeze()
        if ground_truth_mask.ndim > 2: 
             ground_truth_mask = np.mean(ground_truth_mask, axis=0)

    # Se tamanhos não baterem, aborta
    if explanation.shape != ground_truth_mask.shape:
        return 0.0
    
    # Cálculo do IoU
    explanation_binary = (explanation > threshold).astype(bool)
    gt_binary = (ground_truth_mask > 0.5).astype(bool) 
    
    intersection = np.logical_and(explanation_binary, gt_binary).sum()
    union = np.logical_or(explanation_binary, gt_binary).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)