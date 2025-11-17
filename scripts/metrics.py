import numpy as np

def completeness(explanation, prediction, threshold=0.5):
    # if the prediction is a atribuiton vector (multi-label)
    if prediction.ndim == 2 and prediction.shape[1] > 1 and prediction.shape[0] == 1:
        # Uses the mean of the atributtes as prediction
        prediction = np.mean(prediction, axis=1, keepdims=True)
    
    # Normalizes the explanation
    if explanation.ndim > 2:
        explanation = np.abs(explanation).squeeze(0).mean(axis=0)
    
    explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation) + 1e-8)
    
    explanation_binary = (explanation > threshold).astype(float)
    
    if np.sum(prediction) == 0:
        return 0.0
    
    completeness_score = np.sum(explanation_binary * prediction) / np.sum(prediction)
    return completeness_score


def attribution_localization(explanation, ground_truth, threshold=0.5):
    # if the ground truth is a atribuiton vector (multi-label)
    if ground_truth.ndim == 2 and ground_truth.shape[1] > 1 and ground_truth.shape[0] == 1:
        # Uses the mean of the atributtes as ground truth
        ground_truth = np.mean(ground_truth, axis=1, keepdims=True)
    
    # Normalizes the explanation
    if explanation.ndim > 2:
        explanation = np.abs(explanation).squeeze(0).mean(axis=0)
    
    explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation) + 1e-8)
    
    explanation_binary = (explanation > threshold).astype(float)
    
    intersection = np.sum(explanation_binary * ground_truth)
    union = np.sum(explanation_binary) + np.sum(ground_truth) - intersection
    
    if union == 0:
        return 0.0
    
    attribution_localization_score = intersection / union
    return attribution_localization_score