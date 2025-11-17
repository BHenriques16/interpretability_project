import numpy as np

# Calulus of metrics for completeness
def completeness(explanation, prediction, threshold=0.5):
    # Normalizes the explanation
    explanation = explanation / np.max(explanation)
    # Binarizes the explanation with the threshold
    explanation_binary = (explanation > threshold).astype(float)
    # Calculates the sum of importances
    completeness_score = np.sum(explanation_binary * prediction) / np.sum(prediction)
    return completeness_score


# Calulus of metrics for attribution localization
def attribution_localization(explanation, ground_truth, threshold=0.5):
    # Normalizes the explanation
    explanation = explanation / np.max(explanation)
    # Binarizes the explanation with the threshold
    explanation_binary = (explanation > threshold).astype(float)
    # Calculates the intersection between explanation and ground truth
    intersection = np.sum(explanation_binary * ground_truth)
    # Calculates the union between explanation and ground truth
    union = np.sum(explanation_binary) + np.sum(ground_truth) - intersection
    # Calculates the Attribution Localization metric
    if union == 0:
        return 0.0
    attribution_localization_score = intersection / union
    return attribution_localization_score