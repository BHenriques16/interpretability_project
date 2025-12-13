# Interpretability Evaluation in Face Classification Models

This project implements and compares different **Interpretability AI ** methods applied to a facial attribute classification model (ResNet-18) trained on the **CelebA** dataset.

The key differentiator of this project is its **robust quantitative validation** pipeline. Instead of using manual bounding boxes or approximations, it employs real anatomical segmentation masks (from the **CelebAMask-HQ** dataset) to calculate the spatial precision of explanations (Attribution Localization / IoU).

## Objectives

1.  Train/Use a Deep Learning model to classify 40 facial attributes (e.g., *Smiling*, *Young*, *Wearing Lipstick*).
2.  Apply post-hoc explanation methods to understand "where" the model focuses its attention.
3.  Quantitatively validate if the explanations align with real facial anatomy using a **Dynamic Ground Truth** system.

## Methodology

### 1. Model and Data
* **Model:** ResNet-18 (Pretrained on ImageNet -> Fine-tuned on CelebA).
* **Training:** CelebA dataset (using only binary labels, without any localization information).
* **Validation:** An external subset of **CelebAMask-HQ** (High-resolution images + Segmentation Masks).

### 2. Compared XAI Methods
* **LIME:** Perturbation based on superpixels.
* **Occlusion:** Perturbation based on a sliding window.
* **Grad-CAM:** Activation based on gradients in the final convolutional layer.
* **Integrated Gradients:** Axiomatic attribution method based on path integrals.
* **Saliency Maps:** Simple gradient regarding the input.

### 3. Validation with Dynamic Masks (Innovation)
To calculate fair metrics, the system automatically selects the semantic segmentation mask corresponding to the model's specific prediction:
* If prediction is **"Wearing Lipstick"** → System loads and merges `l_lip` and `u_lip` masks.
* If prediction is **"Black Hair"** → System loads the `hair` mask.
* If prediction is **"Young"** → System generates a full facial mask (skin + eyes + nose + mouth).

This approach successfully validates **Weakly Supervised Localization** capabilities.

## Project Structure

```text
tp_interpretabilidade/
│
├── main.py               # Main script (Loads model, runs XAI, calcs metrics)
├── model.py              # ResNet-18 architecture definition
├── methods.py            # Implementation of Interpretability algorithms (LIME, Grad-CAM, etc.)
├── metrics.py            # Evaluation functions (IoU, Completeness)
├── data_transform.py     # Preprocessing pipelines
│
├── models/
│   └── best_model.pth    # Trained model weights
│
├── validation_data/      # Validation Dataset (CelebAMask-HQ)
│   ├── CelebA-HQ-img/    # Original images (.jpg)
│   └── CelebAMask-HQ-mask-anno/  # Segmented mask parts
│
├── images/               # Generated visual outputs

└── requirements.txt      # Project dependencies
