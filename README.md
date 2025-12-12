# Avaliação de Interpretabilidade em Modelos de Classificação Facial

Este projeto implementa e compara diferentes métodos de **Interpretabilidade (XAI)** aplicados a um modelo de classificação de atributos faciais (ResNet-18) treinado no dataset **CelebA**.

O principal diferencial deste projeto é o pipeline de **validação quantitativa robusta**, que utiliza máscaras de segmentação anatómicas reais (do dataset **CelebAMask-HQ**) para calcular a precisão espacial das explicações (Attribution Localization / IoU), em vez de utilizar bounding boxes manuais ou aproximações.

## Objetivos

1.  Treinar/Utilizar um modelo de Deep Learning para classificar 40 atributos faciais (ex: *Smiling*, *Young*, *Wearing Lipstick*).
2.  Aplicar métodos de explicação pós-hoc para entender "onde" o modelo olha.
3.  Validar quantitativamente se as explicações coincidem com a anatomia facial real usando **Ground Truth Dinâmico**.

## Metodologia

### 1. Modelo e Dados
* **Modelo:** ResNet-18 (Pretrained on ImageNet -> Fine-tuned on CelebA).
* **Treino:** Dataset CelebA (apenas labels binárias, sem informação de localização).
* **Validação:** Subconjunto externo do **CelebAMask-HQ** (Imagens de alta resolução + Máscaras de Segmentação).

### 2. Métodos de Interpretabilidade Comparados
* **LIME:** Perturbação baseada em superpixéis.
* **Occlusion:** Perturbação baseada em janela deslizante.
* **Grad-CAM:** Ativação baseada em gradientes na última camada convolucional.
* **Integrated Gradients:** Método baseado em axiomas de gradiente.
* **Saliency Maps:** Gradiente simples em relação à entrada.

### 3. Validação com Máscaras Dinâmicas (Inovação)
Para calcular métricas justas, o sistema seleciona automaticamente a máscara de segmentação correta baseada na predição do modelo:
* Se a predição for **"Wearing Lipstick"** → O sistema carrega e funde as máscaras `l_lip` e `u_lip`.
* Se a predição for **"Black Hair"** → O sistema carrega a máscara `hair`.
* Se a predição for **"Young"** → O sistema gera uma máscara facial completa (pele + olhos + nariz + boca).

Isto permite validar a **Localização Fracamente Supervisionada** (Weakly Supervised Localization).

## Estrutura do Projeto

```text
tp_interpretabilidade/
│
├── main.py               # Script principal (Carrega modelo, gera explicações e métricas)
├── model.py              # Definição da arquitetura ResNet-18
├── methods.py            # Implementação dos algoritmos XAI (LIME, Grad-CAM, etc.)
├── metrics.py            # Funções de avaliação (IoU, Completeness)
├── data_transform.py     # Pipelines de pré-processamento
│
├── models/
│   └── best_model.pth    # Pesos do modelo treinado
│
├── validation_data/      # Dataset de Validação (CelebAMask-HQ)
│   ├── CelebA-HQ-img/    # Imagens originais (.jpg)
│   └── CelebAMask-HQ-mask-anno/  # Máscaras segmentadas por partes
│
├── images/               # Outputs visuais gerados
└── requirements.txt      # Dependências do projeto