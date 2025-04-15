# Paddy Leaf Disease Detection

## Overview

This project focuses on detecting diseases in paddy leaves using deep learning. We developed a fine-tuned **EfficientNetB4 Convolutional Neural Network (CNN)** to classify rice leaf diseases with high accuracy. The system helps identify diseases early, allowing farmers and agricultural professionals to take preventive actions and reduce crop loss.

## Project Members

- Abhigna Ragala (21WU0102051)  
- Abhijeeth Ragala (21WU0102050)  
- Nihitha Vadlamuri (21WU0102049)  
- Prasanth Damma (21WU0102047)  

## University

**Woxsen University, Hyderabad**  
Bachelor of Technology in DSAI  
April 2025

## Abstract

Rice, being a staple food for a large global population, is highly vulnerable to leaf diseases that can severely impact crop yield and food security. This project proposes an automated detection system leveraging EfficientNetB4 CNN architecture. The model outperformed other popular architectures like MobileNetV3, DenseNet201, InceptionNetV3, and XceptionNet in terms of accuracy, precision, recall, and F1-score.

## Dataset

The dataset contains 5932 images categorized into:
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

The dataset is split into:
- 60% Training
- 20% Validation
- 20% Testing

## Technologies Used

- Python 3
- TensorFlow & Keras
- OpenCV, PIL
- Google Colab
- Matplotlib, Seaborn

## Model Architecture

- **Base Model**: EfficientNetB4 (pretrained on ImageNet)
- **Modifications**:
  - Global Average Pooling
  - Dense layers with ReLU
  - Softmax output layer

## Performance Metrics

| Model | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| EfficientNetB4 | 0.9974 | 0.9975 | 0.9974 |
| MobileNetV3 | 0.9777 | 0.9756 | 0.9764 |
| DenseNet201 | 0.9498 | 0.9523 | 0.9491 |
| XceptionNet | 0.8906 | 0.8707 | 0.8736 |
| InceptionNetV3 | 0.8201 | 0.8147 | 0.8736 |

## Features

- Robust image preprocessing: resizing, augmentation
- Accurate multi-class classification
- Visualizations: confusion matrix, accuracy/loss plots
- Real-time deployment-ready model

## Future Work

- Expand dataset for better generalization
- Add attention mechanisms or ensemble models
- Build mobile or drone-based real-time detection tools
- Include disease prediction based on environmental data

## References

Refer to the [Final Capstone Report](./Final%20capstone%20Report%20final.pdf) for a detailed explanation of the project methodology, results, and citations.

## License

This project is intended for academic and research purposes.
