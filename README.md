# CNN Multi-Class Image Classification (Airplane, Ship, Truck)

This project implements a **Convolutional Neural Network (CNN)** designed to classify images into three categories: Airplanes, Ships, and Trucks. The project explores various architectural modifications and hyperparameter tuning to achieve high generalization accuracy.

## 🚀 Highlights & Performance
* **Best Accuracy:** Achieved **86.00%** test accuracy using an Ensemble & Refinement strategy.
* **Core Architecture:** 3-layer CNN with Batch Normalization, ReLU activation, and Max-Pooling.
* **Optimization:** Experimented with AdamW optimizer, Cosine Annealing scheduler, and Label Smoothing.

## 🧪 Experiments Conducted
The project includes a detailed analysis of:
1.  **Hyperparameter Tuning:** Impact of Batch Size, Learning Rate, and Early Stopping.
2.  **Regularization:** Evaluating Dropout effects at different layers (Low-level vs. High-level).
3.  **Filter Configurations:** Comparing different widths (e.g., Inverted Pyramid 64-32-16).
4.  **Pooling Strategies:** Analyzing Max vs. Average vs. L2 Pooling performance.
5.  **Activation Functions:** Comparative study between ReLU and Tanh.
6.  **Model Complexity:** Impact of adding Fully Connected hidden layers.

## 📊 Feature Visualization
Includes visualization of feature maps across different layers to understand how the network detects edges (Low-level), shapes (Mid-level), and semantic parts (High-level).

## 🛠️ Tech Stack
* Python
* PyTorch
* Matplotlib (for visualization)
