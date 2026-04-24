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

### 📈 Performance Comparison
| Configuration | Test Accuracy | Improvement |
| :--- | :--- | :--- |
| **Baseline (Best Single Model)** | 80.00% | - |
| **Final Strategy (Ensemble + TTA)** | **86.00%** | **+6.00%** |

### 💡 Why did the performance jump?
The 6% absolute improvement wasn't just luck; it was achieved by focusing on **Model Variance Reduction**:
**Ensemble Learning**: Trained 3 independent models to "average out" stochastic errors[cite: 183, 192].
**Test-Time Augmentation (TTA)**: Aggregated predictions across horizontal flips to ensure robust inference[cite: 184].
**Optimization Refinement**: Used **AdamW** with a **Cosine Annealing Scheduler** to find a more stable minimum[cite: 185, 193].
**Label Smoothing**: Prevented the model from becoming over-confident and overfitting to label noise[cite: 186, 193].

  
## 📊 Feature Visualization
Includes visualization of feature maps across different layers to understand how the network detects edges (Low-level), shapes (Mid-level), and semantic parts (High-level).

## 🛠️ Tech Stack
* Python
* PyTorch
* Matplotlib (for visualization)
