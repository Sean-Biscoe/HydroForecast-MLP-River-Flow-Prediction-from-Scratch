Here is a clean, professional `README.md` file tailored for your project. I’ve structured it to reflect both the data preprocessing pipeline and the custom Multi-Layer Perceptron (MLP) implementation.

---

# River Flow & Rainfall Prediction Pipeline

This project provides a comprehensive end-to-end pipeline for hydrological data analysis. It includes advanced data cleaning, outlier removal, missing value interpolation, and a custom **Multi-Layer Perceptron (MLP)** built from scratch using NumPy for predictive modeling.

## ## Project Overview

The objective is to predict river flow at a specific station (**Skelton**) using historical flow data from upstream stations (**Crakehill, Skip Bridge, Westwick**) and regional rainfall data.

## ## Key Features

### **1. Robust Data Cleaning (`cleaning.py`)**

* **Dual-Stage Outlier Removal**:
* **Standard Deviation Filter**: Identifies and removes values beyond  from the mean.
* **Interquartile Range (IQR)**: Specifically targets local anomalies using  bounds.


* **Data Normalization**: Handles "garbage" values (e.g., -999 or non-numeric strings) by converting them to NaNs for systematic processing.

### **2. Preprocessing & Feature Engineering**

* **PCHIP Interpolation**: Uses Piecewise Cubic Hermite Interpolating Polynomials to fill missing data while preserving the shape and monotonicity of river flow curves.
* **Rolling Averages**: Smooths high-frequency noise using a configurable window size.
* **Time-Lagging**: Automatically generates lagged features (up to 7 days) for rainfall and river flow to capture environmental delay.
* **Correlation Analysis**: Includes automated Seaborn heatmaps to identify the strongest predictors for the MLP model.

### **3. Custom MLP Implementation**

A flexible, from-scratch Neural Network implementation supporting:

* **Multiple Activations**: Sigmoid, Tanh, and ReLU.
* **Loss Functions**: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
* **Optimization Techniques**:
* **Momentum**: Accelerates gradients and smooths updates.
* **Bold Driver**: An adaptive learning rate scheduler that increases/decreases  based on performance.
* **Weight Decay ( Regularization)**: Prevents overfitting by penalizing large weights.
* **Batch Learning**: Support for mini-batch updates to improve convergence speed.



---

## ## File Structure

| File | Purpose |
| --- | --- |
| `cleaning.py` | Contains the `remove_outliers` function and core logic. |
| `main.py` | The primary script for data loading, visualization, and MLP training. |
| `Mean water.xlsx` | **(Input)** Raw river and rainfall data. |
| `Cleaned_Data.txt` | **(Output)** Fully processed dataset ready for training. |

---

## ## Getting Started

### **Prerequisites**

```bash
pip install numpy pandas matplotlib seaborn scipy openpyxl

```

### **Usage**

1. **Prepare Data**: Place your `Mean water.xlsx` file in the root directory.
2. **Run Pipeline**:
```python
python main.py

```


3. **Analyze Results**: The script will display "Before vs After" scatter plots for data cleaning and a correlation heatmap before saving the final `Lagged_Rainfall_Data.txt`.

---

## ## MLP Training Options

You can choose from several training methods depending on your requirements:

```python
# Example: Training with all improvements
mlp = MLP(input_size=10, hidden_size=20, output_size=1, learning_rate=0.1, epochs=1000)
history = mlp.train_with_bold_driver_momentum_weightdecay(X_train, y_train)

```

---

## ## Math & Logic

The network uses backpropagation to minimize error. For the Weight Decay implementation, the cost function is modified as:


The **Bold Driver** logic follows:

* If : 
* Else: 

Would you like me to help you write the code to split your data into training and testing sets for the MLP?
