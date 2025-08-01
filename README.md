# Surface Defect Detection in Hot Rolled Steel Strips

Automatic metallic surface defect inspection has received increased attention in relation to the quality control of industrial products across industries. This project aims to automatically detect metal surface defects such as rolled-in scale, patches, crazing, pitted surface, inclusion, and scratches (as depicted in the image below) in Hot-Rolled Steel Strips. The defects are classified into their specific classes via a convolutional neural network (CNN). 

<p align="center">
    <img width="400" height="400" src="https://github.com/ChavdaDhvani/steel-surface-defect-detection-system/blob/main/Surface%20Defects.png?raw=true">
</p>

A Convolutional Neural Network (CNN) model is implemented and trained on the NEU (Northeastern University) Metal Surface Defects Database, which contains 1800 grayscale images with 300 samples each of six different kinds of surface defects in hot-rolled steel strips collected from industry. The convolutional neural network model architecture implemented is shown below.

<p align="center">
    <img width="420" height="1000" src="https://github.com/ChavdaDhvani/steel-surface-defect-detection-system/blob/main/cnn_architecture.png?raw=true">
</p>

The following Model Accuracy & Validation Loss curves are obtained after 20 epochs.


<p align="center">
    <img width="420" height="180" src="https://github.com/ChavdaDhvani/steel-surface-defect-detection-system/blob/main/Model%20Accuracy%20and%20Loss.png?raw=true">
</p>

The results obtained have high test set accuracy and are shown below.

<p align="center">
    <img width="420" height="400" src="https://github.com/ChavdaDhvani/steel-surface-defect-detection-system/blob/main/Result.png?raw=true">
</p>

The experimental results demonstrate that this method meets the robustness and accuracy requirements for metallic defect detection. Meanwhile, it can also be extended to other detection applications.

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```
### 2. Install Dependencies

```bash 
pip install -r requirements.txt
```
### 3. Create Model File Placeholder

```bash
New-Item steel_defect_model.h5
```
### 4.  Run the Project
```bash
python steel_surface_defect_detection.py
```
or launch the web app
```bash
python app.py
```

