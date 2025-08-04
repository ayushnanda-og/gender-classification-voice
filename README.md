# Gender Recognition from Voice using Deep Learning

This project presents a **CNN-LSTM hybrid model** for **gender classification** from voice signals. It leverages **machine-crafted MFCCs** and **spectrogram features** from the **TIMIT speech dataset**, achieving **98.85% accuracy** on the test set. The core implementation is in the [`Gender_recognition.ipynb`](Gender_recognition.ipynb) notebook.

---

## 📁 Project Structure

- ├── archive/ # Contains raw dataset files and related data
- ├── Gender_recognition.ipynb # Main Jupyter notebook with code and analysis
- ├── README.md # Project documentation

---

## 🧠 Project Summary

- **Objective**: To identify speaker gender (male/female) from raw audio using deep learning.
- **Dataset**: [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1)
- **Model**: CNN for spatial feature extraction + LSTM for temporal dependencies.
- **Features**: Machine-learned Mel-Frequency Cepstral Coefficients (MFCCs), Spectrograms.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

---

## 📊 Model Highlights

- **Feature Extraction**: MFCCs from WAV files using `librosa`.
- **Preprocessing**: Resampling, padding, SMOTE oversampling to handle class imbalance.
- **Model Training**: Trained for 30 epochs with Adam optimizer, categorical cross-entropy loss.
- **Performance**:
  - Accuracy: **98.85%**
  - Precision (Male): **0.98**, (Female): **0.96**
  - F1-Score (Male): **0.98**, (Female): **0.96**

---

## 💻 Running in GitHub Codespaces

To run this project in **GitHub Codespaces**, follow these steps:

### 1. Create Codespace
- Go to your GitHub repo.
- Click **Code > Codespaces > Create codespace on main**.

### 2. Install Dependencies
Once the Codespace is ready, open a terminal and run:

```bash
pip install -r requirements.txt
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras librosa imbalanced-learn
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

```
📜 Reference
This project is based on the paper:

- "A Multi-Feature Deep Learning Approach for Gender Recognition from Voice"
- Rashmita Barik, Ayush Nanda
- Odisha University of Technology and Research
