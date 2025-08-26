# 🔧 Predictive Maintenance Using Deep Learning (CNN + RNN)

This project explores the use of deep learning for **predictive maintenance** in industrial machinery. The goal is to anticipate machine failures before they occur by analyzing time-series sensor data using a hybrid **Convolutional Neural Network (CNN)** and **Recurrent Neural Network (RNN)** architecture.

The methodology is grounded in academic research and real-world applications, including a pilot study with Syncrobot.io and analysis based on the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 📦 Project Structure

```bash
📁 src/                   # Core logic and model architecture
  └── model.py           # CNN-RNN definition for predictive maintenance
  └── utils.py           # Preprocessing, scaling, and dataset handling

📁 notebooks/             # Evaluation notebooks and threshold optimization

📁 results/               # Output files from evaluations (ignored in Git)
📁 data/                  # Local folder for input datasets (not tracked)

.gitignore               # Excludes data, venv, and large artifacts
README.md                # Project documentation
