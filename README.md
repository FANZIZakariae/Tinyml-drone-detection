# 🛸 TinyML-Based Real-Time Drone Detection System

A lightweight AI system that detects and classifies drone sounds using deep learning on embedded hardware. Powered by **CNNs**, optimized with **TensorFlow Lite**, and deployed on a **Raspberry Pi 4**, this project showcases the power of **TinyML** for real-time, low-power audio classification.

---

## 🎯 Project Overview

With the rise of commercial and malicious drone use, there’s a growing need for robust and energy-efficient detection systems. This project leverages **acoustic profiling** and TinyML to detect drone presence and classify drone types **without relying on radar, RF, or visual tracking**.

---

## 🧠 Key Features

* 🔊 **Sound-Based Detection** via ambient audio
* 🧬 **Dual Deep Learning Models**:

  * Spectrogram-based CNN
  * Raw audio-based CNN
* ⚙️ **Edge Deployment** on **Raspberry Pi 4** using TFLite
* 🧪 **Tested in real-time** with live microphone input
* 💡 **Offline, low-latency inference** with no cloud dependency

---

## 🛠️ Tools & Technologies

| Component               | Role                                           |
| ----------------------- | ---------------------------------------------- |
| **Python / Keras**      | Model building and training                    |
| **TensorFlow Lite**     | Model quantization and edge deployment         |
| **Raspberry Pi 4**      | Embedded inference with mic array              |
| **ReSpeaker Mic Array** | High-quality sound capture                     |
| **Librosa**             | Audio preprocessing and spectrogram generation |
| **PyAudio**             | Real-time audio stream handling                |

---

## 📁 Project Structure

```plaintext
tinyml-drone-detection/
├── dataset/           # Labeled audio data used for training and testing
│   ├── unknown/       # Background/environmental audio samples
│   └── yes-drone/     # Drone sound audio samples
│
├── models/            # Trained models (.h5 and .tflite files)
│
├── notebooks/         # Jupyter notebooks for training, evaluation, and analysis
│
├── src/               # Python scripts for preprocessing, real-time detection, and Raspberry Pi integration
```

---

## 🔍 Methodology

### 1. **Spectrogram-Based Model**

* Converts 1s audio clips into **Mel spectrograms**
* Trained with CNN (2.3M parameters)
* Achieved **98.8% accuracy** on test data
* Best for visual interpretability

### 2. **Raw Audio-Based Model**

* Processes raw 16 kHz waveform directly
* Lighter preprocessing, faster inference
* After optimization:

  * **Validation Accuracy:** 94.9%
  * **Test Accuracy:** 93.8%

### 3. **Optimizations**

* Dropout, L2 regularization
* Model quantization via TFLite
* Testing with shorter clips (0.25s)
* Amplification of mic input for better real-world match

---

## 📊 Results

| Model Type      | Accuracy | Deployment            |
| --------------- | -------- | --------------------- |
| Spectrogram CNN | 98.8%    | PC                    |
| Raw Audio CNN   | 93.8%    | Raspberry Pi (TFLite) |

> Signal amplification boosted live mic accuracy from \~0.6% to \~46.75%

---

## 🚀 Deployment Strategy

* Trained models exported as `.tflite`
* Integrated on **Raspberry Pi 4**
* Real-time audio streamed and fed to inference engine via `tflite_runtime`
* No need for cloud or external compute

---

## 🧪 Limitations & Challenges

* **Live mic input** varies from training data due to mic quality and environment
* **Spectrogram generation** is too resource-intensive for real-time edge use
* **Signal inconsistencies** reduce model confidence on real recordings

---

## 🔄 Future Improvements

* Improve audio acquisition pipeline with better microphones
* Apply **gain calibration**, filtering, and normalization
* Expand drone dataset with more models and real-world scenarios
* Explore hybrid models or newer TinyML architectures (e.g., MobileNet)

---
