#Drowsiness Detection System

A deep learning-powered, vision-based system for real-time driver drowsiness detection using facial video analysis. The model combines **CNN** for spatial feature extraction and **LSTM** for temporal pattern modeling to classify drivers as **Alert** or **Drowsy**, helping reduce fatigue-related road accidents.

##  Features

- Hybrid **CNN-LSTM** architecture for spatio-temporal analysis
- Trained on real-world driver videos from the **UTA-RLDD** dataset
- Binary classification (Alert vs. Drowsy) with high emphasis on **recall** for the Drowsy class
- Two deployment modes:
  - Video file upload (Streamlit interface)
  - Real-time webcam inference with rolling buffer
- Confidence scores and visual feedback
- Lightweight model (~3.2M parameters) suitable for further edge optimization

## Project Overview

Driver fatigue remains one of the leading causes of road accidents worldwide. Traditional solutions often rely on intrusive sensors or manual monitoring. This project presents a **non-invasive**, camera-based AI system that:

- Detects prolonged eye closure and fatigue-related facial patterns
- Analyzes short video sequences (10 frames)
- Provides real-time / near-real-time predictions
- Is fully implemented with preprocessing, training, evaluation, and deployment code

##  Dataset

The model is trained on the **[UTA Real-Life Drowsiness Dataset (UTA-RLDD)](https://sites.google.com/view/utarldd/home)** — one of the largest realistic drowsiness datasets publicly available.
