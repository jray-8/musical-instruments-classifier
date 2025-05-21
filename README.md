# 🎶 Musical Instrument Classifier

A deep learning web app that classifies musical instruments from images using a fine-tuned ResNet18 model.

## 🔍 Supported Instruments
Accordion, Banjo, Drum, Flute, Guitar, Harmonica, Saxophone, Sitar, Tabla, Violin

## 🚀 Features
- Upload an image of one of the musical instruments
- Model predicts its class in real time
- Built with PyTorch + Streamlit

## 📁 Dataset

This project uses a curated image dataset of musical instruments, organized into class-specific folders.

The dataset (~6MB) is included in the `data/` folder.

You can view it on __[kaggle.](https://www.kaggle.com/datasets/nikolasgegenava/music-instruments)__  

## 🧠 Tech Stack
- Python
- PyTorch
- torchvision
- Pillow
- Streamlit

## 🔧 How to Run

1. Clone this repository:

		git clone https://github.com/jray-8/musical-instruments-classifier.git

		cd musical-instruments-classifier

2. Install dependencies:

		pip install -r requirements.txt

3. Train the model:

		python train.py

4. Run the Streamlit app:

		streamlit run app.py
