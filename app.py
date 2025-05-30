import os
import requests
import streamlit as st
st.set_page_config(page_title='Instrument Classifier', layout='centered')

from PIL import Image
import torch
from torchvision import transforms, models

from config import IMG_SIZE, MODEL_PATH, CLASS_NAMES

# Backed-up, pretrained model
MODEL_DOWNLOAD_URL = 'https://huggingface.co/jray-8/resnet-instrument-classifier/resolve/main/resnet18_instruments.pth'

# ====== ENSURE MODEL EXISTS ======
def download_model():
	if not os.path.exists(MODEL_PATH):
		with st.spinner("Downloading model..."):
			try:
				response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
				response.raise_for_status()
				with open(MODEL_PATH, "wb") as f:
					for chunk in response.iter_content(chunk_size=8192):
						if chunk:
							f.write(chunk)
				st.success("Model downloaded successfully.")
			except Exception as e:
				st.error(f"Failed to download model: {e}")

download_model()

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
	model = models.resnet18(weights=None)
	model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
	model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
	model.eval()
	return model

model = load_model()

# ====== TRANSFORM ======
transform = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225])
])

# ====== UI ======
st.title('🎶 Instrument Classifier')
st.caption('Upload an image — get the instrument. Powered by PyTorch + ResNet18.')

st.markdown('##### 🎼 Supported Instruments:')

rows = [CLASS_NAMES[i:i+5] for i in range(0, len(CLASS_NAMES), 5)]
for row in rows:
	cols = st.columns(len(row))
	for i, cls in enumerate(row):
		cols[i].markdown(
			f'''
			<div style='
				background-color: #f0f2f6;
				padding: 8px 12px;
				border-radius: 12px;
				text-align: center;
				font-weight: 600;
				font-size: 14px;
				color: #000000;
				border: 1px solid #d3d3d3;
				margin-bottom: 10px;
			'>{cls.title()}</div>
			''',
			unsafe_allow_html=True
		)

st.markdown('---')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
	image = Image.open(uploaded_file).convert('RGB')
	st.image(image, caption='Uploaded Image', use_container_width=True)

	input_tensor = transform(image).unsqueeze(0)

	with torch.no_grad():
		output = model(input_tensor)
		_, pred = torch.max(output, 1)
		predicted_class = CLASS_NAMES[pred.item()].title()

	st.success(f'🎯 Predicted Instrument: **{predicted_class}**')

else:
	st.info('Upload an image of one instrument above.')
