import torch
from torchvision import transforms, models
from PIL import Image
import sys
import matplotlib.pyplot as plt

# ====== CONFIG ======
from config import IMG_SIZE, MODEL_PATH, CLASS_NAMES

# ======= LOAD MODEL =======
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ======= TRANSFORM =======
transform = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225])
])

# ======= LOAD IMAGE FROM ARG =======
if len(sys.argv) != 2:
	print('Usage: python predict.py <path_to_image>')
	sys.exit(1)

img_path = sys.argv[1]
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# ======= PREDICT =======
with torch.no_grad():
	output = model(input_tensor)
	_, pred = torch.max(output, 1)
	predicted_class = CLASS_NAMES[pred.item()]

# ======= SHOW =======
plt.imshow(image)
plt.title(f'Prediction: {predicted_class}')
plt.axis('off')
plt.show()
