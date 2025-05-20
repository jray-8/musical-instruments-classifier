# Image dimensions used for model input (ResNet)
IMG_SIZE = 224

# Path to save the model weights
MODEL_PATH = 'resnet18_instruments.pth'

# `torchvision.datasets.ImageFolder` sets class labels based on 
# case-sensitive folder names in `data/`, sorted alphabetically
CLASS_NAMES = [
	'accordion', 'banjo', 'drum', 'flute', 'guitar',
	'harmonica', 'saxophone', 'sitar', 'tabla', 'violin'
]
