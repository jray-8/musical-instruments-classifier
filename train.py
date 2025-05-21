from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from config import IMG_SIZE

# ======= CONFIG =======
DATA_DIR = Path('data')
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# ======= SETUP =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

# ======= TRANSFORMS =======
transform = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])
])

# ======= LOAD DATA =======
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes

# ======= SPLIT DATA =======
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ======= MODEL =======
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze base layers
for param in model.parameters():
	param.requires_grad = False

# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# ======= LOSS & OPTIMIZER =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ======= TRAINING =======
for epoch in range(EPOCHS):
	model.train()
	train_loss = 0.0
	correct = 0

	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, preds = torch.max(outputs, 1)
		correct += (preds == labels).sum().item()

	train_acc = correct / len(train_ds)
	print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}')

# ======= TEST EVALUATION =======
model.eval()
correct = 0

with torch.no_grad():
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, preds = torch.max(outputs, 1)
		correct += (preds == labels).sum().item()

test_acc = correct / len(test_ds)
print(f'Test Accuracy: {test_acc:.4f}')

# ======= SAVE MODEL =======
torch.save(model.state_dict(), 'resnet18_instruments.pth')
