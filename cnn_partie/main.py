import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Télécharger CIFAR-10 automatiquement
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print("Dataset chargé :", len(trainset))

# Afficher une image
image, label = trainset[0]

plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {label}")
plt.show()