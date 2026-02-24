import zipfile
with zipfile.ZipFile("lung-cancer-histopathological-images.zip", "r") as zip_ref:
    zip_ref.extractall("lung_cancer_dataset")

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

root = '/content/lung_cancer_dataset'
classes = os.listdir(root)

for i in range(3):
    print(f"# of images in class {classes[i]}: {len(os.listdir(root+'/'+classes[i]))}")

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Redimensionamento das imagens
    transforms.ToTensor(), # Transformação das imagens para tensores
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalização das imagens
])

dataset = datasets.ImageFolder(root=root, transform=transform)

train_size, val_size = int(0.8 * len(dataset)), int(0.2 * len(dataset)) # 80% das imagens vai para treino, enquanto que 20% para teste

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # shuffle para evitar memorização (overfitting)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Definindo o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo ResNet-50 pré-treinado
model_resnet = models.resnet50(pretrained=True)

# Substituindo a última camada para classificar 3 classes
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 3)

# Movendo o modelo para o dispositivo
model_resnet.to(device)

# Definindo o critério de perda (CrossEntropyLoss) e o otimizador (SGD)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9)

print(model_resnet)  # Para verificar a arquitetura modificada

dataset = datasets.ImageFolder(root=root, transform=transform)

train_size, val_size = int(0.8 * len(dataset)), int(0.2 * len(dataset)) # 80% das imagens vai para treino, enquanto que 20% para teste

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # shuffle para evitar memorização (overfitting)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Laço de treinamento
for epoch in range(15):
    model_resnet.train()  # Colocar o modelo em modo de treinamento
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Transferir para o dispositivo

        optimizer.zero_grad()  # Zerar os gradientes

        outputs = model_resnet(inputs)  # Passar as entradas pelo modelo
        loss = criterion(outputs, labels)  # Calcular a perda
        loss.backward()  # Retropropagar a perda
        optimizer.step()  # Atualizar os pesos

        running_loss += loss.item()  # Acumular a perda

    # Exibir a perda média por época
    print(f"Epoch: {epoch+1} | Loss: {running_loss / len(train_loader)}")

# Salvar o modelo treinado
torch.save(model_resnet.state_dict(), "resnet50.pth")