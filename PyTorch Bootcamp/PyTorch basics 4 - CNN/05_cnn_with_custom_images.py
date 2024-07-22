import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Settiamo i transform per i modelli che useremo
# questi sono dei valori consigliati dai thread del sito di pytorch
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    # Teniamo la chance base del metodo (50%)
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Facciamone uno per i test set ora, in questo caso
# togliamo le operazioni casuali visto che non è un training
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Carichiamo i dataset da Pytorch
root = '/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/'
train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)
torch.manual_seed(42)
# Le batch sono arbitrarie
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)
# Per come sono ordinati i file è più semplice
# ossia in sotto cartelle per i cani e per i gatti
# è più semplice l'import dei nomi delle classi
class_names = train_data.classes
# Ricordiamo che nel training le immagini non saranno mai le stess
# a cause dei random che abbiamo messo nel transform
print(class_names, len(train_data), len(test_data))
# Ora per essere sicuri che il loader prenda a random un cane o un gatto
for images, labels in train_loader:
    break
print(images.shape)
im = make_grid(images, nrow=5)
# Ricorda che le immagini sono state normalizzate, quindi invertiamo
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
# plt.show()
# Creiamo il modello
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) # (1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # (((224-2)/2)-2)/2 = 54
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        # Ricorda che i parametri sono arbitrari
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
print(CNNmodel)
for p in CNNmodel.parameters():
   # I parametri saranno molti di più dei precedenti
   # oltre i 5 milioni, immagina con una ANN...
   print(p.numel())



# Iniziamo il training
start_time = time.time()
epochs = 3
# OPZIONALE: Limitiamo il numero di batches
max_trn_batch = 800 # ognuna avrà 10 immagini, ossia 8000 immagini
max_tst_batch = 300 # 3000
train_losses = []
test_losses = []
train_correct = []
test_correct = []
# For loop per le epoch
for i in range(epochs):
   trn_corr = 0
   tst_corr = 0
   # Eseguiamo il training delle batch
   for b, (X_train, y_train) in enumerate(train_loader):
       # OPZIONALE: limitare il numero di batch
       # non passare su tutto il dataset se enorme
       if b == max_trn_batch:
           break
       b += 1
       # Istanziamo il modello
       y_pred = CNNmodel(X_train)
       loss = criterion(y_pred, y_train)
       # Contiamo il numero delle previsioni corrette
       predicted = torch.max(y_pred.data, 1)[1]
       # Si fa la somma tra i true (1) e i false (0)
       batch_corr = (predicted==y_train).sum()
       trn_corr += batch_corr
       # Aggiorniamo i parametri
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       # Stampiamo i risultati a interim
       if b&200 == 0:
           print(f'Epoch: {i+1} '
                 f'Batch: {b:4} [{10*b:6}/8000] '
                 f'Loss: {loss.item():10.8f} '
                 f'Accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
   train_losses.append(loss.item())
   train_correct.append(trn_corr)
   # Eseguiamo i testing batches
   with torch.no_grad():
       for b, (X_test, y_test) in enumerate(test_loader):
           # OPZIONALE: limitare il numero di batch
           # non passare su tutto il dataset se enorme
           if b == max_tst_batch:
               break
           b += 1
           # Istanziamo il modello
           y_val = CNNmodel(X_test)
           # Contiamo il numero delle previsioni corrette
           predicted = torch.max(y_val.data, 1)[1]
           tst_corr += (predicted==y_test).sum()
   loss = criterion(y_val, y_test)
   test_losses.append(loss.item())
   test_correct.append(tst_corr)
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')
torch.save(CNNmodel.state_dict(), 'cats_dogs_3_epoch_model.pt')
plt.clf()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at epoch')
plt.legend()
plt.show()
plt.plot([t/80 for t in train_correct], label='train accuracy')
plt.plot([t/30 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
num_c = test_correct[-1].item()
print(num_c * 100 / 3000)



# Possiamo usare dei modelli pre-allenati salvati da torchvision
AlexNetmodel = models.alexnet(pretrained=True)
print(AlexNetmodel)
# Visto è già allenata e la vogliamo così com'è
# congeliamo il calcolo del gradiente
for param in AlexNetmodel.parameters():
    param.requires_grad = False
# Se vogliamo possiamo modificare la classificazione di questa rete
# visto che era allenata a riconoscere 1000 categorie diverse,
# modifichiamola in modo che ne classifichi solo due
# per alexnet va modificato il parametro Sequential
torch.manual_seed(42)
AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                        nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(1024,2),
                                        nn.LogSoftmax(dim=1))
print(AlexNetmodel)
# Contiamo i parametri
for param in AlexNetmodel.parameters():
    print(param.numel())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr=0.001)
# Iniziamo il training
start_time = time.time()
epochs = 1
# OPZIONALE: Limitiamo il numero di batches
max_trn_batch = 800 # ognuna avrà 10 immagini, ossia 8000 immagini
max_tst_batch = 300 # 3000
train_losses = []
test_losses = []
train_correct = []
test_correct = []
# For loop per le epoch
for i in range(epochs):
   trn_corr = 0
   tst_corr = 0
   # Eseguiamo il training delle batch
   for b, (X_train, y_train) in enumerate(train_loader):
       # OPZIONALE: limitare il numero di batch
       # non passare su tutto il dataset se enorme
       if b == max_trn_batch:
           break
       b += 1
       # Istanziamo il modello
       y_pred = AlexNetmodel(X_train)
       loss = criterion(y_pred, y_train)
       # Contiamo il numero delle previsioni corrette
       predicted = torch.max(y_pred.data, 1)[1]
       # Si fa la somma tra i true (1) e i false (0)
       batch_corr = (predicted==y_train).sum()
       trn_corr += batch_corr
       # Aggiorniamo i parametri
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       # Stampiamo i risultati a interim
       if b&200 == 0:
           print(f'Epoch: {i+1} '
                 f'Batch: {b:4} [{10*b:6}/8000] '
                 f'Loss: {loss.item():10.8f} '
                 f'Accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
   train_losses.append(loss.item())
   train_correct.append(trn_corr)
   # Eseguiamo i testing batches
   with torch.no_grad():
       for b, (X_test, y_test) in enumerate(test_loader):
           # OPZIONALE: limitare il numero di batch
           # non passare su tutto il dataset se enorme
           if b == max_tst_batch:
               break
           b += 1
           # Istanziamo il modello
           y_val = AlexNetmodel(X_test)
           # Contiamo il numero delle previsioni corrette
           predicted = torch.max(y_val.data, 1)[1]
           tst_corr += (predicted==y_test).sum()
   loss = criterion(y_val, y_test)
   test_losses.append(loss.item())
   test_correct.append(tst_corr)
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')
# torch.save(CNNmodel.state_dict(), 'cats_dogs_3_epoch_model.pt')
num_c = test_correct[-1].item()
print(num_c * 100 / 3000)



image_index = 2019
im = inv_normalize(test_data[image_index][0])
plt.clf()
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
plt.show()
# Compariamo i due modelli con 5 esempi
CNNmodel.eval()
with torch.no_grad():
    new_pred = CNNmodel(test_data[image_index][0].view(1, 3, 224, 224)).argmax()
print(class_names[new_pred.item()])
AlexNetmodel.eval()
with torch.no_grad():
    new_pred = AlexNetmodel(test_data[image_index][0].view(1, 3, 224, 224)).argmax()
print(class_names[new_pred.item()])
