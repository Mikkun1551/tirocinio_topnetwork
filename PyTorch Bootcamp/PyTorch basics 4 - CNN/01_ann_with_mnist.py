import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time

# I DATI
# Vogliamo prendere delle immagini dal dataset MNIST e convertirle in tensor
# grazie alla libreria torchvision abbiamo dei dataset da usare
# MNIST è un dataset di numeri scritti a mano
transform = transforms.ToTensor()
# Scegliamo come e dove inserire le immagini di training e test
train_data = datasets.MNIST(root='.../Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                            train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.../Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                           train=False, download=True, transform=transform)
print(train_data, test_data)
print(type(train_data))
print(type(test_data))
# Se vediamo il tensor sarà per lo più composto da 0, bianco, il type è una tuple
print(train_data[0])
# Impacchetiamo l'immagine e il label in una variabile
image, label = train_data[0]
# Vedendo la dimensione l'1 sta a indicare che è monocolore, bianco e nero,
# 0 per uno e 1 per l'altro, a volte potrebbe essere da -1 a 1
print(image.shape)
# La label invece è un singolo numero
print(label)
# Vediamolo a grafico, ricorda l'immagine va ridimensionata a causa dell'1
plt.imshow(image.reshape((28, 28)), cmap='gist_yarg')
# L'immagine sarà colorata in automatico da matplot, non indica il
# colore vero, di base è viridis ma si può modificare
plt.show()
# Ora vogliamo caricare in batch le immagini, settando il seed anche
# lo shuffle con il quale verrà presi i dati sarà fisso nel suo random
torch.manual_seed(101)
# Prendiamo 100 immagini alla volta, mettiamo shuffle=true per essere
# sicuri di non avere immagini in ordine, potrebbe andare in overfitting
# con un solo numero
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)
# Vediamo un paio di immagini con make_grid
np.set_printoptions(formatter=dict(int=lambda  x: f'{x:4}')) # formatter
# Vediamo il primo batch
# Ricorda che sono 60000 immagini diviso 100, per cui facciamo un break
images, labels = 0, 0
for images, labels in train_loader:
    # Le info sono: 100 - tot immagini, 1 - monocolore, 28 e 28 base e altezza
    print(images.shape)
    print(labels.shape)
    # Stampiamo le prime 12 label
    print('Labels: ', labels[:12].numpy())
    # Stampiamo le prime 12 immagini
    im = make_grid(images[:12], nrow=12) # nrow di default è 8
    plt.figure(figsize=(10, 4))
    # Dobbiamo trasporre le immagini da CWH (color, width, height) a WCH,
    # ossia spostiamo gli indici dell'immagine
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    # plt.show()
    plt.clf()
    break



# CREAZIONE DELLA RETE
class MultilayerPerceptron(nn.Module):
    # input size sarà 784 perchè 28x28 = 784
    def __init__(self, in_sz=784, out_sz=10, layers=None):
        super().__init__()
        if layers is None:
            layers = [120, 84]
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1) # multi-class classification

torch.manual_seed(101)
model = MultilayerPerceptron()
print(model)
for param in model.parameters():
    print(param.numel())
# Una differenza tra ANN e CNN è che cnn richiede anche meno parametri
# in questo caso l'ANN sta usando 105.214 parametri con immagini molto piccole
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Dobbiamo convertire le immagini da [100, 28, 28] a [100. 784]
# con -1 diciamo che il resto delle dimensioni vanno modellate in una sola
# se non specifico le successive
images = images.view(100, -1)
print(images.shape)



# TRAINING
# Teniamo traccia del tempo del training
start_time = time.time()
# Allenamento
epochs = 10
# Tracciatori, non necessari, solo per vedere delle statistiche
train_losses =  []
test_losses = []
train_correct = []
test_correct = []
for i in range(epochs):
    # Teniamo traccia delle volte che indovina
    trn_corr = 0
    tst_corr = 0
    # Carichiamo le batch del training
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        # Inizializiamo la previsione
        y_pred = model(X_train.view(100, -1))
        # Paragoniamo la previsione alla risposta giusta per la loss
        loss = criterion(y_pred, y_train)
        # Visto che è una multi-class classification
        # prendiamo il risultato con la probabilità più alta
        predicted = torch.max(y_pred.data, 1)[1]
        # print(y_pred.data)
        # Vediamo quanti risultati giusti ci sono in una batch
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Stampiamo un piccolo report della batch
        if b%200 == 0:
            acc = trn_corr.item()*100/(100*b)
            print(f'Epoch {i+1} batch {b} loss {loss.item()} accuracy {acc}')
    # Prendiamo i dati ricavati
    train_losses.append(loss.item())
    train_correct.append(trn_corr)
    # Eseguiamo il test, stesso processo
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500, -1))
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted==y_test).sum()
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)
total_time = time.time() - start_time
print(f'Duration: {total_time/60} mins')



# VALUTAZIONE
# Facciamo qualche grafico
plt.plot(train_losses, label='Training loss')
# In un ambito reale la test loss sarà più alta di quella del training
plt.plot(test_losses, label='Test Loss')
# Come visto dai risultati oltre la decima epoch si rischia un overfitting
plt.legend()
plt.show()
train_acc = [t/600 for t in train_correct]
print(train_acc)
test_acc = [t/100 for t in test_correct]
print(test_acc)
plt.plot(train_acc, label='Train accuracy')
plt.plot(test_acc, label='Test accuracy')
plt.legend()
plt.show()
# Vediamo nuovi dati mai visti
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted==y_test).sum()
    # Accuratezza con dati nuovi
    print(100*correct.item()/len(test_data))
# Se vogliamo una matrice di confusione
# Per leggerla basta che prendi le classi previste in verticale
# e le combaci con le classi con lo stesso indice in orizzontale
# in pratica avrai in diagonale da in alto a sinistra a in basso
# a destra i numeri più alti della matrice
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))
