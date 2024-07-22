import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                            train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                           train=False, download=True, transform=transform)
print(train_data, test_data)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
# Le immagini nel CNN hanno 4 dimensioni
# Il primo indica il colore, il secondo i filtri (che sono arbitrari),
# il terzo la grandezza dei filtri (3x3) e il quarto lo stride, ossia
# di quanti pixel il filtro si deve spostare
# Il numero di filtri è arbitrario
conv1 = nn.Conv2d(1, 6, 3, 1)
# Visto che l'immagine nel primo layer passa per 6 filtri, poi per
# il pooling e infine al secondo layer, per questo il secondo layer
# ha il canale d'entrata con 6, la grandezza dei filtri e lo stride
# devono rimanere uguali
conv2 = nn.Conv2d(6, 16, 3, 1)
for i, (X_train, y_train) in enumerate(train_data):
    print(X_train.shape) # -> 4D batch (batch di un'immagine)
    break
x = X_train.view(1, 1, 28, 28) # da -> 1, 28, 28
x = F.relu(conv1(x))
# x.shape -> 1, 6, 26, 26
# Ricorda che nel cnn le immagini perdono infomazioni sui bordi
# per questo da 28 è diventato 26, non è un gran problema visto
# che le immagini di questo dataset sono molto centrali e vuote
# ai bordi, in altri casi potrebbe essere importante, per questo
# bisognerebbe passare il parametro padding al conv
x = F.max_pool2d(x, 2, 2)
# Visto che abbiamo un kernel e uno stride di 2 il risultato
# è di dimezzare altezza e larghezza
# x.shape -> 1, 6, 13, 13
x = F.relu(conv2(x))
# x.shape -> 1, 16, 11, 11
x = F.max_pool2d(x, 2, 2)
# In questo caso farà un approssimazione per difetto
# x.shape -> 1, 16, 5, 5
# In pratica si perdono pixel, si dimezza e si ripete, quindi da 28 a 5
# Usiamo -1 per tenere invariata la prima dimensione
x.view(-1, 16*5*5) # 1, 400



# Creiamo una classe
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        # Con il continuare degli strati riduci un po' i neuroni
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(42)
model = ConvolutionalNetwork()
print(model)
# Vediamo rispetto ai 100k caratteri della ANN quanti parametri usa CNN
for param in model.parameters():
    # Circa 60k, quasi metà molto meglio
    print(param.numel())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
start_time = time.time()
# Variabili tracker, non necessarie ma comode
epochs = 5
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
        b += 1
        # Visto che è una CNN non serve il flatten perchè accetta dati 2D
        # Istanziamo il modello
        y_pred = model(X_train)
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
        if b&600 == 0:
            print(f'Epoch: {i+1} Batch: {b} Loss: {loss.item()}')
    train_losses.append(loss.item())
    train_correct.append(trn_corr)
    # Eseguiamo i testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Istanziamo il modello
            y_val = model(X_test)
            # Contiamo il numero delle previsioni corrette
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted==y_test).sum()
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')



plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.title('Loss at epoch')
plt.legend()
plt.show()
plt.plot([t/600 for t in train_correct], label='train accuracy')
plt.plot([t/100 for t in test_correct], label='test accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted==y_test).sum()
    print(correct.item()/len(test_data))
np.set_printoptions(formatter=dict(int=lambda  x: f'{x:4}')) # formatter
print(np.arange(10).reshape(1, 10))
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))
# Passiamo una immagine al modello, scegliamo un indice arbitrario
plt.imshow(test_data[2019][0].reshape(28, 28))
plt.show()
model.eval()
with torch.no_grad():
    new_pred = model(test_data[2019][0].view(1, 1, 28, 28))
    print(new_pred.argmax())
