import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Creiamo una classe modello
class Model(nn.Module):
    # Specifichiamo quante features ha il dataset (caratteristiche),
    # poi scegliamo quanti neuroni mettere per i layer specificati,
    # ricorda, non c'è un numero giusto di neuroni,
    # poi si indicano quante classi ha il dataset (tipi iris)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        # Fully connected, creiamo le connesioni tra layer,
        # ricorda, anche gli hidden layer sono arbitrari
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)


    def forward(self, x):
        # Scegliamo la activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


torch.manual_seed(32)
# Istanziamo il modello
model = Model()
# Importiamo il dataset
df = pd.read_csv('../PYTORCH_NOTEBOOKS/Data/iris.csv')
print(df.head)
# Target indica il tipo di iris con un label da 0 a 2
print(df.tail)
# Vediamo il dataset
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()
plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']
for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)
fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
# plt.show()
plt.clf()
# Ora eseguiamo un train_test_split
# Axis = 1 indica che è una colonna
X = df.drop('target', axis=1)
y = df['target']
# X è un pandas dataframe mentre y è un pandas series
# per convertirle in dei numpy arrays per poi ricavare dei tensor
X = X.values
y = y.values
# Per il placeholder delle variabili guarda la documentazione
# come sempre settare il random solo per avere i risultati del docente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
# Ora trasformiamo le features in float tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# Quando lavoriamo con le multi class classification vogliamo codificare
# le nostre labels, ma con pytorch se usiamo la cross entropy loss
# non serve fare quel passaggio
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
# Il criterion è il criterio che stai usando per calcolare la loss,
# di base è la loss function che prevede quanto il modello sta sbagliando
criterion = nn.CrossEntropyLoss()
# I paragoni del modello sono: prima si allena con x per poi passare a y
# con dei dati mai visti, poi finito il train fa lo stesso con i test
# Il learning rate è arbitrario, se vedi che il modello non migliora
# ossia la loss non cala, puoi diminuirlo o aumentarlo a seconda dei casi
# di base più basso è meglio ma più tempo ci mette, se lo alzi è il contrario
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model.parameters()) # <generator object Module.parameters at 0x7777377c76f0>
# Per vedere i parametri serve non mettere le parentesi, vedrai
# anche i layer e i neuroni di ciascuno
print(model.parameters) # <bound method Module.parameters of Model(
                        # (fc1): Linear(in_features=4, out_features=8, bias=True)
                        # (fc2): Linear(in_features=8, out_features=9, bias=True)
                        # (out): Linear(in_features=9, out_features=3, bias=True)
                        # )>
# Ora quanti epochs vogliamo eseguire? Il consiglio è iniziare da pochi
# se si vede un dataset nuovo e grande per poi ricavare la loss function,
# poi decidi se vuoi aumentare i numeri di epochs
# Un epoch è un passaggio della rete su tutto il training dataset
epochs = 100
# Per tenere traccia delle perdite inizializiamo una lista
losses = []
for i in range(epochs):
    # Passare il metodo forward fa si che: si passa la x, poi la x viene
    # passata alla rete, passa per la activation function e i risultati
    # vengono passati al layer successivo, alla activation function
    # e continua finchè non si ricava l'output
    y_pred = model.forward(X_train)
    # All'inizio la perdita sarà molto probabilmente alta,
    # per cui calcoliamo la loss
    loss = criterion(y_pred, y_train)
    # Teniamo traccia della loss
    losses.append(loss.item())
    if i%10 == 0:
        print(f'Epoch: {i} Loss: {loss.item()}')
    # Eseguiamo la back propagation per ricavare il gradiente
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# Vediamo a grafico come va la loss
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
# plt.show()
# Quando vedi che il modello non abbassa più la sua loss significa che
# hai raggiunto il numero ideale di epochs



# Ora alleniamo il modello sul test datase
# Ora diremo a pytorch che la backpropagation non è più necessaria
# visto che passiamo alla valutazione e non serve cambiare i pesi
# e i bias, ciò risparmia risorse e aiuta la valutazione
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(loss) # tensor(0.0581)
    # Il risultato è molto positivo, il modello non è andato in
    # overfitting con i training data, anzi ha performato anche meglio
# Ora facciamo in esercizio di classificazione visto che usiamo
# la funzione della cross entropy loss
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        # Vediamo le percentuali che la rete da per ogni classificazione
        # con a destra invece la classe giusta per indice delle classi
        print(f'{i+1}.) {str(y_val)} {y_test[i]}')
        # con argmax prendiamo l'indice del numero più grande nel tensor
        print(f'    {y_val.argmax()} {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'Abbiamo {correct} corretti!')
# Ora per salvare il modello
torch.save(model.state_dict(), 'my_iris_model_L04.pt')



# Creiamo un nuovo modello
new_model = Model()
# Per caricare un modello salvato in locale su una variabile
new_model.load_state_dict(torch.load('my_iris_model_L04.pt'))
# new_model.eval() per vedere com'è fatto il modello
# Proviamo a usare questo modello su dati mai visti
# Creiamo un'iris nuova
mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
# Visualizziamola su un grafico
plt.clf()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()
plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']
for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)
        # Inseriamo la nuova iris, che è simile a una setosa
        ax.scatter(mystery_iris[plots[i][0]], mystery_iris[plots[i][1]], color='y')
fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
# plt.show()
with torch.no_grad():
    print(new_model(mystery_iris))
    # Il modello prevede che il fiore che abbiamo creato sia una setosa
    print(new_model(mystery_iris).argmax())
