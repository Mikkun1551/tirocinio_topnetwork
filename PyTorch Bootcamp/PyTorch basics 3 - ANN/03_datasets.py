import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Importiamo un dataset
df = pd.read_csv('../PYTORCH_NOTEBOOKS/Data/iris.csv')
print(df.head())
print(df.shape)
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



# Vediamo come creare dei tensor divisi train-test
features = df.drop('target', axis=1).values
label = df['target'].values
# Specifichiamo la random per avere i risultati del docente
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=33)
# Ora abbiamo degli array, trasformiamoli in tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# Considerando che i target saranno o 0.0 o 1.0 possiamo convertire
# le y in longtensor
y_train = torch.LongTensor(y_train).reshape(-1, 1)
y_test = torch.LongTensor(y_test).reshape(-1, 1)
# Questo è un metodo molto manualizzato, se si vuole
# usare i metodi forniti da PyTorch
data = df.drop('target', axis=1).values
labels = df['target'].values
iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
print(type(iris), len(iris))
# for i in iris:
    # print(i)
# Dopo aver convertito i dati con il tensor dataset
# si possono avvolgere con il data loader
# Il problema poi sorge quando si passano questi dataset alla rete
# che se sono in elevate quantità ne risente, quindi serve passare
# un po' alla volta in gruppi di dati in ordine randomico
# Manualmente è un casino, ma con data loader
iris_loader = DataLoader(iris, batch_size=50, shuffle=True)
# In questo caso vogliamo batch da 50 elementi, ossia avremo 3 batch
for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)
