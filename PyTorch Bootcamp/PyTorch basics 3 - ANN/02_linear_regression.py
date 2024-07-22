import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Inizializiamo in tensor
X = torch.linspace(1, 50, 50).reshape(-1, 1)
print(X)
# Settiamo un seme per i random per avere i risultati del docente
torch.manual_seed(71)
# Creiamo un tensor per fare "casino", specifichiamo che siano float
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
print(e)
# Creiamo una funzione, la e di prima serve per non avere un risultato
# completamente lineare, senza specificare il gradiente y non sarà
# in grado di elaborarlo
y = 2*X + 1 + e
print(y.shape) # torch.Size([50, 1])
# Per usare plot bisogna convertire il tensor in un numpy array
plt.scatter(X.numpy(), y.numpy()) # plt.show()
# Creiamo una rete con pesi e bias scelti a random
# Specifichiamo un altro seed per avere i risultati del docente
torch.manual_seed(59)
# Creiamo un modello
model = nn.Linear(in_features=1, out_features=1)
print(model.weight) # tensor([[0.1060]], requires_grad=True)
print(model.bias) # tensor([0.9638], requires_grad=True)



# Pytorch permette di definire i modelli come classi oggetto che possono
# conservare molteplici layer
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


torch.manual_seed(59)
# Instaziamo un modello
model = Model(1, 1)
print(model.linear.weight) # tensor([[0.1060]], requires_grad=True)
print(model.linear.bias) # tensor([0.9638], requires_grad=True)
# Con il complicarsi dei modelli conviene iterare
# attraverso tutti i parametri del modello
# I metodi non specificati della classe model vengono dalla classe
# ereditata usata come base di tutte le classi per le reti neurali
for name, param in model.named_parameters():
    print(name, '\t', param.item())
    # linear.weight 	 0.10597813129425049
    # linear.bias 	 0.9637961387634277
# Ora vediamo che succede se passiamo un tensor al nostro modello
x = torch.tensor([2.0])
print(model.forward(x)) # tensor([1.1758], grad_fn=<ViewBackward0>)
# Vediamo la performance del modello, senza allenamento e ne una
# loss function il modello non performerà bene
x1 = np.linspace(0.0, 50.0, 50)
print(x1)
w1 = 0.1059
b1 = 0.9637
y1 = w1*x1 + b1
plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, 'r')
# Il blu sono i risultati giusti, il rosso è l'andamento del modello
# plt.show()
# Fortunamente pytorch ha delle loss function da poter usare
criterion = nn.MSELoss()
# Ora ottimiziamo il nostro modello, per iniziare 0.001 va bene
# per impostare un learning rate, se il modello è lento scegliere
# un numero più basso e viceversa, serve sperimentare
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# Impostiamo un epoch, ossia un periodo in cui la rete
# passa per tutto il dataset
epochs = 50
losses = []
for i in range(epochs):
    i += 1
    # Prevedendo sul forward pass
    y_pred = model.forward(X)
    # Calcoliamo gli errori (error)
    loss = criterion(y_pred, y)
    # Registriamo gli errori
    losses.append(loss.item())
    print(f'epoch: {i} loss: {loss.item()} '
          f'weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}')
    # Per aggiustare i pesi e bias dopo aver calcolato la perdita e aver
    # stampato i risutati, i gradienti si accumulano con ogni back propagation
    # quindi a ogni epoch andrà resettato il gradiente
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # In base ai risultati quando vedi che l'errore non varia molto
    # il modello è stato allenato a sufficenza
plt.clf()
plt.plot(range(epochs), losses)
plt.ylabel('MSE LOSS')
plt.xlabel('Epoch')
# plt.show()
# Dopodichè avremo i nostri pesi e bias finalizzati, ora
# settiamo la nostre equazioni di previsione
x = np.linspace(0.0, 50.0, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
predicted_y = current_weight*x + current_bias
# Ora vediamo come sono a confronto con i valori veri
plt.clf()
plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, 'r')
plt.show()
