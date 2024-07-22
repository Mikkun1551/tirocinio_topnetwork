import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Creiamo dei dati
x = torch.linspace(0, 799, 800)
print(x)
# Definiamo una sine wave
y = torch.sin(x*2*3.141/40)
print(y)
# Mettiamo a grafico
"""
plt.figure(figsize=(12, 4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(y.numpy())
plt.show()
"""
# Ora creiamo un training e test set
test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]
"""
plt.figure(figsize=(12, 4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(train_set.numpy())
plt.show()
"""
# Definiamo una funzione che spezzi in batches i dati dei dataset
# ws = windowsize
def input_data(seq, ws):
    # Facciamo una lista di tuple, con una sequenza e una previsione
    out = [] # ([0, 1, 2, 3], [4]), ([1, 2, 3, 4], [5]) ...
    L = len(seq)
    # Visto che non possiamo fare previsioni su tutto il train
    # a causa dei valori mancanti, le previsioni dovranno fermarsi
    # quando l'ultima previsione sarà l'ultimo valore del dataset
    for i in range(L - ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out

# Il valore è arbitrario
window_size = 40
train_data = input_data(train_set, window_size)
print(len(train_data)) # 0-799 = 800 - 40 = 760 - 40 = 720
print(train_data[0])
print(train_data[1])



# Definiamo una classe per LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        # In questo caso vogliamo che LSTM stampi solo la previsione
        return pred[-1]

torch.manual_seed(42)
model = LSTM()
# Visto che usiamo valori continui non usiamo la cross entropy loss
# usiamo il mean square error loss
criterion = nn.MSELoss()
# Per l'RNN Adam non è indicato, conviene usare SGD (stocastic gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(model)
for p in model.parameters():
    # Circa 10k
    print(p.numel())



# Eseguiamo il training
epochs = 10
# Settiamo la lunghezza delle nostre previsioni
future = 40
for i in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1} Loss {loss.item()}')
    preds = train_set[-window_size:].tolist()
    for f in range(future):
        seq = torch.FloatTensor(preds[-window_size:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            preds.append(model(seq).item())
    loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
    print(f'Performance on test range: {loss}')
    """
    plt.figure(figsize=(12, 4))
    plt.xlim(700, 801)
    plt.grid(True)
    plt.plot(y.numpy())
    plt.plot(range(760, 800), preds[window_size:])
    plt.title(f'Epoch {i+1}')
    plt.show()
    """
# Facciamolo su tutti i dati
epochs = 15
all_data = input_data(y, window_size)
print(len(all_data))
start_time = time.time()
for i in range(epochs):
    for seq, y_train in all_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1} Loss {loss.item()}')
total_time = time.time() - start_time
print(f'It took: {total_time/60} minutes')
# Ora facciamo una previsione in un futuro non sicuro senza dati
# e vediamo se il modello mantiene l'onda sinusoidale
preds = y[-window_size:].tolist()
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())
plt.figure(figsize=(12, 4))
plt.xlim(0, 841)
plt.grid(True)
plt.plot(y.numpy())
# Stampiamo la previsione
plt.plot(range(800, 800+future), preds[window_size:])
plt.show()
