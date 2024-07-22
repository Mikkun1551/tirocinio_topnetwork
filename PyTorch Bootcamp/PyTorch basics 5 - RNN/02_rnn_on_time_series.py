import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Il seguente viene usato per illustrare i valori datatime con matplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Il seguente database indica i guadagni mensili di alcol in america
# dal 92 al 2019 in milioni
df = pd.read_csv('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/TimeSeriesData/Alcohol_Sales.csv',
    index_col=0, parse_dates=True)
print(df.columns)
print(len(df))
# Devi decidere cosa fare se il tuo dataframe ha dei valori nulli
# in questo caso non li ha
df = df.dropna()
"""
df.plot(figsize=(12, 4))
plt.show()
"""
# Il df tratta i numeri come object, rendiamoli float
y = df['S4248SM144NCEN'].values.astype(float)
# Ora prendiamo solo gli ultimi 12 mesi per i test data
# e il resto per i training data
test_size = 12
train_set = y[:-test_size]
test_set = y[-test_size:]
# Spesso se il dataset è normalizzato le RNN performano meglio
# per il modo in cui si aggiornano i pesi e i bias
scaler = MinMaxScaler(feature_range=(-1, 1))
# In questo caso bisogna dare valore minimo e massimo SOLO DEL TRAINING
# altrimenti avviene il data leakege, semplicemente la rete sarà a
# conoscenza di un valore massimo che non dovrebbe conoscere dai test data
# anche perchè poi perderebbero di senso visto che non dovrebbe conoscerli
scaler.fit(train_set.reshape(-1, 1))
# Per trasformare un set
train_norm = scaler.transform(train_set.reshape(-1, 1))
# Passiamo il set da un array a un tensor
train_norm = torch.FloatTensor(train_norm).view(-1)
window_size = 12
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

train_data = input_data(train_norm, window_size)


class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size, output_size)
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]  # we only want the last value

torch.manual_seed(101)
model = LSTMnetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch = 100
start_time = time.time()
for i in range(epoch):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1} Loss {loss.item()}')
total_time = time.time() - start_time
print(f'Training took {total_time/60} minutes')
future = 12
preds = train_norm[-window_size:].tolist()
model.eval()
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())
# Ora devo invertire la normalizzazione
true_pred = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))
print(true_pred)
print(df['S4248SM144NCEN'][-12:])
# Mettiamo a grafico
x = np.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]')
plt.figure(figsize=(12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(df['S4248SM144NCEN'])
plt.plot(x,true_pred)
plt.show()
epochs = 100
# set model to back to training mode
model.train()
# feature scale the entire dataset
y_norm = scaler.fit_transform(y.reshape(-1, 1))
y_norm = torch.FloatTensor(y_norm).view(-1)
all_data = input_data(y_norm, window_size)
start_time = time.time()
for epoch in range(epochs):
    # train on the full set of sequences
    for seq, y_train in all_data:
        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    # print training result
    print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')
window_size = 12
future = 12
L = len(y)
preds = y_norm[-window_size:].tolist()
model.eval()
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())
true_pred = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
x = np.arange('2019-02-01', '2020-02-01', dtype='datetime64[M]').astype('datetime64[D]')
plt.figure(figsize=(12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(df['S4248SM144NCEN'])
plt.plot(x, true_pred[window_size:])
plt.show()
