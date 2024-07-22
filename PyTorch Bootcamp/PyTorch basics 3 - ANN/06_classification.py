import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Facciamo una prova di classificazione
df = pd.read_csv('../PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')
print(df.head())
print(df['fare_class'].value_counts())

def haversine_distance(df, lat1, long1, lat2, long2):
    # Calculates the haversine distance between 2 sets of GPS coordinates in df
    r = 6371  # average radius of Earth in kilometers
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers
    return d

df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude',
                                   'dropoff_latitude', 'dropoff_longitude')
df['EDT date'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4)
df['Hour'] = df['EDT date'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDT date'].dt.strftime('%a')
print(df.head())
print(df['EDT date'].min())
print(df['EDT date'].max())
cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
             'dropoff_latitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount'] # questa colonna contiene i label
# Convertiamo le tre colonne categoriche in dtype categorico
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values
cats = np.stack([hr, ampm, wkdy], axis=1)
print(cats[:5])
# Convertiamo le variabili categoriche in tensor
cats = torch.tensor(cats, dtype=torch.int64)
# la sintassi è ok, visto che i dati sorgente sono un array e non un tensor
print(cats[:5])
# Convertiamo le variabili continue in tensor
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
print(conts[:5])
print(conts.type())
# Convertiamo i label in tensor
y = torch.tensor(df[y_col].values).flatten()
print(y[:5])
print(cats.shape, conts.shape, y.shape)
# Settiamo la embedding sizes per hour, amorpm e weekday
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
# Questi sono i nostri dati sorgente
catz = cats[:4]
# Passiamo emb_szs quando il modello viene istanziato
# Questo viene assegnato dentro il metodo init
selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
# Questo avviene dentro il metodo forward()
embeddingz = []
for i, e in enumerate(selfembeds):
    embeddingz.append(e(catz[:, i]))
# Concateniamo le sezioni embedding (12, 1, 4) in una (17)
z = torch.cat(embeddingz, 1)
# Questo è stato assegnato sotto il metodo init
selfembdrop = nn.Dropout(0.4)
z = selfembdrop(z)
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        # emb_szs è una lista di tuple, ogni dimensione della variabile categoriale
        # è accoppiata a un embedding size
        # Impostiamo gli embedded layers, i dati categorici saranno filtrati
        # tramite questi embenddings nel metodo forward
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        # p sarà la percentuale di dropout per ridurre le possibilità
        # di overfitting
        # Impostiamo una funzione di dropout
        self.emb_drop = nn.Dropout(p)
        # n_cont è il numero di variabili continue
        # Settiamo una funzione di normalizzazione per le variabili continue
        self.bn_cont = nn.BatchNorm1d(n_cont)
        # Settiamo una sequenza di layer per la rete neurale dove ciascuno include
        # una funzione lineare, una activation function (relu), un normalization
        # step, e un dropout layer
        layerlist = []
        n_emb = sum([nf for ni, nf in emb_szs])
        n_in = n_emb + n_cont
        # Layers sarà un lista con il numero di neuroni per strato:
        # ex: layers = [200, 100, 50...]
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        # out_sz è la grandezza dell'output
        layerlist.append(nn.Linear(layers[-1], out_sz))
        # Combiniamo i layer di seguito
        self.layers = nn.Sequential(*layerlist)

    # Preprocessiamo gli embenddings e normalizziamo le variabili continue
    # prima di passarle attraverso i layer
    def forward(self, x_cat,x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        # Combiniamo i tensor in uno solo
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

torch.manual_seed(33)
"""
ECCO COSA CAMBIA
"""
# Modifichiamo le dimensioni dell'output
model = TabularModel(emb_szs, conts.shape[1], 2, [200, 100], p=0.4) # out_sz = 2
# Il criterion pure cambia, non più MSE
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 60000
test_size = int(batch_size*0.2)
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]
print(len(con_train), len(cat_test))
start_time = time.time()
epochs = 300
losses = []
for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    """
        loss error:
        RuntimeError: expected scalar type Long but found Double
    """
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    if i%10 == 1:
        print(f'Epoch {i} loss is {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
duration = time.time() - start_time
print(f'Tempo impiegato {duration/60} minuti')
plt.clf()
plt.plot(range(epochs), losses)
plt.show()
# Vedrai che rispetto alla rigressione essendo un problema più
# semplice l'apprendimento sarà più rapido
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
    print(loss)
rows = 50
correct = 0
print(f'{"MODEL OUTPUT":26} ARGMAX Y_TEST')
for i in range(rows):
    print(f'{str(y_val[i]):26} {y_val[i].argmax():^7}{y_test[i]:^7}')
    if y_val[i].argmax().item() == y_test[i]:
        correct += 1
print(f'\n{correct} out of {rows} = {100*correct/rows:.2f}% correct')
# Salvare il modello solo dopo il training
if len(losses) == epochs:
    torch.save(model.state_dict(), 'my_taxi_model_L06.pt')
else:
    print('Model has not been trained. Consider loading a trained model instead.')
