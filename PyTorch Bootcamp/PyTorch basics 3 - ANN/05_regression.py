import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('../PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')
print(df.head())
# Vediamo la descrizione di una delle colonne
print(df['fare_amount'].describe())
# Uno dei primi problemi che sorgono è come usare la latitudine e
# longitudine per capire quanta distanza è stata percorsa
# In nostro soccorso arriva la formula di Haversine, ciò ci permette
# di calcolare la distanza percorsa su una sfera data una latitudine
# e una longitudine
# Ovviamente bisogna anche capire come tradurre la formula in python,
# in questo caso la formula sarà una cosa del genere
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

# Ora calcoliamo le distanze con una nuova colonna nel dataframe
df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude',
                                   'dropoff_latitude', 'dropoff_longitude')
# print(df.columns) # per mettere il nome delle colonne puoi stamparle e copiarle
# Vediamo la distanza percorsa di ogni viaggio e se la colonna ha funzionato
# la feature engineering consiste in questo, prendi caratteristiche che conosci
# e le lavori per crearne una nuova che ti è più utile
# È prevedibile immaginare che ci sarà una correlazione tra costo e distaza ora
print(df.head())
df.info()
# Vediamo anche che il pickup datetime è di tipo stringa, bisogna convertirlo
# in un formato numerico per usarlo in modo efficente
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df.info()
# Ora possiamo estrarre informazioni come le ore o il giorno della settimana
# Specificando un indice dopo il nome della colonna vai a prendere
# la riga con quell'indice
print(df['pickup_datetime'][0])
my_time = df['pickup_datetime'][0]
# Come vedrai ora puoi usare i metodi degli oggetti datetime
print(my_time.hour)
# Capire cosa può servire nei dati è fondamentale
# Un piccolo problema è che le date del csv erano in fusorario EDT,
# ma la conversione che abbiamo fatto le ha rese in UTC, quindi i nostri
# dati orari avranno una differenza di 4 ore
# Possiamo creare una colonna e preservare il fusorario
df['EDT date'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
print(df.head())
# Creiamo altre colonne possibilmente utili
df['Hour'] = df['EDT date'].dt.hour
# Tra cui vediamo di differenziare orari am da quelli pm
# in questa condizione la prima am è la true value, pm è la false
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
print(df.head())
# Vediamo che giorno della settimana era di preciso
# per un formato numerico usa il metodo dayofweek
df['Weekday'] = df['EDT date'].dt.strftime('%a')
print(df.head())



# Differenziamo le colonne con dati continui da quelli categorici
# Le ore possono essere 24, ogni ora è una categoria, o continua
# dove va da 1 a 24, devi decidere come preferisci
# In questo caso la tratteremo come categorica
cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
             'dropoff_latitude', 'passenger_count', 'dist_km']
# Scegliamo una colonna per allenarci non la regressione
y_col = ['fare_amount']
# Cambiamo il tipo degli elementi categorizzati
print(df.dtypes)
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
    # Ora ogni elemento che avevamo messo in cat cols è categorizzato,
    # ossia abbiamo distinto gli elementi in valori distinti, tipo unique
print(df['Hour'].head())
# Come vedi ora la categoria 'Hour' ha 24 elementi distinti
print(df['AMorPM'].head())
# Invece am e pm ha solo 2 categorie distinte, weekday ne avrà 7
# Per vedere quali sono queste categorie
print(df['Weekday'].cat.categories)
# A ogni categoria viene assegnato un codice, tipo un indice
# se aggiungi il metodo values ti verrà restituito un numoy array
print(df['Weekday'].cat.codes)
# Salviamoci gli array
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values
# Stackiamoli insieme come colonne
cats = np.stack([hr, ampm, wkdy], axis=1)
print(cats)
# Trasformiamo gli array in tensor
cats = torch.tensor(cats, dtype=torch.int64)
# Per le variabili continue invece
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
# Ora rendiamo un tensor anche i label
y = torch.tensor(df[y_col].values, dtype=torch.float)
# Vediamo le shape
print(cats.shape, conts.shape, y.shape)
# Ora bisogna impostare una embedding size, dimensione di incorporamento,
# per le colonne categoriche, così poi un modello tabulare sarà in grado
# prendere le colonne categoriche e i loro embenddings.
# Raggruppiamo il totale di valori unici per ogni categoria
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
# Ora vogliamo impostare le nostre embedding sizes basandoci sulle categorie
# in base al valore prenderà 50 o più basso diviso 2, non vogliamo float
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]



# Creiamo il modello tabulare
# Prendiamo un pezzo dei nostri cat data
catz = cats[:2]
# Impacchettiamo le dimensioni
selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
# FORWARD METHOD (cats)
embeddingz = []
for i, e in enumerate(selfembeds):
    embeddingz.append(e(catz[:, i]))
z = torch.cat(embeddingz, 1)
print(z)
# Proviamo a ridurre i dati a 0 ogni tanto dando una percentuale di avvenimento
# è un metodo per non andare in overfitting
selfembdrop = nn.Dropout(0.4)
z = selfembdrop(z)
print(z)
# Creiamo la classe per il modello tabulare abbastanza flessibile
# per essere usato con altri dataset
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



# Proviamo il modello
torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200, 100], p=0.4)
# print(model)
criterion = nn.MSELoss() #np.sqrt(MSE) --> RMSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 60000
test_size = int(batch_size*0.2)
# Mescoliamo i dati
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]
# Vediamo le grandezze dei train e test dataset
print(len(con_train), len(cat_test))
# Teniamo traccia del tempo impiegato
start_time = time.time()
# Iniziamo il modello
epochs = 300
losses = []
for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss.item())
    if i%10 == 1:
        print(f'Epoch {i} loss is {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
duration = time.time() - start_time
print(f'Tempo impiegato {duration/60} minuti')
# Mettiamo a grafico
plt.clf()
plt.plot(range(epochs), losses)
# plt.show()
# Vediamo con i test data
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
    print(loss)
# Vediamo quanto ha indovinato
for i in range(10):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f'{i+1}.) Previsto: {y_val[i].item():8.2f} Veri valori: {y_test[i].item():8.2f} '
          f'Differenza: {diff:8.2f}')
# Salviamo il modello
torch.save(model.state_dict(), 'my_taxi_model_L05.pt')
