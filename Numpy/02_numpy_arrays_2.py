import numpy as np


# Per creare un array di numeri casuali tra 0 e 1 dato un numero specificato
# uso il seguente metodo
my_arr = np.random.rand(5)
print(my_arr)
# Posso anche scegliere le dimensioni dell'array mettendole prima
my_arr = np.random.rand(3, 5)
print(my_arr)
# Queste sono distrubuzioni uniformi

# Se voglio un numero casuale con la distribuzione normale
# dove il mean è 0 e la deviazione standard è 1
my_arr = np.random.randn(5)
# Con la deviazione di 1 avrò un numero casuale tra -1 e 1
print(my_arr)
# Se voglio selezionare il mean e deviazione serve un altro metodo dove
# il primo argomento è il mean desiderato, il secondo è la deviazione e
# il terzo è la quantità di numeri che voglio
my_arr = np.random.normal(5, 3, 2)
print(my_arr)

# Per creare numeri casuali entro un certo range specificato e in quanità
# desiderata si usa il seguente metodo dove: il primo numero è l'inizio,
# il secondo è il range a cui arrivare (escluso) e il terzo opzionale
# è il totale di numeri che voglio, altrimenti sarà uno solo
my_arr = np.random.randint(1, 50, 5)
print(my_arr)

# E' possibile settare un seed per creare un set di numeri casuali fissi,
np.random.seed(15)
# Scelto il seed i numeri che verranno creati saranno gli stessi e nello
# stesso ordine ogni volta che si richiama il seed
my_arr = np.random.rand(5)
print(my_arr)
# Da notare che il seed imposta un ordine di numeri casuali, ossia se nel
# caso precedente volevo sei numeri invece che cinque, il sesto numero
# sarebbe stato il primo dell'esempio appena sotto
my_arr = np.random.rand(5)
print(my_arr)

# Per un dato array è possibile modificarlo e dargli più dimenzioni
my_arr = np.arange(25)
print(my_arr)
# Chiederò di aggiungere 5 dimensioni con 5 elementi l'una, attenzione
# a chiedere solo sequenze possibili altrimenti verrà sollevato un errore
my_arr = my_arr.reshape(5, 5)
print(my_arr)

# Posso trovare il valore più alto in un array specificato
ranarr = np.random.randint(0, 50, 10)
print(ranarr)
print(ranarr.max())
# Anche il più basso
print(ranarr.min())
# E trovare il loro index
print(ranarr.argmax())
print(ranarr.argmin())

# Posso vedere il tipo del dato in un array
print(ranarr.dtype)