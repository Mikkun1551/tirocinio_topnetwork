import torch


# Rivediamo l'indexing e slicing
x = torch.arange(6).reshape(3, 2)
print(x)
print(x[1, 1])
# Se stampassi il type di sopra avrei semplicemente il type del tensor
print(x[:, 1])
print(x[:, 1:])

# Vediamo la differenza tra il metodo view e reshape,
# che sono molto simili tra di loro
x = torch.arange(10)
x.view(2, 5)
print(x)
# Il cambiamento non è permanente
x.reshape(2, 5)
print(x)
# La differenza è se vanno a modificare l'oggetto in memoria
# ma semplicemente prima esisteva view ma è stato richiesto
# reshape perchè la gente era abituata a numpy
# Sia con view che reshape, dovrai creare una nuova variabile
# per conservare le modifiche
z = x.view(2, 5)
print(z)
x[0] = 9999
print(x)
# Come su numpy modificare l'array originale modifichi anche la copia tensor
x = torch.arange(10)

# Durante il reshape posso selezionare e dimensioni senza preoccuparmi
# di dover specificare la quantità di elementi con -1,
# ovvimente se non è divisibile per le dimensioni selezionate non funzionerà
print(x.reshape(2, -1))
# Funziona anche nel caso che sappia quanti elementi voglio
# ma non voglio pensare alle dimensioni
print(x.reshape(-1, 5))

# Vediamo alcune operazioni tra tensor
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
# Facciamo una somma
print(a + b)
# Usando metodi di pytorch
print(torch.add(a, b))
# Una moltiplicazione
print(a.mul(b))
# Questo metodo non va a modificare i tensor
print(a)
# ma se metto un underscore al metodo invece
# andrà a modificare la varibile, questo vale per tutte le operazioni
print(a.mul_(b))
print(a)