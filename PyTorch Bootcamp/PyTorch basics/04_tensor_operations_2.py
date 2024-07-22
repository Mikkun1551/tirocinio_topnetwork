import torch


a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
# Moltiplicare due liste tra loro darà semplicemente
# una lista con i prodotti tra gli indici uguali
print(a.mul(b))
# Il prodotto scalare è la somma dei prodotti (dot product)
print(a.dot(b))
# ciò è possibile solo tra tensor con la stessa forma

# Creiamo due tensor con forma diversa
a = torch.tensor([[0, 2, 4],[1, 3, 5]])
b = torch.tensor([[6, 7],[8, 9],[10, 11]])
print(a)
print(a.shape)
print(b)
print(b.shape)
# Proviamo una moltiplicazione tra matrici
print(torch.mm(a, b))
# è possibile farla anche con @, ma non è consigliabile
# visto che viene usato per i decoratori
print(a @ b)
# La moltiplicazione tra matrici si fa prendendo tabelle che hanno
# un totale di colonne e righe invertite, prendendo le righe di una si
# moltiplicano per le colonne dell'altra in ordine ciascun numero, nel caso:
# C[0, 0] = (0*6) + (2*8) + (4*10) = 0 + 16 + 40 = 56
# C[0, 1] = (0*7) + (2*9) + (4*11) = 0 + 18 + 44 = 62
# C[1, 0] = (1*6) + (3*8) + (5*10) = 6 + 24 + 50 = 80
# C[1, 1] = (1*7) + (3*9) + (5*11) = 7 + 27 + 55 = 89

# Vediamo la norma euclidea
x = torch.tensor([1., 3., 4., 5.])
print(x.norm())
# La norma euclidea è la radice quadrata della somma dei quadrati

# Se vuoi sapere il numero di elementi in un tensor
print(x.numel())
# è uguale se fai
print(len(x))
# ma nei tensor a più dimensioni
print(len(a))
# restituirà il numero di vettori
# ma se usi numel come prima
print(a.numel())
# ti darà tutti gli elementi del tensor
