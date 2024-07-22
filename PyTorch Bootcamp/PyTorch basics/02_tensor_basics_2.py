import torch
import numpy as np


new_array = np.array([1, 2, 3])
# Come hai visto esiste il metodo tensor
# però esiste un metodo uguale ma con la maiuscola
print(torch.tensor(new_array))
print(torch.tensor(new_array).dtype)
t_new_array = torch.Tensor(new_array)
print(t_new_array)
print(t_new_array.dtype)
# La differenza sta nel fatto che il metodo con la maiuscola
# convertirà i numeri degli array in un formato float

# Per creare dei tensor da zero creiamo un placeholder
# un specie di tensor non inizializzato
# Di base vogliamo inizializzare un tensor vuoto
# che prenda già in anticipo uno spazio in memoria
print(torch.empty(4, 2))
# Se vogliamo un tensor di zeri, possiamo anche specificare di che tipo
# di base saranno sempre float
print(torch.zeros(4, 3, dtype=torch.int))
# Come con gli array esiste anche un metodo per creare un tensor di uno
print(torch.ones(4, 3))
# Ci sono anche altri metodi presi da numpy su pytorch
print(torch.arange(0, 18, 2).reshape(3, 3))
print(torch.linspace(0, 18, 12).reshape(3, 4))
print(torch.tensor([1, 2, 3]))
# Nel caso tu voglia cambiare il type specifico
# di un elemento, non solo tipologia
my_tensor = torch.tensor([1, 2 ,3])
print(my_tensor.dtype)
my_tensor = my_tensor.type(torch.int32)
print(my_tensor.dtype)
# Con i metodi di numpy per creare numeri casuali
print(torch.rand(4, 3))
print(torch.randn(4, 3))
# Per un risultato più customizzato posso settare
# un valore minimo, massimo e la forma con un altro metodo
print(torch.randint(low=0, high=11, size=(5, 5)))

# Se voglio crearmi un tensor partendo dalla forma di un altro
x = torch.zeros(2, 5)
print(x)
print(torch.rand_like(x))
print(torch.randn_like(x))
# Con randint dovrai comunque passare un low e high
print(torch.randint_like(x, low=0, high=11))
# Anche pytorch ha i seed per i numeri casuali
# comodo nel caso tu voglia dei numeri casuali costanti
torch.manual_seed(42)
# Solitamente alcuni scelgono 42 come citazione a
# "The Hitchhiker Guide to the Galaxy"
print(torch.rand(2, 3))
# Il seed è uguale a come funzionava su numpy,
# i numeri che escono fuori saranno nello stesso ordine
