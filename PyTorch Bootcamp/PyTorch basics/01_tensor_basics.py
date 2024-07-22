import torch
import numpy as np


# Per vedere la versione di pytorch in possesso
print(torch.__version__)
# Al momento delle lezioni la 2.2.2 è la stable version
# Creiamo un array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))
print(arr.dtype)
# Per convertire un array in un tensor con pytorch
t_arr = torch.from_numpy(arr)
print(t_arr)
print(type(t_arr))
print(t_arr.dtype)
# si può usare anche altro metodo, questo è più generico
t_arr = torch.as_tensor(arr)
print(t_arr)
# Facciamo un esempio con un array a due dimensioni
arr2d = np.arange(0.0, 12.0).reshape(4, 3)
print(arr2d)
t_arr2d = torch.from_numpy(arr2d)
print(t_arr2d)

# Il punto di questa conversione pero è che l'array e il tensor
# sono linkati e modificare uno va a modificare anche l'altro
arr[0] = 99
print(arr)
print(t_arr)
# Per non avere il link tra i due
my_arr = np.arange(0, 10)
t_my_arr = torch.tensor(my_arr)
t2_my_arr = torch.from_numpy(my_arr)
print(t_my_arr)
print(t2_my_arr)
my_arr[0] = 999
print(t_my_arr)
print(t2_my_arr)
# Come vedi usando il metodo tensor (non as_tensor)
# modificare l'array d'origine non influenza il tensor
