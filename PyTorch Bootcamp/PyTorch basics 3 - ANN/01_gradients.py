import torch

# Inizializiamo un tensor con "requires_grad" per preparare il
# tracciamento computazionale del tensor, questo perchè le operazioni
# diventeranno attributi del tensor
x = torch.tensor(2.0, requires_grad=True)
# Definiamo una funzione polinominale
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
# Visto che y è stato creato come il risultato di un'operazione su un tensor
# su y c'è una funzione associata al gradiente, grad_fn di seguito
print(y) # tensor(63., grad_fn=<AddBackward0>)
print(type(y)) # <class 'torch.Tensor'>
# Il seguente comando esegue la back propagation e calcolerà tutti
# i gradienti automaticamente, essi stranno nell'attributo "grad"
y.backward()
print(x.grad) # tensor(93.)
# In questo modo abbiamo ricavato la derivata di y e applicato x



# Facciamo un esempio più ampio con più layer, creiamo un tensor
x = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
print(x) # tensor([[1., 2., 3.],
         # [3., 2., 1.]], requires_grad=True)
# Creiamo il primo layer
y = 3*x + 2
print(y) # tensor([[ 5.,  8., 11.],
         # [11.,  8.,  5.]], grad_fn=<AddBackward0>)
# Creiamo il secondo layer
z = 2*y**2
print(z) # tensor([[ 50., 128., 242.],
         # [242., 128.,  50.]], grad_fn=<MulBackward0>)
# Settiamo l'output come la matrice media
out = z.mean()
print(out) # tensor(140., grad_fn=<MeanBackward0>)
# Ora eseguiamo la back propagation per ricavare i gradienti
out.backward()
print(x.grad)  # tensor([[10., 16., 22.],
               # [22., 16., 10.]])
