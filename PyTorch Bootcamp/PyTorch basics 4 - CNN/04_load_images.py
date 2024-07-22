from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Dataset cani e gatti
# https://drive.google.com/file/d/1fuFurVV8rcrVTAFPjhQvzGLNdnTi1jWZ/view
# Path cartella immagini
# /home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS
"""
# Testiamo l'import di una delle immagini
with Image.open('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/test/CAT/10107.jpg') as im:
    im.show()
"""
# Facciamo una lista di immagini
# Inserisci il path alla cartella dei cani e gatti, no test no train
path = '/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/'
# Inizializiamo una lista
img_names = []
# Iniziamo a caricare le immagini
for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'/'+img)
# Vediamo se abbiamo caricato le immagini, sono quasi 25k
print(len(img_names))
# Creiamo un dataframe delle dimensioni e forme di queste immagini
img_sizes = []
rejected = []
# Salviamo le immagini che il programma riesce ad aprire per non avere
# errori successivamente
for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
# Vediamo quante immagini andavano bene
print(len(img_sizes))
print(len(rejected))
# Nessun problema
# Trasformiamo la lista in un dataframe
df = pd.DataFrame(img_sizes)
# I file sono base (0) e altezza (1)
print(df.head())
# Vediamo la larghezza media di tutte le immagini
print(df[0].describe())
# invece per l'altezza
print(df[1].describe())



# Aprimo l'immagine di un cane
dog = Image.open('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/train/DOG/14.jpg')
# dog.show()
print(dog.size)
# Vediamo il valore di un pixel specifico, da 1 a 255 nell'RGB
print(dog.getpixel((0, 0)))
# Ora per lavorarci ci serve rende l'immagine un tensor
# Con compose possiamo passare diversi tipi di trasformazione
transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(dog)
print(type(im), im.shape) # colori, altezza, base
# Vediamo a grafico l'immagine
# ma per matplot le dimensioni dell'immagine non vanno bene
# le vuole: altezza, base, colori
# Quindi cambiamo l'ordine degl'indici
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()
# Vediamo l'immagine in formato tensor, sono da 0 a 1 i pixel
# i numeri si posso convertire nel formato del RGB da 1 a 255
# semplicemente moltiplicando per 255 e viceversa se vogliamo da 0 a 1
print(im[:,0 , 0])
print(np.array((90, 95, 98)) / 255)
# E se vogliamo ridimensionarla?
transform = transforms.Compose([
    # Ridimensioniamo: altezza, base
    # Ovviamente è anche possibile stretchare le immagini
    transforms.Resize((250, 250)),

    # Si può eseguire il cosidetto center crop, ossia
    # si prende un quadrato centrale dell'immagine per il
    # lato specificato, nel caso 250x250 dal centro
    # Nel caso di immagini centrate può essere comodo per
    # avere tutte le immagini con la stessa grandezza
    # può aiutare fare un ridimensionamento e poi il CenterCrop
    transforms.CenterCrop(250),
    transforms.ToTensor()
])
im = transform(dog)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()
# Facciamo delle operazioni per modificare le immagini
transform = transforms.Compose([
    # Passiamo una percentuale di successo di ruotare orizzontalmente
    # da 0 a 1 (0% - 100%)
    # transforms.RandomHorizontalFlip(p=1),

    # Facciamo una rotazione casuale, il primo parametro è di quanti gradi
    # girerà da un lato casuale in un range da 0 gradi a quelli desiderati
    transforms.RandomRotation(30),
    transforms.ToTensor()
])
im = transform(dog)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()
# Ovviamente è possibile usare più metodi insieme
# basta usare prima il Compose method

# Ora se vogliamo usare reti già allenate dovremmo allenarle con
# immagini già normalizzate allo stesso modo, per questo
# ci sono dei valori comodi e fissi usati da tanti
transform = transforms.Compose([
    transforms.ToTensor(),
    # I valori passati sono usati in tantissimi training
    # e c'è un motivo specifico per il quale sono comodi
    # e molto efficenti, ovviamente sono tre per l'RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
im = transform(dog)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# Ci sarà un warning perchè alcuni numeri non sono validi per il grafico
# a cause del mean e std inseriti
# plt.show()
# L'immagine avrà dei valori negativi in alcuni punti
print(im)
# Per risolvere facciamo la normalizzazione inversa
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
plt.show()
