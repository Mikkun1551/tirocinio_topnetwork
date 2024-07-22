# Progetto finale - Full stack con deep learning

## Contenuti
* [Panoramica](#panoramica)
* [Struttura](#struttura)

## Panoramica
Il seguente progetto consiste nell'avviare una pagina html con un server flask su browser dove caricare immagini che verranno 
passate a un modello EfficientNetB2 per fare image classification sulla presenza di 5 classi specifiche

## Struttura
Di seguito una spiegazione delle varie cartelle e file del progetto
### Training
Sono presenti i file usati per creare un dataset minore di Food101 con 5 classi e minori immagini usato per allenare 
il modello usato per fare image classification
### static
Sono presenti i file css e javascript per la pagina html
### templates
Cartella del file html, va sistemato un url
### server_dev_db.py
File di configurazione del server flask, vanno modificati alcuni path
### inference.py
File dove viene eseguita l'inference per l'image classification delle immagini
### 5_foods_classes.txt
File dove sono presenti le classi del modello EffNetB2
