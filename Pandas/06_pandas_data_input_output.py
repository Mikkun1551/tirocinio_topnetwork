import pandas as pd


# Per importare dei dataframe in formato csv
df = pd.read_csv('example.csv')
# Salvalo come variabile
print(df)
# Se sono interessato solo ad alcune colonne del dataframe
# le salvo in un nuovo dataframe
newdf = df[['a', 'b']]
print(newdf)
# Per salvare il nuovo dataframe in un file in locale
# Attenzione alla nomenclatura, se è un nome di un file esistente
# verrà sovrascritto altrimenti verrà creato un file nuovo
newdf.to_csv('mynew.csv', index=False)
# Per evitare di salvare l'indice creato da pandas
# settare a False in parametro dell'index, nel caso di
# voler conservare l'indice settarlo a True ma di base
# è prefefinito su True

# Per importare file excel, selezionare il sheet desiderato
ex_df = pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1')
print(ex_df)
# Per rimuovere eventuali indici resi come colonne
ex_df.drop('Unnamed: 0', axis=1)
print(df)

# Pandas può anche interagire con tabelle in pagine html,
# anche se per farlo servono diverse librerie esterne
# mylist_of_tables = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
# Di base pandas salverà i risultati come una lista
# print(type(mylist_of_tables))
# Quindi puoi vederne anche la lunghezza
# len(mylist_of_tables)
# Sarà di lunghezza 1, per cui per interagirsci tocca
# chiamare il primo elemento
# print(mylist_of_tables[0])
