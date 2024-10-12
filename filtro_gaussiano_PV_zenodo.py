import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def safe_convert(x):
    try:
        return float(x.replace(',', '.'))
    except ValueError:
        return x

# Leggi il dataset applicando il converter a tutte le colonne
df = pd.read_csv('./data/Dataset/Data_PV.csv', sep=';', converters={i: safe_convert for i in range(100)}, parse_dates=['Data'], dayfirst=True)
df.set_index('Data', inplace=True)

# Applica il filtro gaussiano a tutto il dataset per tutte le colonne numeriche
sigma = 2  # Ampiezza del filtro gaussiano
columns_to_filter = ['Producer_1', 'Producer_2', 'Producer_3']

df_filtered = df.copy()
for column in columns_to_filter:
    if df_filtered[column].dtype == np.float64:
        df_filtered[column] = gaussian_filter1d(df_filtered[column], sigma=sigma)

# Arrotonda i valori filtrati a tre cifre decimali
df_filtered = df_filtered.round(3)

# Filtra i dati per il mese di ottobre (ad esempio, ottobre 2019)
anno = 2019
df_october = df[(df.index.year == anno) & (df.index.month == 10)]
df_october_filtered = df_filtered[(df_filtered.index.year == anno) & (df_filtered.index.month == 10)]

# Verifica se ci sono dati per il mese selezionato
if df_october.empty:
    print(f"Non ci sono dati per ottobre {anno}.")
else:
    # Grafico dei dati originali di ottobre per Producer_1
    plt.figure(figsize=(12,6))
    plt.plot(df_october.index, df_october['Producer_1'], label='Dati Originali')
    plt.title(f'Dati Originali di Ottobre {anno} - Producer_1')
    plt.xlabel('Data')
    plt.ylabel('Produzione')
    plt.legend()

    # Grafico dei dati filtrati di ottobre per Producer_1
    plt.figure(figsize=(12,6))
    plt.plot(df_october_filtered.index, df_october_filtered['Producer_1'], label='Dati Filtrati', color='orange')
    plt.title(f'Dati Filtrati di Ottobre {anno} - Producer_1')
    plt.xlabel('Data')
    plt.ylabel('Produzione')
    plt.legend()

    # Mostra entrambi i grafici
    plt.show()

# Resetti l'indice per includere la colonna datetime nel CSV
df_filtered.reset_index(inplace=True)

# Salva il DataFrame filtrato nel CSV, includendo l'indice come colonna
df_filtered.to_csv('./data/Dataset/Data_PV_filtro_gaussiano.csv', index=False)
