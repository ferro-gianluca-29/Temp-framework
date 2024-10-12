import pandas as pd

# Caricamento dei dati
df = pd.read_csv('./data/Dataset/Data_PV.csv', sep=';')

# Converti la colonna che contiene le informazioni temporali in datetime specificando il formato corretto
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M:%S')

# Converti le colonne numeriche che sono state lette come 'object' in tipo numerico
# Usa 'pd.to_numeric' per convertire le stringhe in float e gestisci gli errori con 'coerce'
for column in df.columns:
    if column != 'Data':
        df[column] = pd.to_numeric(df[column].str.replace(',', '.'), errors='coerce')

# Imposta la colonna di timestamp come indice del DataFrame
df.set_index('Data', inplace=True)

# Ricampionamento dei dati per ora e calcolo della media
resampled_df = df.resample('h').mean()  # Usa 'h' minuscolo

# Troncare i valori alla terza cifra decimale
resampled_df = resampled_df.round(3)

# Formatta l'indice per l'esportazione, mantenendo il datetime per le operazioni interne
resampled_df.index = resampled_df.index.strftime('%Y-%m-%d %H:%M:%S')

# Salva i dati ricampionati in un nuovo file CSV
resampled_df.to_csv('./data/Dataset/Data_PV_resampled_H.csv')
