import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Definizione dei modelli
lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=7)
dt = DecisionTreeRegressor(max_depth=10, random_state=27)
rf = RandomForestRegressor(n_estimators=20, random_state=1)

# Percorso della cartella e lista dei file
folder = './UCSD Microgrid Database/Data Files/PVGenerator/'
files = ['BioEngineeringPV.csv', 'BSB_BuildingPV.csv', 'BSB_LibraryPV.csv', 'CSC_BuildingPV.csv', 
         'CUP_PV.csv', 'EBU2_A_PV.csv', 'EBU2_B_PV.csv', 'ElectricShopPV.csv', 'GarageFleetsPV.csv', 
         'GilmanParkingPV.csv', 'HopkinsParkingPV.csv', 'KeelingA_PV.csv', 'KeelingB_PV.csv', 
         'KyoceraSkylinePV.csv', 'LeichtagPV.csv', 'MayerHallPV.csv', 'MESOM_PV.csv', 'OslerParkingPV.csv', 
         'PowellPV.csv', 'PriceCenterA_PV.csv', 'PriceCenterB_PV.csv', 'SDSC_PV.csv', 'SME_SolarPV.csv', 
         'StephenBirchPV.csv']

# Dizionario per memorizzare gli RMSE e le predizioni per ogni modello
rmse_results = {model: [] for model in ["Linear Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Naive"]}
predictions = {}

# Funzione per elaborare ogni file
def process_file(file_path, last_file=False):
    df = pd.read_csv(file_path)  
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.sort_values(by="DateTime", inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop("DateTime", axis=1, inplace=True)
    df.rename(columns={'RealPower': 'Present'}, inplace=True)
    df['Future'] = df['Present'].shift(-1)
    df = df.head(df.shape[0] - 1)
    x = df['Present']
    y = df['Future']
    workflow(x, y, last_file)

# Funzione principale
def workflow(x, y, last_file):
    global y_test, y_pred
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
    
    # Converto in array numpy e reshaping
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    
    # Scaling delle feature e dei target
    min_max_scaler = MinMaxScaler()
    x_train_norm = min_max_scaler.fit_transform(x_train)
    x_test_norm = min_max_scaler.transform(x_test)
    y_train_norm = min_max_scaler.transform(y_train)
    y_test_norm = min_max_scaler.transform(y_test)
    
    models = [lr, knn, dt, rf, "Naive"]
    model_names = ["Linear Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Naive"]

    for model, name in zip(models, model_names):
        if model == "Naive":
            y_pred_norm = x_test_norm.flatten()
        else:
            model.fit(x_train_norm, y_train_norm.ravel())
            y_pred_norm = model.predict(x_test_norm)
        
        # Calcolo dell'RMSE sulla scala normalizzata
        rmse = math.sqrt(mean_squared_error(y_test_norm, y_pred_norm))
        rmse_results[name].append(rmse)
        
        if last_file and (name in ["Linear Regression", "Naive"]):
            # Inversione della scalatura per le predizioni e y_test
            y_pred = min_max_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            y_test_unscaled = min_max_scaler.inverse_transform(y_test_norm).flatten()
            predictions[name] = y_pred

    if last_file:
        plot_predictions(y_test_unscaled)

# Funzione per plottare le predizioni
def plot_predictions(y_test_unscaled):
    # Plot per la Regressione Lineare
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:480], label='Valori Real - LR')
    plt.plot(predictions['Linear Regression'][:480], '--', label='Predizioni - LR')
    plt.legend()
    plt.title('Predizioni vs Valori Reali - Regressione Lineare')
    plt.xlabel('Time Steps')
    plt.ylabel('Power')
    plt.show()

    # Plot per il Modello Naive
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:480], label='Valori Real - Naive')
    plt.plot(predictions['Naive'][:480], '--', label='Predizioni - Naive')
    plt.legend()
    plt.title('Predizioni vs Valori Reali - Modello Naive')
    plt.xlabel('Time Steps')
    plt.ylabel('Power')
    plt.show()

# Elaborazione di ogni file
for file_name in files:
    file_path = folder + file_name
    last_file = (file_name == files[-1])
    print(f"Elaborazione di {file_name}")
    process_file(file_path, last_file)
    
# Calcolo e stampa dell'RMSE medio per ogni modello
for model in rmse_results:
    average_rmse = np.mean(rmse_results[model])
    print(f"RMSE medio per {model} su tutti i file: {average_rmse:.4f}")
