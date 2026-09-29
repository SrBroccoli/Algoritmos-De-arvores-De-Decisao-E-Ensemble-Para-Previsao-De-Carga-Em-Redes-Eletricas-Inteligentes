import random
from G_XGBoost import DecisionPrediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Integer
import skopt
from skopt.space import Real

# Carrega e remove valores nulos da base de dados de energia
energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

# Define limites para treinamento (80%) e predição
endOfTraining = int(len(energyData)*0.8)
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Converte e extrai variáveis de data e hora
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour
features = 2

# Cria defasagens baseadas na quantidade definida na variável 'features'
for i in range(1, features+1):
    energyData[f"AEP_t-{i}"] = energyData["AEP_MW"].shift(i)

# Define as colunas de entrada com base nas defasagens geradas
lag_features = [f"AEP_t-{i}" for i in range(1, features)]

trainingSampleY = energyData.loc[range(0,endOfTraining)]["AEP_MW"]

trainingSampleX = energyData.loc[range(0,endOfTraining)][
    ["Year", "Month", "Day", "Hour"] + lag_features
]

# Instancia o modelo HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=1)

# Define o espaço de busca amplo para a otimização Bayesiana de múltiplos hiperparâmetros
search_space = {
    "learning_rate": Real(0.001,0.2),
    "max_iter": Integer(20,400),
    "max_depth": Integer(2, 500),
    "max_leaf_nodes": Integer(2, 400),
    "min_samples_leaf": Integer(2, 50)
}

# Configura a busca Bayesiana com validação cruzada para séries temporais e 250 iterações
bayesSearch = skopt.BayesSearchCV(model, search_spaces=search_space, cv=TimeSeriesSplit(), scoring="neg_mean_absolute_percentage_error", random_state=1, n_iter=250)

# Executa o processo de otimização no conjunto de treinamento
bayesSearch.fit(trainingSampleX, trainingSampleY)

# Mostra os melhores parâmetros e o melhor score encontrados
print(bayesSearch.best_params_)
print(bayesSearch.best_score_)