import random
from F import DecisionPrediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit # (import mantido conforme original)
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Integer
import skopt

# Carrega e limpa os dados do arquivo CSV
energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

# Define o ponto de corte para divisão de treino (80%) e teste (20%)
endOfTraining = int(len(energyData)*0.8)
print(endOfTraining)
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Converte e extrai informações temporais
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

# Cria lags horários de 1 até 23 horas
for i in range(1, 24):
    energyData[f"AEP_t-{i}"] = energyData["AEP_MW"].shift(i)

# Mantém também os lags maiores já existentes (24h e 168h)
energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

# Agrupa todas as features de defasagem criadas em uma lista
lag_features = [f"AEP_t-{i}" for i in range(1, 24)] + ["AEP_t-24", "AEP_t-168"]

# Define a amostra alvo (Y) e de entrada (X) para a otimização
trainingSampleY = energyData.loc[range(0,endOfTraining)]["AEP_MW"]

trainingSampleX = energyData.loc[range(0,endOfTraining)][
    ["Year", "Month", "Day", "Hour"] + lag_features
]

# Inicializa o modelo base de Regressão por Random Forest
model = RandomForestRegressor()

# Define o espaço de busca para os hiperparâmetros na otimização Bayesiana
search_space = {
    #"n_estimators": Integer(50,300),
    "max_depth": Integer(500, 50000),
    "max_leaf_nodes": Integer(500, 20000)
}

# Configura a busca Bayesiana utilizando validação cruzada para séries temporais (TimeSeriesSplit)
bayesSearch = skopt.BayesSearchCV(model, search_spaces=search_space, cv=TimeSeriesSplit(), scoring="neg_mean_absolute_percentage_error", random_state=1)

# Executa a busca pelos melhores hiperparâmetros
bayesSearch.fit(trainingSampleX, trainingSampleY)

# Exibe os melhores parâmetros encontrados e a pontuação obtida
print(bayesSearch.best_params_)
print(bayesSearch.best_score_)