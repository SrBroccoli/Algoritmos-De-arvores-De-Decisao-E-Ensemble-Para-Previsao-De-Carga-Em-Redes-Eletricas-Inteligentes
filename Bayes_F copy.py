import random
from F import DecisionPrediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Integer
import skopt

energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

endOfTraining = int(len(energyData)*0.8)
print(endOfTraining)
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour
# Criando lags de 1 até 12 horas
for i in range(1, 24):
    energyData[f"AEP_t-{i}"] = energyData["AEP_MW"].shift(i)

# Mantendo também os lags maiores já existentes
energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

# Definindo as colunas de entrada
lag_features = [f"AEP_t-{i}" for i in range(1, 24)] + ["AEP_t-24", "AEP_t-168"]

trainingSampleY = energyData.loc[range(70000,endOfTraining)]["AEP_MW"]

trainingSampleX = energyData.loc[range(70000,endOfTraining)][
    ["Year", "Month", "Day", "Hour"] + lag_features
]


model = RandomForestRegressor()

search_space = {
    "n_estimators": Integer(50,100),
    "max_depth": Integer(500, 1000),
    "max_leaf_nodes": Integer(500, 1000)
}

bayesSearch = skopt.BayesSearchCV(model, search_spaces=search_space, cv=TimeSeriesSplit(), scoring="neg_mean_absolute_percentage_error", random_state=1, n_iter=10)

bayesSearch.fit(trainingSampleX, trainingSampleY)

print(bayesSearch.best_params_)
print(bayesSearch.best_score_)