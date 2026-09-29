def DecisionPrediction(particleVector):

    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestRegressor
    import sklearn.metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Carrega e trata dados ausentes do arquivo CSV
    energyPredictionFilePath = 'AEP_hourly.csv'
    energyData = pd.read_csv(energyPredictionFilePath) 
    energyData = energyData.dropna(axis=0)

    # Define os limites de treino (80%) e teste/predição (20%)
    beginningOfTraining = 0
    endOfTraining = int(len(energyData)*0.8)
    beginningOfPrediction = endOfTraining + 1
    endOfPrediction = len(energyData)

    # Define intervalo de plotagem de 1 semana
    beginningOfPredictionPlot = beginningOfPrediction
    endOfPredictionPlot = beginningOfPredictionPlot + 168

    # Converte e extrai campos temporais do datetime
    energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

    energyData["Year"] = energyData["Datetime"].dt.year
    energyData["Month"] = energyData["Datetime"].dt.month
    energyData["Day"] = energyData["Datetime"].dt.day
    energyData["Hour"] = energyData["Datetime"].dt.hour

    # Cria múltiplas defasagens temporais (lags), incluindo t-2 em relação à versão A
    energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)
    energyData["AEP_t-2"] = energyData["AEP_MW"].shift(2) # Adicional em relação ao A <------
    energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
    energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

    # Seleciona os dados de treinamento para Y e X (com o lag adicional t-2)
    trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]
    trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2","AEP_t-24", "AEP_t-168"]]

    # Inicializa e treina o modelo de Random Forest com os parâmetros recebidos do otimizador
    forestmodel = RandomForestRegressor(random_state=1, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
    forestmodel.fit(trainingSampleX,trainingSampleY)
    
#--------- Calculo de MAE para PSO usando fragmentos, e não o período temporal completo

    # Faz previsões para o conjunto de predição e calcula o erro MAPE médio
    predictionforest = forestmodel.predict(energyData.loc[range(beginningOfPrediction, endOfPrediction), ["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2","AEP_t-24", "AEP_t-168"]]) # Prediction of future values
    predictionforestMAPE = sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction), "AEP_MW"], predictionforest)
   
    E_prediction = np.mean(predictionforestMAPE)
    return E_prediction
#---------