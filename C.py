def DecisionPrediction(particleVector):

    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestRegressor
    import sklearn.metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Carrega e limpa a base de dados de energia por hora
    energyPredictionFilePath = 'AEP_hourly.csv'
    energyData = pd.read_csv(energyPredictionFilePath) 
    energyData = energyData.dropna(axis=0)

    # Define os índices para treino (80%) e teste
    beginningOfTraining = 0
    endOfTraining = int(len(energyData)*0.8)
    beginningOfPrediction = endOfTraining + 1
    endOfPrediction = len(energyData)

    beginningOfPredictionPlot = beginningOfPrediction
    endOfPredictionPlot = beginningOfPredictionPlot + 168

    # Tratamento da coluna de data/hora
    energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

    energyData["Year"] = energyData["Datetime"].dt.year
    energyData["Month"] = energyData["Datetime"].dt.month
    energyData["Day"] = energyData["Datetime"].dt.day
    energyData["Hour"] = energyData["Datetime"].dt.hour

    # Cria defasagens temporais, incluindo até t-3 em relação à versão B
    energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)
    energyData["AEP_t-2"] = energyData["AEP_MW"].shift(2) # Adicional em relação ao A <------
    energyData["AEP_t-3"] = energyData["AEP_MW"].shift(3) # Adicional em relação ao B <------
    energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
    energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

    # Separação das amostras de treinamento contendo as novas defasagens
    trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]
    trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3","AEP_t-24", "AEP_t-168"]]

    # Configura e ajusta o modelo Random Forest usando parâmetros do vetor de partículas
    forestmodel = RandomForestRegressor(random_state=1, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
    forestmodel.fit(trainingSampleX,trainingSampleY)
    
#--------- Calculo de MAE para PSO usando fragmentos, e não o período temporal completo

    # Realiza a predição e calcula o erro MAPE médio para retornar ao algoritmo de otimização
    predictionforest = forestmodel.predict(energyData.loc[range(beginningOfPrediction, endOfPrediction), ["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3","AEP_t-24", "AEP_t-168"]]) # Prediction of future values
    predictionforestMAPE = sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction), "AEP_MW"], predictionforest)
   
    E_prediction = np.mean(predictionforestMAPE)
    return E_prediction
#---------