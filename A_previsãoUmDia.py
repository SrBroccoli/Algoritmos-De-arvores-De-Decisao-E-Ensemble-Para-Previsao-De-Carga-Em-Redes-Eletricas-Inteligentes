def DecisionPrediction(particleVector):

    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestRegressor
    import sklearn.metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Carrega e limpa os dados de consumo de energia
    energyPredictionFilePath = 'AEP_hourly.csv'
    energyData = pd.read_csv(energyPredictionFilePath) 
    energyData = energyData.dropna(axis=0)

    # Converte e extrai componentes temporais da coluna de data/hora
    energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

    energyData["Year"] = energyData["Datetime"].dt.year
    energyData["Month"] = energyData["Datetime"].dt.month
    energyData["Day"] = energyData["Datetime"].dt.day
    energyData["Hour"] = energyData["Datetime"].dt.hour

    # Cria variáveis de defasagem (lags) de 24, 25 e 168 horas
    energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
    energyData["AEP_t-25"] = energyData["AEP_MW"].shift(25)

    energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

    # Define o conjunto de dados de treinamento com base em um intervalo específico de índices
    trainingSampleY = energyData.loc[range(100000,108600)]["AEP_MW"]
    trainingSampleX = energyData.loc[range(100000,108600)][["Year", "Month", "Day", "Hour","AEP_t-24", "AEP_t-25","AEP_t-168"]]

    # Instancia e treina o modelo de Random Forest com os parâmetros do vetor de partículas
    forestmodel = RandomForestRegressor(random_state=1, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
    forestmodel.fit(trainingSampleX,trainingSampleY)
    
#--------- Calculo de MAE para PSO usando fragmentos, e não o período temporal completo
    predictionforest = [None] * 5
    predictionforestMAE = [None] * 5

    # Avalia o modelo em 5 blocos/fragmentos sequenciais para o cálculo do Erro Absoluto Médio (MAE)
    for i in range(1,6):
        predictionforest[i-1] = forestmodel.predict(energyData.loc[range(110000 + 500 * (i-1), 110000 + 500*i)][["Year", "Month", "Day", "Hour","AEP_t-24", "AEP_t-25", "AEP_t-168"]]) # Prediction of future values
        predictionforestMAE[i-1] = sklearn.metrics.mean_absolute_error(energyData.loc[range(110000 + 500 * (i-1), 110000 + 500*i)]["AEP_MW"], predictionforest[i-1])
   
    E_prediction = np.mean(predictionforestMAE)
    return E_prediction
#---------

#--------- Calculo de MAE usando PSO usando período temporal completo
    #predictionforest = forestmodel.predict(energyData.loc[range(110000, 112500)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3","AEP_t-24", "AEP_t-168"]]) # Prediction of future values
    #predictionforestMAE = sklearn.metrics.mean_absolute_error(energyData.loc[range(110000, 112500)]["AEP_MW"], predictionforest)

    #return predictionforestMAE
#---------