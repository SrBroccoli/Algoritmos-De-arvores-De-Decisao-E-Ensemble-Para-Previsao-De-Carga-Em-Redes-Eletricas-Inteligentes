def DecisionPrediction(particleVector):

    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestRegressor
    import sklearn.metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Carrega os dados de consumo de energia por hora e remove valores nulos
    energyPredictionFilePath = 'AEP_hourly.csv'
    energyData = pd.read_csv(energyPredictionFilePath) 
    energyData = energyData.dropna(axis=0)

    # Define os limites dos conjuntos de treinamento (80%) e predição (20%)
    beginningOfTraining = 0
    endOfTraining = int(len(energyData)*0.8)
    beginningOfPrediction = endOfTraining + 1
    endOfPrediction = len(energyData)

    # Define o intervalo para a plotagem dos resultados (1 semana / 168 horas)
    beginningOfPredictionPlot = beginningOfPrediction
    endOfPredictionPlot = beginningOfPredictionPlot + 168

    # Converte a coluna de data/hora para o formato datetime do pandas
    energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

    # Extrai características temporais (Ano, Mês, Dia, Hora) para o modelo
    energyData["Year"] = energyData["Datetime"].dt.year
    energyData["Month"] = energyData["Datetime"].dt.month
    energyData["Day"] = energyData["Datetime"].dt.day
    energyData["Hour"] = energyData["Datetime"].dt.hour

    # Cria defasagem (lag) de 1 hora para a série temporal
    energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)

    # Separa os dados de treino em variáveis alvo (Y) e preditoras (X)
    trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]
    trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-24", "AEP_t-168"]]

    # Configura e treina o modelo Random Forest com hiperparâmetros vindos do vetor de partículas (PSO)
    forestmodel = RandomForestRegressor(random_state=1, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
    forestmodel.fit(trainingSampleX,trainingSampleY)

    # Realiza a predição para o período de teste e calcula o Erro Percentual Absoluto Médio (MAPE)
    predictionforest = forestmodel.predict(energyData.loc[range(beginningOfPrediction, endOfPrediction), ["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-24", "AEP_t-168"]]) # Prediction of future values
    predictionforestMAPE = sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction), "AEP_MW"], predictionforest)
   
    # Retorna a média do erro MAPE como função objetivo para a otimização
    E_prediction = np.mean(predictionforestMAPE)

    return E_prediction