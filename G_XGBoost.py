# G_XGBoost.py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Função usada pelo PSO para avaliar uma combinação de hiperparâmetros do modelo.
def DecisionPrediction(particleVector):

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
    import sklearn.tree as sktree
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn.metrics
    import pandas as pd
    import matplotlib.pyplot as plt
# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
    import numpy as np

# Define o arquivo que contém os dados históricos de consumo de energia.
    energyPredictionFilePath = 'AEP_hourly.csv'
# Carrega os dados do arquivo CSV para um DataFrame do pandas.
    energyData = pd.read_csv(energyPredictionFilePath) 
# Remove linhas que possuem valores ausentes.
    energyData = energyData.dropna(axis=0)

# Define os índices que delimitam os conjuntos de treinamento e de previsão.
    beginningOfTraining = int(len(energyData)*0.6)
# Define os índices que delimitam os conjuntos de treinamento e de previsão.
    endOfTraining = int(len(energyData)*0.8)
# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
    beginningOfPrediction = endOfTraining + 1
    endOfPrediction = len(energyData)

# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
    beginningOfPredictionPlot = beginningOfPrediction
    endOfPredictionPlot = beginningOfPredictionPlot + 168

# Converte a coluna de data e hora para o formato datetime.
    energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

# Extrai informações de ano, mês, dia e hora para utilizar como variáveis de entrada.
    energyData["Year"] = energyData["Datetime"].dt.year
    energyData["Month"] = energyData["Datetime"].dt.month
    energyData["Day"] = energyData["Datetime"].dt.day
    energyData["Hour"] = energyData["Datetime"].dt.hour
    # Criando lags de 1 até 12 horas
    features = 2
# Cria variáveis defasadas (lags) da demanda para representar valores passados do consumo.
    for i in range(1, features + 1):
        energyData[f"AEP_t-{i}"] = energyData["AEP_MW"].shift(i)

    # Mantendo também os lags maiores já existentes


    # Definindo as colunas de entrada
# Lista as variáveis defasadas que serão utilizadas como entradas do modelo.
    lag_features = [f"AEP_t-{i}" for i in range(1, features)]

# Define a variável alvo do treinamento: a demanda real em MW.
    trainingSampleY = energyData.loc[range(beginningOfTraining,endOfTraining)]["AEP_MW"]

# Define as variáveis de entrada utilizadas para treinar o modelo.
    trainingSampleX = energyData.loc[range(beginningOfTraining,endOfTraining)][
        ["Year", "Month", "Day", "Hour"] + lag_features
    ]

    
# Cria o modelo de Gradient Boosting com os hiperparâmetros definidos ou fornecidos pelo PSO.
    XGBoostModel = GradientBoostingRegressor(random_state=1, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
# Treina o modelo utilizando os dados de entrada e os valores reais de demanda.
    XGBoostModel.fit(trainingSampleX,trainingSampleY)
    
#--------- Calculo de MAE para PSO usando fragmentos, e não o período temporal completo
    #predictionXGBoost = [None] * 5
    #predictionXGBoostMAE = [None] * 5

# Cria uma cópia dos dados para atualizar os lags com valores previstos durante a previsão recursiva.
    copyOfEnergyData = energyData.copy()

    # Previsão
# Realiza a previsão para o período definido.
    predictionXGBoost= [None] * (endOfPrediction-beginningOfPrediction)
    predictionXGBoostMAE= [None] * (endOfPrediction-beginningOfPrediction)
    for i in range (0,endOfPrediction-beginningOfPrediction):

        predictionXGBoost[i] = XGBoostModel.predict(copyOfEnergyData.loc[[beginningOfPrediction+i], ["Year", "Month", "Day", "Hour"] + lag_features])
        for k in range(0, len(lag_features)):
            copyOfEnergyData.loc[[beginningOfPrediction+i+k], lag_features[k]] = predictionXGBoost[i]

# Calcula o erro absoluto médio das previsões obtidas.
    predictionXGBoostMAE = sklearn.metrics.mean_absolute_error(copyOfEnergyData.loc[range(beginningOfPrediction, endOfPrediction)],predictionXGBoost)
# Define o valor do erro que será retornado ao PSO como função objetivo.
    E_prediction = predictionXGBoostMAE
# Retorna o erro calculado para a combinação de hiperparâmetros avaliada.
    return E_prediction
#---------

#--------- Calculo de MAE usando PSO usando período temporal completo
    #predictionXGBoost = XGBoostModel.predict(energyData.loc[range(110000, 112500)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3","AEP_t-24", "AEP_t-168"]]) # Prediction of future values
# Calcula o erro absoluto médio das previsões obtidas.
    #predictionXGBoostMAE = sklearn.metrics.mean_absolute_error(energyData.loc[range(110000, 112500)]["AEP_MW"], predictionXGBoost)

    #return predictionXGBoostMAE
#---------

