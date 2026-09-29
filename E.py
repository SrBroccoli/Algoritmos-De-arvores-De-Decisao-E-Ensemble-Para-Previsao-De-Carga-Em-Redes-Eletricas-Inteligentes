# E(1).py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Função usada pelo PSO para avaliar uma combinação de hiperparâmetros do modelo.
def DecisionPrediction(particleVector):

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestRegressor
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
    beginningOfTraining = 0
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
# Cria variáveis defasadas (lags) da demanda para representar valores passados do consumo.
    for i in range(1, 24):
        energyData[f"AEP_t-{i}"] = energyData["AEP_MW"].shift(i)

    # Mantendo também os lags maiores já existentes
# Cria o lag de 24 horas, permitindo utilizar o consumo do mesmo horário do dia anterior.
    energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)
# Cria o lag de 168 horas, correspondente ao mesmo horário da semana anterior.
    energyData["AEP_t-168"] = energyData["AEP_MW"].shift(168)

    # Definindo as colunas de entrada
# Lista as variáveis defasadas que serão utilizadas como entradas do modelo.
    lag_features = [f"AEP_t-{i}" for i in range(1, 24)] + ["AEP_t-24", "AEP_t-168"]

# Define a variável alvo do treinamento: a demanda real em MW.
    trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]

# Define as variáveis de entrada utilizadas para treinar o modelo.
    trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][
        ["Year", "Month", "Day", "Hour"] + lag_features
    ]

    
# Cria o modelo Random Forest usando os valores da partícula como hiperparâmetros.
    forestmodel = RandomForestRegressor(random_state=0, max_depth=particleVector[0], max_leaf_nodes=particleVector[1])
# Treina o modelo utilizando os dados de entrada e os valores reais de demanda.
    forestmodel.fit(trainingSampleX,trainingSampleY)
    
#--------- Calculo de MAE para PSO usando fragmentos, e não o período temporal completo
    
# Realiza a previsão para o período definido.
    predictionforest = forestmodel.predict(energyData.loc[range(beginningOfPrediction, endOfPrediction), ["Year", "Month", "Day", "Hour"] + lag_features]) # Prediction of future values
# Calcula o erro percentual absoluto médio (MAPE) entre os valores reais e previstos.
    predictionforestMAPE = sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction), "AEP_MW"], predictionforest)
   
# Define o valor do erro que será retornado ao PSO como função objetivo.
    E_prediction = np.mean(predictionforestMAPE)
# Retorna o erro calculado para a combinação de hiperparâmetros avaliada.
    return E_prediction
#---------

#--------- Calculo de MAE usando PSO usando período temporal completo
    #predictionforest = forestmodel.predict(energyData.loc[range(110000, 112500)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3","AEP_t-24", "AEP_t-168"]]) # Prediction of future values
# Calcula o erro absoluto médio (MAE) para o trecho avaliado.
    #predictionforestMAE = sklearn.metrics.mean_absolute_error(energyData.loc[range(110000, 112500)]["AEP_MW"], predictionforest)

    #return predictionforestMAE
#---------

