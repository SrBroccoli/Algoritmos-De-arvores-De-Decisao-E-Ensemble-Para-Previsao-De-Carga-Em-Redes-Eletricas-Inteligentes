import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carrega o arquivo de dados horários e remove dados inconsistentes/nulos
energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

# Delimita os blocos de treinamento e teste
beginningOfTraining = 0
endOfTraining = int(len(energyData)*0.8)
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

# Janela de visualização gráfica de 168 horas (1 semana)
beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Conversão e extração de propriedades de data e hora
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

# Criação de defasagens horarias até t-3
energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)
energyData["AEP_t-2"] = energyData["AEP_MW"].shift(2) # Adicional em relação ao A <------
energyData["AEP_t-3"] = energyData["AEP_MW"].shift(3) # Adicional em relação ao B <------

# Amostras de treinamento utilizando as variáveis de entrada expandidas
trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]
trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3"]]

    
forestmodel = []
predictionforest = []
    
# Criação de um comitê (ensemble) de 10 árvores de regressão com sementes e hiperparâmetros fixos
for i in range(0,10):
    forestmodel.append(RandomForestRegressor(random_state=i*10, max_depth=179, max_leaf_nodes=36000))
    forestmodel[i].fit(trainingSampleX,trainingSampleY)
    predictionforest.append(forestmodel[i].predict(energyData.loc[range(beginningOfPrediction, endOfPrediction)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2", "AEP_t-3"]])) # Prediction of future values

# Calcula a média das previsões obtidas pelo ensemble
predictionforestmean = np.mean(predictionforest, axis=0)

#---- Plotagem de uma fatia dos resultados obtidos
# Gráfico superior: Compara visualmente os valores previstos (média) com os reais
plt.subplot(2,1,1)
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction], linestyle="dashed", label="Valor previsto")
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

# Gráfico inferior: Mostra o comportamento do erro percentual ao longo da semana testada
plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

mediaDoErroPercentual = [sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction)]["AEP_MW"], predictionforestmean[0:endOfPrediction-beginningOfPrediction])]*(endOfPrediction-beginningOfPrediction)
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[0:endOfPredictionPlot-beginningOfPredictionPlot], label=stringMediaDoErroPercentual)
plt.yticks(ticks=[-0.05, 0, 0.05, 0.1],labels=["-5%", "0%", "5%", "10%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("comparacaoDePrevisoes_C_treinamento15Anos.png", dpi=300)
plt.show()
#----