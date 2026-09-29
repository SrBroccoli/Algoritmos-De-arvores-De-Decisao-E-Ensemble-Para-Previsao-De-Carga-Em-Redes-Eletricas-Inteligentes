import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carrega a base de dados horários e remove nulos
energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

# Define índices para particionar treino (80%) e teste (20%)
beginningOfTraining = 0
endOfTraining = int(len(energyData)*0.8)
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

# Janela de plotagem de 1 semana
beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Conversão de datas e extração de atributos de tempo
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

# Criação das variáveis de atraso t-1 e t-2
energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)
energyData["AEP_t-2"] = energyData["AEP_MW"].shift(2) # Adicional em relação ao A <------

# Amostras de treino X e Y
trainingSampleY = energyData.loc[range(beginningOfTraining,endOfTraining)]["AEP_MW"]
trainingSampleX = energyData.loc[range(beginningOfTraining,endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2"]]

forestmodel = []
predictionforest = []
    
# Executa um ensemble com 10 modelos Random Forest variando a semente (random_state) para mitigar variabilidade
for i in range(0,10):
    forestmodel.append(RandomForestRegressor(random_state=i*10, max_depth=333, max_leaf_nodes=51000))
    forestmodel[i].fit(trainingSampleX,trainingSampleY)
    predictionforest.append(forestmodel[i].predict(energyData.loc[range(beginningOfPrediction, endOfPrediction)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2"]])) # Prediction of future values

# Calcula a média das predições de todos os modelos do ensemble
predictionforestmean = np.mean(predictionforest, axis=0)

#---- Plotagem de uma fatia dos resultados obtidos
# Gráfico superior: Comparação visual entre o consumo real e a média prevista
plt.subplot(2,1,1)
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction], linestyle="dashed", label="Valor previsto")
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

# Gráfico inferior: Erro percentual do modelo em relação aos dados reais
plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

mediaDoErroPercentual = [sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction)]["AEP_MW"], predictionforestmean[0:endOfPrediction-beginningOfPrediction])]*(endOfPrediction-beginningOfPrediction)
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction], label=stringMediaDoErroPercentual)
plt.yticks(ticks=[-0.05, 0, 0.05, 0.1],labels=["-5%", "0%", "5%", "10%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("comparacaoDePrevisoes_B_treinamento15Anos.png", dpi=300)
plt.show()
#----