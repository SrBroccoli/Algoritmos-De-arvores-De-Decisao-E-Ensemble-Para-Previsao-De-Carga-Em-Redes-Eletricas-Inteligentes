import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define os intervalos de início e fim para a predição e para a plotagem gráfica
beginningOfPrediction = 110000
endOfPrediction = beginningOfPrediction + 2500

beginningOfPredictionPlot = 110000
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Carrega e limpa a base de dados horários de energia
energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)

# Converte datas e extrai atributos de tempo
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

# Cria a defasagem de 24 horas
energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)

# Define as amostras de treino (alvo e preditores)
trainingSampleY = energyData.loc[range(0,108600)]["AEP_MW"]
trainingSampleX = energyData.loc[range(0,108600)][["Year", "Month", "Day", "Hour", "AEP_t-24", "AEP_t-25","AEP_t-168"]]

    
# Instancia e ajusta o modelo Random Forest com hiperparâmetros fixos
forestmodel = RandomForestRegressor(random_state=1, max_depth=1018, max_leaf_nodes=65)
forestmodel.fit(trainingSampleX,trainingSampleY)
    
# Executa a predição para o período estipulado
predictionforest = forestmodel.predict(energyData.loc[range(beginningOfPrediction, endOfPrediction)][["Year", "Month", "Day", "Hour","AEP_t-24", "AEP_t-25","AEP_t-168"]]) # Prediction of future values

#---- Plotagem de uma fatia dos resultados obtidos
# Subgráfico superior: Comparação visual entre a curva prevista e a real (em MW)
plt.subplot(2,1,1)
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot], linestyle="dashed", label="Valor previsto")
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

# Subgráfico inferior: Erro percentual ao longo do tempo e cálculo da métrica média
plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

mediaDoErroPercentual = [np.sum(np.abs((predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(endOfPredictionPlot-beginningOfPredictionPlot)]*(endOfPredictionPlot-beginningOfPredictionPlot)
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[0:endOfPredictionPlot-beginningOfPredictionPlot], label=stringMediaDoErroPercentual)
plt.yticks(ticks=[-0.05, 0, 0.05, 0.1],labels=["-5%", "0%", "5%", "10%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("comparacaoDePrevisoes_A_previsaoUmDia.png", dpi=300)
plt.show()
#----