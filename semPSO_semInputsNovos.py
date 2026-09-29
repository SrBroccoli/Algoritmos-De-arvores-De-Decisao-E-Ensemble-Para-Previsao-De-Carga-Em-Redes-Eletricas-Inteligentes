import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

beginningOfPrediction = 110000
endOfPrediction = beginningOfPrediction + 2500

beginningOfPredictionPlot = 110000
endOfPredictionPlot = beginningOfPredictionPlot + 168

energyPredictionFilePath = 'AEP_hourly.csv'
energyData = pd.read_csv(energyPredictionFilePath) 
energyData = energyData.dropna(axis=0)


energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

trainingSampleY = energyData.loc[range(0,108600)]["AEP_MW"]
trainingSampleX = energyData.loc[range(0,108600)][["Year", "Month", "Day", "Hour"]]

for i in range(0,10):
    forestmodel = RandomForestRegressor(random_state=1, max_depth=500, max_leaf_nodes=500)
    forestmodel.fit(trainingSampleX,trainingSampleY)
    predictionforest = forestmodel.predict(energyData.loc[range(110000, 112500)][["Year", "Month", "Day", "Hour"]]) # Prediction of future values
    


#---- Plotagem de uma fatia dos resultados obtidos
plt.subplot(2,1,1)
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot], linestyle="dashed", label="Valor previsto")
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

mediaDoErroPercentual = [np.sum(np.abs((predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(endOfPredictionPlot-beginningOfPredictionPlot)]*(endOfPredictionPlot-beginningOfPredictionPlot)
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[0:endOfPredictionPlot-beginningOfPredictionPlot], label=stringMediaDoErroPercentual)
plt.legend()
plt.yticks(ticks=[-0.05, 0.1, 0.25, 0.4],labels=["-5%", "10%", "25%", "40%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()
plt.savefig("comparacaoDePrevisoes_semPSO_semInputsNovos.png", dpi=300)
plt.show()
#----