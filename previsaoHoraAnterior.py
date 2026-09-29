# previsaoHoraAnterior.py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import numpy as np

# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
beginningOfPrediction = 110000
endOfPrediction = beginningOfPrediction + 2500

# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
beginningOfPredictionPlot = 110000
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Define o arquivo que contém os dados históricos de consumo de energia.
energyPredictionFilePath = 'AEP_hourly.csv'
# Carrega os dados do arquivo CSV para um DataFrame do pandas.
energyData = pd.read_csv(energyPredictionFilePath) 
# Remove linhas que possuem valores ausentes.
energyData = energyData.dropna(axis=0)


# Converte a coluna de data e hora para o formato datetime.
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

# Extrai informações de ano, mês, dia e hora para utilizar como variáveis de entrada.
energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour


# Realiza a previsão para o período definido.
predictionforest = energyData.loc[range(beginningOfPrediction-1, endOfPrediction-1)]["AEP_MW"]
# Realiza a previsão para o período definido.
predictionforest = predictionforest.tolist()
#---- Plotagem de uma fatia dos resultados obtidos
# Inicia a configuração dos gráficos de comparação entre previsão e valor real.
plt.subplot(2,1,1)
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot], linestyle="dashed", label="Valor previsto")
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

# Inicia a configuração dos gráficos de comparação entre previsão e valor real.
plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

# Calcula o erro percentual médio utilizado para avaliar a qualidade da previsão.
mediaDoErroPercentual = [np.sum(np.abs((predictionforest[0:endOfPredictionPlot-beginningOfPredictionPlot]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))/(endOfPredictionPlot-beginningOfPredictionPlot)]*(endOfPredictionPlot-beginningOfPredictionPlot)
# Formata o erro médio percentual para exibição.
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[0:endOfPredictionPlot-beginningOfPredictionPlot], label=stringMediaDoErroPercentual)
plt.legend()
plt.yticks(ticks=[-0.05, 0.0, 0.05, 0.1, 0.15],labels=["-5%", "0%", "5%", "10%", "15%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()
# Salva a figura gerada em arquivo PNG.
plt.savefig("comparacaoDePrevisoes_HoraAnterior.png", dpi=300)
# Exibe os gráficos na tela.
plt.show()
#----
