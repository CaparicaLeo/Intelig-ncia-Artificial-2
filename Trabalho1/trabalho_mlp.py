# 1. Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ETAPA 1: Carregar e Preparar os Dados ---
print("--- Carregando e preparando os dados ---")

# Carregue o dataset. Certifique-se que o arquivo 'StressLevelDataset.csv' está na mesma pasta.
try:
    dados = pd.read_csv('StressLevelDataset.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'StressLevelDataset.csv' não encontrado.")
    print("Por favor, faça o download do dataset do Kaggle e coloque na mesma pasta do script.")
    exit()

# Visualização inicial dos dados
print("Primeiras linhas do dataset:")
print(dados.head())
print("\nInformações do dataset:")
dados.info()

# --- ETAPA 2: Pré-processamento ---
print("\n--- Pré-processando os dados ---")

# a) Separar as características (X) e o alvo (y)
# Define qual é a nossa coluna alvo (o que queremos prever)
alvo = 'stress_level'

# Pega TODAS as colunas do DataFrame e remove a coluna 'alvo' para criar a lista de características
# Isso torna o código mais flexível e adaptado ao seu arquivo!
caracteristicas = dados.columns.drop(alvo) 

print(f"\nCaracterísticas utilizadas para o modelo: {list(caracteristicas)}")
print(f"Alvo do modelo: {alvo}")

X = dados[caracteristicas]
y = dados[alvo]

# b) Codificar a variável alvo (y)
# A rede neural precisa que a saída seja numérica.
# O LabelEncoder transforma 'Low/Normal', 'Medium', 'High' em 0, 1, 2.
codificador = LabelEncoder()
y_codificado = codificador.fit_transform(y)

print(f"\nClasses originais: {codificador.classes_}")
print(f"Classes codificadas: {codificador.transform(codificador.classes_)}")

# --- ETAPA 3: Divisão em Dados de Treino e Teste ---
# Dividimos os dados para treinar o modelo e depois testá-lo em dados que ele nunca viu.
# 80% para treino, 20% para teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_codificado, test_size=0.2, random_state=42, stratify=y_codificado)

print(f"\nTamanho do conjunto de treino: {X_treino.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_teste.shape[0]} amostras")

# d) Normalização/Escalonamento dos dados
# Isso é MUITO importante para o bom funcionamento da MLP.
# Garante que todas as características tenham a mesma escala.
normalizador = StandardScaler()
X_treino_normalizado = normalizador.fit_transform(X_treino)
X_teste_normalizado = normalizador.transform(X_teste)


# --- ETAPA 4: Implementação e Treinamento da MLP ---
# Esta é a parte central, onde criamos a rede neural.
# Vamos criar uma função para facilitar a execução dos experimentos.

def treinar_e_avaliar_mlp(tamanhos_camadas_ocultas, taxa_aprendizado, X_treino, y_treino, X_teste, y_teste):
    """
    Função para criar, treinar e avaliar um modelo MLP com parâmetros específicos.
    """
    print(f"\n--- Treinando com Configuração: Camadas={tamanhos_camadas_ocultas}, Taxa Aprendizado={taxa_aprendizado} ---")

    # Criação do modelo MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=tamanhos_camadas_ocultas, # N° de neurônios e camadas. Ex: (10,) para 1 camada com 10 neurônios.
        learning_rate_init=taxa_aprendizado,         # Taxa de aprendizado.
        max_iter=1000,                               # Número máximo de épocas de treinamento.
        activation='relu',                           # Função de ativação para as camadas ocultas.
        solver='adam',                               # Otimizador que usa o backpropagation.
        random_state=42                              # Para reprodutibilidade dos resultados.
    )

    # Treinamento do modelo
    mlp.fit(X_treino, y_treino)

    # Previsões nos dados de treino e teste
    y_pred_treino = mlp.predict(X_treino)
    y_pred_teste = mlp.predict(X_teste)

    # Cálculo da acurácia
    acuracia_treino = accuracy_score(y_treino, y_pred_treino)
    acuracia_teste = accuracy_score(y_teste, y_pred_teste)

    print(f"Acurácia no Treino: {acuracia_treino:.4f}")
    print(f"Acurácia no Teste: {acuracia_teste:.4f}")

    return acuracia_treino, acuracia_teste, mlp

# --- ETAPA 5: Execução dos Experimentos ---
print("\n\n--- INICIANDO EXPERIMENTOS ---")

# Experimento 1: Variando o número de neurônios em uma camada
print("\n\n===== EXPERIMENTO 1: Variando N° de Neurônios (1 camada) =====")
configs_neuronios = [8, 16, 32, 64, 128]
resultados_exp1 = []
for n_neuronios in configs_neuronios:
    acuracia_treino, acuracia_teste, _ = treinar_e_avaliar_mlp(
        tamanhos_camadas_ocultas=(n_neuronios,), # A vírgula é importante para indicar uma tupla de 1 elemento
        taxa_aprendizado=0.001,
        X_treino=X_treino_normalizado,
        y_treino=y_treino,
        X_teste=X_teste_normalizado,
        y_teste=y_teste
    )
    resultados_exp1.append({'config': f'({n_neuronios},)', 'acuracia_treino': acuracia_treino, 'acuracia_teste': acuracia_teste})

# Experimento 2: Variando o número de camadas intermediárias
print("\n\n===== EXPERIMENTO 2: Variando N° de Camadas (32 neurônios/camada) =====")
configs_camadas = [(32,), (32, 32), (32, 32, 32)] # 1, 2 e 3 camadas
resultados_exp2 = []
for config_camada in configs_camadas:
    acuracia_treino, acuracia_teste, _ = treinar_e_avaliar_mlp(
        tamanhos_camadas_ocultas=config_camada,
        taxa_aprendizado=0.001,
        X_treino=X_treino_normalizado,
        y_treino=y_treino,
        X_teste=X_teste_normalizado,
        y_teste=y_teste
    )
    resultados_exp2.append({'config': str(config_camada), 'acuracia_treino': acuracia_treino, 'acuracia_teste': acuracia_teste})

# Experimento 3: Variando a taxa de aprendizado
print("\n\n===== EXPERIMENTO 3: Variando Taxa de Aprendizado (melhor arquitetura) =====")
# Vamos assumir que a melhor arquitetura foi (32, 32) do experimento anterior.
taxas_aprendizado = [0.2, 0.01, 0.001, 0.0001]
resultados_exp3 = []
melhor_arquitetura = (32, 32)
for taxa in taxas_aprendizado:
    acuracia_treino, acuracia_teste, _ = treinar_e_avaliar_mlp(
        tamanhos_camadas_ocultas=melhor_arquitetura,
        taxa_aprendizado=taxa,
        X_treino=X_treino_normalizado,
        y_treino=y_treino,
        X_teste=X_teste_normalizado,
        y_teste=y_teste
    )
    resultados_exp3.append({'config': taxa, 'acuracia_treino': acuracia_treino, 'acuracia_teste': acuracia_teste})


# --- ETAPA 6: Apresentar Resultados Finais e Análise do Melhor Modelo ---
print("\n\n--- RESULTADOS FINAIS ---")

# Imprimir as tabelas de resultados para o artigo
print("\nResultados Experimento 1 (Neurônios):")
df_exp1 = pd.DataFrame(resultados_exp1)
print(df_exp1)

print("\nResultados Experimento 2 (Camadas):")
df_exp2 = pd.DataFrame(resultados_exp2)
print(df_exp2)

print("\nResultados Experimento 3 (Taxa de Aprendizado):")
df_exp3 = pd.DataFrame(resultados_exp3)
print(df_exp3)

# Treinar o melhor modelo final para análise
print("\n\n--- Análise do Melhor Modelo (Ex: 2 camadas com 32 neurônios e taxa 0.001) ---")
_, _, melhor_modelo = treinar_e_avaliar_mlp(
    tamanhos_camadas_ocultas=(32, 32),
    taxa_aprendizado=0.001,
    X_treino=X_treino_normalizado,
    y_treino=y_treino,
    X_teste=X_teste_normalizado,
    y_teste=y_teste
)

# Matriz de Confusão: ajuda a ver onde o modelo está errando
y_pred_final = melhor_modelo.predict(X_teste_normalizado)
matriz_confusao = confusion_matrix(y_teste, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=codificador.classes_, yticklabels=codificador.classes_)
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão do Melhor Modelo')
plt.show() # Mostra o gráfico