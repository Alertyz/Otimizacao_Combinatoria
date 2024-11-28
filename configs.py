# Configurações Gerais
DEMANDA_MINIMA_TEXTO = "13, 99, 22, 28, 24, 152, 14, 2, 7, 19, 28, 8, 28, 39, 131, 16, 6, 7, 99, 3, 34, 14, 27, 24, 7, 30, 15, 9, 35, 41, 19, 59, 15, 18, 4, 4, 12, 19, 12, 5, 24, 41, 97, 5, 11, 34, 1, 99, 36, 4"
VOLUME_TEXTO = "39, 97, 80, 96, 56, 21, 48, 27, 56, 41, 64, 95, 70, 60, 98, 35, 38, 5, 73, 53, 17, 27, 17, 60, 71, 61, 52, 6, 15, 25, 65, 72, 20, 13, 33, 30, 52, 82, 62, 29, 78, 65, 56, 28, 55, 14, 78, 61, 1, 8"
ESPACO_TOTAL = 83171

DEMANDA_MINIMA = [int(i) for i in DEMANDA_MINIMA_TEXTO.split(",")]
VOLUME = [int(i) for i in VOLUME_TEXTO.split(",")]

NUM_PRODUTOS = len(DEMANDA_MINIMA)
MINIMO = min(DEMANDA_MINIMA)
MAXIMO = int(max(DEMANDA_MINIMA) + (max(DEMANDA_MINIMA) * 0.1))

# Algoritmo Genético
TAM_POP = NUM_PRODUTOS * 5
PROB_CRUZAMENTO = 0.8
PROB_MUTACAO = 0.03
GENERATIONS = 1000
PENALIDADE_FALTA = 8
PENALIDADE_EXCESSO = 8
PENALIDADE_ESPACO = 50
V_ELETISMO = 10

# Algoritmo de Colônia de Formigas (ACO)
ALFA = 1.0  # Peso do Feromônio
BETA = 2.0  # Peso da Heurística
RHO = 0.1   # Taxa de Evaporação
Q = 100     # Quantidade de Depósito de Feromônio
NUM_FORMIGAS = 50
NUM_ITERACOES = 100 
