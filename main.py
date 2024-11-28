"""
LEIA O REAME.md PARA SABER COMO RODAR
GITHUB: https://github.com/Alertyz/Otimizacao_Combinatoria

"""
import streamlit as st
import numpy as np
import random
from decimal import Decimal
import matplotlib.pyplot as plt
import logging
import time
import pandas as pd
import plotly.graph_objects as go

# Importar configurações do configs.py
import configs

logging.basicConfig(level=logging.INFO)
logging.info("Iniciando execução do aplicativo.")

# Função principal do aplicativo
def main():
    st.title("Algoritmos de Otimização para Estoque")
    st.write("Selecione um algoritmo para otimização de estoque.")

    # Seleção do algoritmo
    algoritmo = st.sidebar.selectbox("Selecione o Algoritmo", ["Algoritmo Genético", "Algoritmo de Colônia de Formigas"])

    if algoritmo == "Algoritmo Genético":
        executar_algoritmo_genetico()
    elif algoritmo == "Algoritmo de Colônia de Formigas":
        executar_algoritmo_colonia_formigas()

# Função para executar o Algoritmo Genético
def executar_algoritmo_genetico():
    # Parâmetros
    st.sidebar.title("Configurações")
    demanda_minima = st.sidebar.text_area("Demanda Mínima", value=str(configs.DEMANDA_MINIMA_TEXTO))
    DEMANDA_MINIMA = [int(x) for x in demanda_minima.split(",")]
    volume = st.sidebar.text_area("Volume", value=str(configs.VOLUME_TEXTO))
    VOLUME = [int(x) for x in volume.split(",")]

    NUM_PRODUTOS = len(DEMANDA_MINIMA)
    MINIMO = min(DEMANDA_MINIMA)
    MAXIMO = int(max(DEMANDA_MINIMA) + (max(DEMANDA_MINIMA) * 0.1))

    tam_pop = st.sidebar.slider("Tamanho da População", min_value=10, max_value=500, value=configs.TAM_POP)
    prob_cruzamento = st.sidebar.slider("Probabilidade de Cruzamento", min_value=0.1, max_value=0.9, value=configs.PROB_CRUZAMENTO)
    prob_mutacao = st.sidebar.slider("Probabilidade de Mutação", min_value=0.01, max_value=0.5, value=configs.PROB_MUTACAO)
    generations = st.sidebar.slider("Número de Gerações", min_value=100, max_value=5000, value=configs.GENERATIONS)
    espaco = st.sidebar.slider("Espaço Total", min_value=50000, max_value=1000000, value=configs.ESPACO_TOTAL)
    penalidade_falta = st.sidebar.slider("Penalidade Falta", min_value=1, max_value=10, value=configs.PENALIDADE_FALTA)
    penalidade_excesso = st.sidebar.slider("Penalidade Excesso", min_value=1, max_value=10, value=configs.PENALIDADE_EXCESSO)
    penalidade_espaco_exponencial = st.sidebar.slider("Penalidade Espaço Exponencial", min_value=1, max_value=3, value=0)
    penalidade_espaco = st.sidebar.slider("Penalidade Espaço", min_value=1, max_value=100, value=configs.PENALIDADE_ESPACO)

    PENALIDADE_FALTA = penalidade_falta
    PENALIDADE_ESPACO_EXPONENCIAL = penalidade_espaco_exponencial
    PENALIDADE_ESPACO = penalidade_espaco
    PENALIDADE_EXCESSO = penalidade_excesso
    ESPACO_TOTAL = espaco
    PROB_MUTACAO = prob_mutacao
    elitismo_val = st.sidebar.slider("Elitismo", min_value=1, max_value=20, value=configs.V_ELETISMO)
    
    if st.sidebar.button("Executar Algoritmo Genético"):

        class cromossomo:
        
            def __init__(self, NUM_PRODUTOS, minimo, maximo, DEMANDA_MINIMA, cromossomo=None):
                if cromossomo is None:
                    self.cromossomo = self.gen_populacao(NUM_PRODUTOS, minimo, maximo)
                else:
                    self.cromossomo = np.array(cromossomo)  # Garante que é um array NumPy

                self.fitnesses_falta = self.calcular_fitness_falta(DEMANDA_MINIMA) 
                self.fitnesses_excessos = self.calcular_fitness_excessos(DEMANDA_MINIMA)
                self.fitnesses_espaco = self.calcular_fitness_espaco(ESPACO_TOTAL, VOLUME)
                self.fitnesses = self.calcular_fitness_total(PENALIDADE_FALTA, PENALIDADE_ESPACO, PENALIDADE_EXCESSO)
                self.volume = self.calcular_volume(VOLUME)

            def calcular_volume(self, VOLUME):
                return np.sum(VOLUME * self.cromossomo)


            def gen_populacao(self, NUM_PRODUTOS, minimo, maximo):
                # Gera diretamente um array NumPy
                return np.random.randint(1, 1000, NUM_PRODUTOS)
            
            def calcular_fitness_falta(self, DEMANDA_MINIMA):
                penalidade = 0
                for d, q in zip(DEMANDA_MINIMA, self.cromossomo):
                    if q < d:
                        penalidade += abs(d - q)

                penalidade = penalidade / sum(DEMANDA_MINIMA)
                
                return Decimal(penalidade) 

            def calcular_fitness_excessos(self, DEMANDA_MINIMA):
                penalidade = 0
                for d, q in zip(DEMANDA_MINIMA, self.cromossomo):
                    if q > d:
                        penalidade += abs(d - q)

                penalidade = penalidade / sum(DEMANDA_MINIMA)
                
                return Decimal(penalidade) 

            def calcular_fitness_espaco(self, ESPACO_TOTAL, ESPACO_PRODUTO):
                espaco_total = np.sum(ESPACO_PRODUTO * self.cromossomo)
                
                if espaco_total > ESPACO_TOTAL:
                    penalidade = espaco_total - ESPACO_TOTAL 
                else:
                    penalidade = 0
                penalidade = penalidade / ESPACO_TOTAL

                return Decimal(penalidade) 
            
            def calcular_fitness_balanceamento(self, VOLUME):
                total_volume = np.sum(VOLUME * self.cromossomo)
                media_volume = total_volume / len(self.cromossomo)
                desbalanceamento = 0

                for v, q in zip(VOLUME, self.cromossomo):
                    if q > 0:  # Apenas itens selecionados
                        desbalanceamento += abs((v * q) - media_volume)
                
                desbalanceamento = desbalanceamento / total_volume
                return Decimal(desbalanceamento)


            def calcular_fitness_total(self, peso_falta, peso_espaco, peso_excesso):
                peso_falta = Decimal(peso_falta)
                peso_espaco = Decimal(peso_espaco)
                peso_excesso = Decimal(peso_excesso)

                return (
                    peso_falta * self.fitnesses_falta +
                    peso_espaco * self.fitnesses_espaco +
                    peso_excesso * self.fitnesses_excessos 
                )
                
                
            def mutar(self,pos,valor):
                self.cromossomo[pos] = valor
                self.fitnesses_falta = self.calcular_fitness_falta(DEMANDA_MINIMA)
                self.fitnesses_excessos = self.calcular_fitness_excessos(DEMANDA_MINIMA)
                self.fitnesses_espaco = self.calcular_fitness_espaco(ESPACO_TOTAL, VOLUME)
                self.fitnesses = self.calcular_fitness_total(PENALIDADE_FALTA, PENALIDADE_ESPACO, PENALIDADE_EXCESSO)

            def __str__(self):
                return f"População: {self.cromossomo}\nFitness: {self.fitnesses}\nFitness Falta: {self.fitnesses_falta}\nFitness Espaço: {self.fitnesses_espaco}\n"

        def selecionar_pais_roleta(populacao):
            try:
                fitnesses = [1 / (float(c.fitnesses) + 1e-6) for c in populacao]
                soma_fitnesses = sum(fitnesses)
                probabilidades = [f / soma_fitnesses for f in fitnesses]
                pais = np.random.choice(populacao, size=2, p=probabilidades, replace=False)
                return pais
            except Exception as e:
                print(e)
                return None,None
            
        def cruzamento_1_corte(pais,prob_cruzamento):
            
            # Gerando numero entre 0 e 1
            prob = random.random()

            if prob < prob_cruzamento:
                ponto = random.randint(1, NUM_PRODUTOS - 1)
                filho_1 = cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA, np.concatenate((pais[0].cromossomo[:ponto], pais[1].cromossomo[ponto:])))
                filho_2 = cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA, np.concatenate((pais[1].cromossomo[:ponto], pais[0].cromossomo[ponto:])))
                return filho_1, filho_2
            else:
                filho_1 = cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA, pais[0].cromossomo)
                filho_2 = cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA, pais[1].cromossomo)
                return filho_1, filho_2

        def cruzamento_2_cortes(pais):
            ponto_1 = random.randint(1, NUM_PRODUTOS - 2)
            ponto_2 = random.randint(ponto_1, NUM_PRODUTOS - 1)
            filho_1 = np.concatenate((pais[0].cromossomo[:ponto_1], pais[1].cromossomo[ponto_1:ponto_2], pais[0].cromossomo[ponto_2:]))
            filho_2 = np.concatenate((pais[1].cromossomo[:ponto_1], pais[0].cromossomo[ponto_1:ponto_2], pais[1].cromossomo[ponto_2:]))
            return cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA,filho_1), cromossomo(NUM_PRODUTOS,MINIMO,MAXIMO,DEMANDA_MINIMA,filho_2)

        def mutacao(cromossomo, prob_mutacao):
            for i,alelo in enumerate(cromossomo.cromossomo):
                prob = random.random()
                if prob < prob_mutacao:
                    while True:
                        novo_alelo = random.randint(MINIMO, MAXIMO)
                        if novo_alelo != alelo:
                            break
                    cromossomo.mutar(i,novo_alelo)
            return cromossomo

        def elitismo(populacao, n):
            populacao.sort(key=lambda x: x.fitnesses)
            return populacao[:n]

        def print_melhores(populacao):
            populacao.sort(key=lambda x: x.fitnesses)
            print("Melhores:")
            print(populacao[0])
            print("\n")

        def print_piores(populacao):
            populacao.sort(key=lambda x: x.fitnesses, reverse=True)
            print("Piores:")
            print(populacao[0])
            print("\n")

        def ajustar_prob_mutacao(progresso_geracao, prob_mutacao_atual, max_sem_melhoria, fator_exploracao=1.5):
            if progresso_geracao > max_sem_melhoria:
                return 0, min(0.8, prob_mutacao_atual * fator_exploracao)
            else:
                return progresso_geracao, max(prob_mutacao, prob_mutacao_atual / fator_exploracao)
            
        # Inicialização do AG
        populacao = [cromossomo(NUM_PRODUTOS, MINIMO, MAXIMO, DEMANDA_MINIMA) for _ in range(tam_pop)]
        lista_completa = {}
        melhor_fitness_anterior = float("inf")
        geracoes_sem_melhoria = 0
        progress_bar = st.progress(0)
        # Box para mensagens dinâmicas
        message_box = st.empty()
        message_box_2 = st.empty()
        message_box_3 = st.empty()

        for i in range(generations):
            # Atualizar o progresso das gerações
            logging.info(f"GERAÇÃO: {i}")
            message_box.write(f"Executando geração {i + 1}/{generations}...")
            message_box_2.write(f"Melhor Fitness: {melhor_fitness_anterior}")
            message_box_3.write(f"Probabilidade de mutação: {prob_mutacao}")
            # Atualize dados no loop
            # Código principal...
            progress_bar.progress((i + 1) / generations)
            time.sleep(0.01)
            nova_populacao = []
            nova_populacao = elitismo(populacao, elitismo_val)
            while len(nova_populacao) < tam_pop:
                pais = selecionar_pais_roleta(populacao)
                filho_1, filho_2 = cruzamento_1_corte(pais, prob_cruzamento)
                filho_1 = mutacao(filho_1, prob_mutacao)
                filho_2 = mutacao(filho_2, prob_mutacao)
                if len(nova_populacao) < tam_pop:
                    nova_populacao.append(filho_1)
                if len(nova_populacao) < tam_pop:
                    nova_populacao.append(filho_2)

            # Atualiza a população
            populacao = nova_populacao
            melhor_fitness_atual = min([ind.fitnesses for ind in populacao])
            lista_completa[i] = {
                "max": max([ind.fitnesses for ind in populacao]),
                "min": melhor_fitness_atual,
                "med": sum([ind.fitnesses for ind in populacao]) / tam_pop,
                "cromossomo" : min(populacao, key=lambda x: x.fitnesses).cromossomo
            }


            # Verifica se houve progresso na solução
            if melhor_fitness_atual < melhor_fitness_anterior:
                geracoes_sem_melhoria = 0
            else:
                geracoes_sem_melhoria += 1
            logging.info(geracoes_sem_melhoria)

            # Ajusta a probabilidade de mutação
            geracoes_sem_melhoria, prob_mutacao = ajustar_prob_mutacao(
                geracoes_sem_melhoria, prob_mutacao, max_sem_melhoria=25
            )
            logging.info(f"Probabilidade de mutação: {prob_mutacao}")

            # Atualiza o melhor fitness anterior
            melhor_fitness_anterior = melhor_fitness_atual

            # # Critério de parada
            # if melhor_fitness_atual <= 10:    
            #     break
        
        st.success("Execução Concluída")
        st.write(f"Melhor Fitness: {melhor_fitness_anterior}")
        m = min(populacao, key=lambda x: x.fitnesses)
        st.write(f"Melhor Solução: {m.cromossomo}")
        st.write(f"Fitness Falta: {m.fitnesses_falta}")
        st.write(f"Fitness Espaço: {m.fitnesses_espaco}")
        st.write(f"Fitness Excessos: {m.fitnesses_excessos}")
        st.write(f"Volume: {m.volume}")
        st.write(f"RESPOSTA: {DEMANDA_MINIMA}")
        plot_generations_with_plotly(lista_completa)
        plot_convergencia_interativa_ag(lista_completa)
        st.success("Algoritmo concluído!")
        st.write(f"Melhor Fitness: {melhor_fitness_anterior}")
        i = 0
        for qtd,volume in zip(range(len(populacao[0].cromossomo)),DEMANDA_MINIMA):
            st.write(f"Produto {i+1}: {populacao[0].cromossomo[qtd]}, demanda {DEMANDA_MINIMA[qtd]}, faltou {DEMANDA_MINIMA[qtd] - populacao[0].cromossomo[qtd]}")
            i += 1
        # Salvar os resultados em um DataFrame para download
        resultados_df = pd.DataFrame([
            {
                "Geracao": i,
                "Fitness": lista_completa[i]['min'],
                "Cromossomo": ", ".join(map(str, lista_completa[i]['cromossomo']))  # Converte a lista em string sem quebras
            }
            for i in lista_completa
        ])
        
        # Salvar o DataFrame no estado da sessão
        st.session_state["resultados_df"] = resultados_df

        # Verificar se há resultados disponíveis para download
        if "resultados_df" in st.session_state:
            st.write("Clique no botão abaixo para baixar os resultados como CSV:")
            st.download_button(
                label="Baixar Resultados",
                data=st.session_state["resultados_df"].to_csv(index=False),
                file_name="resultados.csv",
                mime="text/csv"
            )

def plot_generations_with_plotly(data):
    geracao = list(data.keys())
    melhor = [data[i]['max'] for i in geracao]
    pior = [data[i]['min'] for i in geracao]
    medio = [data[i]['med'] for i in geracao]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=geracao, y=melhor, mode='lines+markers', name='Melhor'))
    fig.add_trace(go.Scatter(x=geracao, y=pior, mode='lines+markers', name='Pior'))
    fig.add_trace(go.Scatter(x=geracao, y=medio, mode='lines+markers', name='Médio'))
    fig.update_layout(
        title="Convergência do Algoritmo Genético",
        xaxis_title="Geração",
        yaxis_title="Fitness",
        yaxis_type="log",
        template="plotly_white"
    )
    st.plotly_chart(fig)

def plot_convergencia_interativa_ag(data):
    geracoes = list(data.keys())
    melhor = [data[i]['min'] for i in geracoes]
    medio = [data[i]['med'] for i in geracoes]
    pior = [data[i]['max'] for i in geracoes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=geracoes, y=melhor, mode='lines+markers', name='Melhor'))
    fig.add_trace(go.Scatter(x=geracoes, y=medio, mode='lines+markers', name='Médio'))
    fig.add_trace(go.Scatter(x=geracoes, y=pior, mode='lines+markers', name='Pior'))
    fig.update_layout(
        title="Convergência do Algoritmo Genético",
        xaxis_title="Geração",
        yaxis_title="Fitness",
        template="plotly_white"
    )
    st.plotly_chart(fig)


def executar_algoritmo_colonia_formigas():

    # Parâmetros
    st.sidebar.title("Configurações")
    demanda_minima = st.sidebar.text_area("Demanda Mínima", value=str(configs.DEMANDA_MINIMA_TEXTO))
    DEMANDA_MINIMA = [int(x) for x in demanda_minima.split(",")]
    volume = st.sidebar.text_area("Volume", value=str(configs.VOLUME_TEXTO))
    VOLUME = [int(x) for x in volume.split(",")]

    NUM_PRODUTOS = len(DEMANDA_MINIMA)
    MINIMO = min(DEMANDA_MINIMA)
    MAXIMO = int(max(DEMANDA_MINIMA) + (max(DEMANDA_MINIMA) * 0.1))

    num_formigas = st.sidebar.slider("Número de Formigas", min_value=10, max_value=1000, value=configs.NUM_FORMIGAS)
    num_iteracoes = st.sidebar.slider("Número de Iterações", min_value=100, max_value=5000, value=configs.NUM_ITERACOES)
    alfa = st.sidebar.slider("Alfa (Peso do Feromônio)", min_value=0.1, max_value=5.0, value=configs.ALFA)
    beta = st.sidebar.slider("Beta (Peso da Heurística)", min_value=0.1, max_value=5.0, value=configs.BETA)
    rho = st.sidebar.slider("Rho (Taxa de Evaporação)", min_value=0.0, max_value=1.0, value=configs.RHO)
    q = st.sidebar.slider("Q (Quantidade de Depósito de Feromônio)", min_value=10, max_value=500, value=configs.Q)
    espaco = st.sidebar.slider("Espaço Total", min_value=50000, max_value=1000000, value=configs.ESPACO_TOTAL)
    penalidade_falta = st.sidebar.slider("Penalidade Falta", min_value=1, max_value=10, value=configs.PENALIDADE_FALTA)
    penalidade_excesso = st.sidebar.slider("Penalidade Excesso", min_value=1, max_value=10, value=configs.PENALIDADE_EXCESSO)
    penalidade_espaco = st.sidebar.slider("Penalidade Espaço", min_value=1, max_value=100, value=configs.PENALIDADE_ESPACO)
    elitismo_val = st.sidebar.slider("Elitismo (Melhores Soluções)", min_value=1, max_value=20, value=configs.V_ELETISMO)
    
    
    if st.sidebar.button("Executar Algoritmo de Colônia de Formigas"):
        with st.spinner("Executando o Algoritmo de Colônia de Formigas..."):
            melhor_solucao, historico_fitness, lista_completa = executar_aco(
                DEMANDA_MINIMA, VOLUME, espaco, num_formigas, num_iteracoes,
                alfa, beta, rho, q, NUM_PRODUTOS, MINIMO, MAXIMO,
                penalidade_falta, penalidade_falta, penalidade_espaco, elitismo_val
            )


            # Resultados
            st.success("Execução Concluída")
            st.write(f"Melhor Fitness: {float(melhor_solucao.fitness)}")
            st.write(f"Melhor Solução: {melhor_solucao.quantidades}")
            st.write(f"Fitness Falta: {float(melhor_solucao.fitnesses_falta)}")
            st.write(f"Fitness Espaço: {float(melhor_solucao.fitnesses_espaco)}")
            st.write(f"Fitness Excessos: {float(melhor_solucao.fitnesses_excessos)}")
            st.write(f"Volume: {melhor_solucao.volume}")
            st.write(f"RESPOSTA: {DEMANDA_MINIMA}")


            # Detalhes por produto
            st.write("Detalhes por Produto:")
            for i, (qtd, demanda) in enumerate(zip(melhor_solucao.quantidades, DEMANDA_MINIMA)):
                faltou = max(0, demanda - qtd)
                st.write(f"Produto {i + 1}: Estoque = {qtd}, Demanda = {demanda}, Faltou = {faltou}")


            # Histórico de Fitness
            st.write("Histórico de Fitness:")
            st.line_chart(historico_fitness)

            # Salvar resultados em um DataFrame para download
            resultados_df = pd.DataFrame([
                {
                    "Iteracao": i,
                    "Fitness": lista_completa[i]['fitness'],
                    "Quantidades": ", ".join(map(str, lista_completa[i]['quantidades']))
                }
                for i in lista_completa
            ])
        
            st.session_state["resultados_df"] = resultados_df

            if "resultados_df" in st.session_state:
                st.write("Clique no botão abaixo para baixar os resultados como CSV:")
                st.download_button(
                    label="Baixar Resultados",
                    data=st.session_state["resultados_df"].to_csv(index=False),
                    file_name="resultados_aco.csv",
                    mime="text/csv"
                )
            plot_convergencia_interativa_aco(lista_completa)
    
def executar_aco(DEMANDA_MINIMA, VOLUME, ESPACO_TOTAL,
                 NUM_FORMIGAS, NUM_ITERACOES, ALFA, BETA, RHO, Q,
                 NUM_PRODUTOS, MINIMO, MAXIMO,
                 PENALIDADE_FALTA, PENALIDADE_EXCESSO, PENALIDADE_ESPACO, ELITISMO_VAL):
    class Solucao:
        def __init__(self, quantidades):
            self.quantidades = quantidades
            self.fitnesses_falta = self.calcular_fitness_falta()
            self.fitnesses_excessos = self.calcular_fitness_excessos()
            self.fitnesses_espaco = self.calcular_fitness_espaco()
            self.fitness = self.calcular_fitness_total()
            self.volume = self.calcular_volume()

        def calcular_volume(self):
            return np.sum(np.array(self.quantidades) * np.array(VOLUME))

        def calcular_fitness_falta(self):
            penalidade = sum(max(0, d - q) for d, q in zip(DEMANDA_MINIMA, self.quantidades))
            penalidade /= sum(DEMANDA_MINIMA)
            return Decimal(penalidade)

        def calcular_fitness_excessos(self):
            penalidade = sum(max(0, q - d) for d, q in zip(DEMANDA_MINIMA, self.quantidades))
            penalidade /= sum(DEMANDA_MINIMA)
            return Decimal(penalidade)

        def calcular_fitness_espaco(self):
            espaco_utilizado = self.calcular_volume()
            if espaco_utilizado > ESPACO_TOTAL:
                penalidade = (espaco_utilizado - ESPACO_TOTAL) / ESPACO_TOTAL
            else:
                penalidade = Decimal(0)
            return Decimal(penalidade)

        def calcular_fitness_total(self):
            return (
                PENALIDADE_FALTA * self.fitnesses_falta +
                PENALIDADE_EXCESSO * self.fitnesses_excessos +
                PENALIDADE_ESPACO * self.fitnesses_espaco
            )

    def inicializar_feromonio():
        feromonio = np.ones((NUM_PRODUTOS, MAXIMO + 1))
        return feromonio

    def calcular_heuristica():
        heuristica = np.zeros((NUM_PRODUTOS, MAXIMO + 1))
        for i in range(NUM_PRODUTOS):
            for q in range(MAXIMO + 1):
                falta = max(0, DEMANDA_MINIMA[i] - q)
                excesso = max(0, q - DEMANDA_MINIMA[i])
                espaco_utilizado = q * VOLUME[i]
                espaco_penalidade = max(0, (espaco_utilizado - ESPACO_TOTAL) / ESPACO_TOTAL)
                heuristica[i, q] = 1.0 / (
                    PENALIDADE_FALTA * falta / sum(DEMANDA_MINIMA) +
                    PENALIDADE_EXCESSO * excesso / sum(DEMANDA_MINIMA) +
                    PENALIDADE_ESPACO * espaco_penalidade + 1e-6
                )
        return heuristica

    def construir_solucoes(feromonio, heuristica):
        solucoes = []
        for _ in range(NUM_FORMIGAS):
            quantidades = []
            for i in range(NUM_PRODUTOS):
                probabilidades = (feromonio[i, :] ** ALFA) * (heuristica[i, :] ** BETA)
                probabilidades /= np.sum(probabilidades)
                quantidade = np.random.choice(range(MAXIMO + 1), p=probabilidades)
                quantidades.append(quantidade)
            solucoes.append(Solucao(quantidades))
        return solucoes

    def atualizar_feromonio(feromonio, solucoes):
        feromonio *= (1 - RHO)
        solucoes.sort(key=lambda s: s.fitness)
        melhores_solucoes = solucoes[:ELITISMO_VAL]
        for solucao in melhores_solucoes:
            for i in range(NUM_PRODUTOS):
                q = solucao.quantidades[i]
                feromonio[i, q] += Q / (float(solucao.fitness) + 1e-6)
        return feromonio

    feromonio = inicializar_feromonio()
    heuristica = calcular_heuristica()
    melhor_solucao_global = None
    historico_fitness = []
    lista_completa = {}

    progress_bar = st.progress(0)
    message_box = st.empty()
    message_box_2 = st.empty()

    for iteracao in range(NUM_ITERACOES):
        solucoes = construir_solucoes(feromonio, heuristica)
        feromonio = atualizar_feromonio(feromonio, solucoes)
        melhor_solucao_iteracao = min(solucoes, key=lambda s: s.fitness)

        if (melhor_solucao_global is None) or (melhor_solucao_iteracao.fitness < melhor_solucao_global.fitness):
            melhor_solucao_global = melhor_solucao_iteracao

        historico_fitness.append(float(melhor_solucao_global.fitness))
        lista_completa[iteracao] = {
            "fitness": float(melhor_solucao_global.fitness),
            "quantidades": melhor_solucao_global.quantidades,
        }

        progress_bar.progress((iteracao + 1) / NUM_ITERACOES)
        message_box.write(f"Iteração {iteracao + 1}/{NUM_ITERACOES}")
        message_box_2.write(f"Melhor Fitness: {float(melhor_solucao_global.fitness)}")
        time.sleep(0.01)

    return melhor_solucao_global, historico_fitness, lista_completa



# Funções de plotagem
def plot_convergencia_interativa_aco(lista_completa):
    iteracoes = list(lista_completa.keys())
    fitness = [lista_completa[i]['fitness'] for i in iteracoes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iteracoes, y=fitness, mode='lines+markers', name='Fitness'))
    fig.update_layout(
        title="Convergência do Algoritmo de Colônia de Formigas",
        xaxis_title="Iteração",
        yaxis_title="Fitness",
        template="plotly_white"
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
