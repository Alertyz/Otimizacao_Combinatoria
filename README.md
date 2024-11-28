# Algoritmos de Otimização para Estoque

Este projeto implementa algoritmos de otimização para gerenciamento de estoque utilizando **Streamlit** para visualização interativa. Os algoritmos disponíveis são:

- Algoritmo Genético
- Algoritmo de Colônia de Formigas (ACO)

---

## Estrutura do Projeto

### Principais Arquivos
- **`main.py`**: Script principal que executa o aplicativo Streamlit.
- **`configs.py`**: Configurações gerais do projeto, como parâmetros dos algoritmos, demanda mínima e volume de produtos.

### Bibliotecas Utilizadas
- **Streamlit**: Interface interativa para configurar e executar os algoritmos.
- **Numpy** e **Pandas**: Manipulação de dados e cálculos numéricos.
- **Plotly** e **Matplotlib**: Visualização de gráficos.
- **Logging**: Registro de eventos durante a execução.

---

## Pré-requisitos

1. **Python 3.11.9**:
   Certifique-se de que você está usando a versão correta do Python.

2. **Bibliotecas necessárias**:
   As dependências estão listadas no arquivo `requirements.txt`. Instale-as com o comando:

   ```bash
   pip install -r requirements.txt
   ```

---

## Como Rodar

1. Execute o aplicativo Streamlit:

   ```bash
   python -m streamlit run main.py
   ```

2. Abra o link exibido no terminal em seu navegador para acessar a interface interativa.

---

## Funcionalidades

### Algoritmo Genético
- Otimização de estoque baseada em:
  - Demanda mínima.
  - Volume de produtos.
  - Penalidades para falta, excesso e uso de espaço.
- Configurações ajustáveis:
  - Tamanho da população.
  - Probabilidade de cruzamento e mutação.
  - Penalidades.
  - Elitismo.

### Algoritmo de Colônia de Formigas
- Otimização de estoque com base em:
  - Feromônio e heurísticas para decisões probabilísticas.
- Configurações ajustáveis:
  - Número de formigas e iterações.
  - Taxa de evaporação.
  - Parâmetros alfa, beta e rho.

---

## Resultados
- Visualização interativa:
  - Convergência dos algoritmos ao longo das gerações/iterações.
  - Melhor solução encontrada, incluindo penalidades e volume.
- Exportação:
  - Resultados podem ser baixados em formato `.csv` diretamente da interface.

---

## Configurações do Projeto (`configs.py`)

Os parâmetros iniciais do projeto estão definidos no arquivo `configs.py`:
- **Demanda mínima** e **volume de produtos**:  
  Arrays pré-configurados com dados para cada produto.
- **Espaço total disponível**:  
  Configuração inicial para restrição de espaço no algoritmo.
- **Parâmetros dos algoritmos**:  
  Incluem penalidades, probabilidades e fatores de ajuste.

---

## Exemplos de Uso

1. **Configurar Parâmetros**:  
   Ajuste os parâmetros no menu lateral para personalizar o comportamento dos algoritmos.

2. **Executar Algoritmos**:  
   Escolha entre Algoritmo Genético ou Colônia de Formigas e execute para otimizar o estoque.

3. **Visualizar Resultados**:  
   Acompanhe a evolução do algoritmo em gráficos interativos.

4. **Baixar Resultados**:  
   Faça o download das soluções em `.csv` para análise posterior.
