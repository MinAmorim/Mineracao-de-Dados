# Trabalho 04 — Implementação do KNN

**Universidade Federal do Ceará · Departamento de Computação**
Disciplina de Mineração de Dados — Prof. José Macedo

- Samyra Vitória Lima de Almeida — 521240
- Yasmin Santos de Amorim — 566326

## Objetivo

Implementar do zero o algoritmo **K-Nearest Neighbors (KNN)** e avaliá-lo em um problema de classificação binária.

## Estrutura

```
Trabalho 04/
├── docs/README.md                      # este documento
└── notebooks/knn_implementation.ipynb  # implementação e experimentos
```

## Implementação

A classe `KNN` usa apenas NumPy e `collections.Counter`:

- `fit(X, y)`: armazena os dados de treino, sem treinamento real.
- `calcular_distancia`: calcula a distância euclidiana.
- `predict(X)`: para cada amostra, pega os `k` vizinhos mais próximos (`argsort` estável, para o desempate ser determinístico) e faz votação majoritária.
- `score(X, y)`: calcula a acurácia.

## Metodologia

1. **Dataset:** Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`), com 569 amostras, 30 features e 2 classes.
2. **Divisão:** 80% treino (455) e 20% teste (114), estratificada, com `random_state=42`.
3. **Normalização:** `StandardScaler` ajustado só no treino.
4. **Escolha de k:** validação cruzada estratificada 5-fold no treino com k ∈ {1, 3, …, 49}. A normalização é refeita dentro de cada fold para evitar vazamento de dados.
5. **Avaliação final:** no conjunto de teste, usando o melhor k.

## Resultados

| Configuração                 | Acurácia no teste |
|------------------------------|-------------------|
| k = 5 (fixo)                 | 95,61%            |
| k = 9 (escolhido por CV)     | **97,37%**        |

## Como executar

```bash
pip install numpy scikit-learn matplotlib seaborn
jupyter notebook "Trabalho 04/notebooks/knn_implementation.ipynb"
```
