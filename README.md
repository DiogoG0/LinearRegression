# LinearRegression
Este projeto implementa um algoritmo básico de Regressão Linear Simples em Python sem utilizar bibliotecas externas, exceto NumPy para algumas operações.

## Implementação
A implementação é mantida simples para maior clareza e facilidade de uso. As principais características incluem:
  - Condições de paragem definidas pelo utilizador (learning_rate, max_iterations, min_delta_iterations).
  - Capacidade de fazer previsões para um conjunto de dados sem resultados conhecidos.
  - Exemplo de aplicação do modelo com suporte gráfico.

## Como usar
- Importe a classe LinearRegression para o seu script
- from linear_regression import LinearRegression
- Crie uma variável da classe LinearRegression
- Traine o modelo (model.fit(X_train, y_train))
- Faça previsões
  - X_test = np.array([[4, 5], [5, 6]])
  - predictions = model.predict(X_test)
  - print("Predictions:", predictions)
