import numpy as np
import matplotlib.pyplot as plt

# Base dados aleatória
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Preparação dos dados para teste e treino
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = LinearRegression(learning_rate=0.001, max_iterations=10000, min_delta_iterations=0.0001)
model.fit(X_train.flatten(), Y_train.flatten())

# Criar e treinar um modelo existente em biblioteca (scikit-learn) para comparação
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, Y_train)

# Previsão dos dados de teste
model_predictions = model.predict(X_test.flatten())
sklearn_model_predictions = sklearn_model.predict(X_test)

# Cálculo do erro médio quadrático para ambos os modelos
model_mse = mean_squared_error(Y_test.flatten(), model_predictions)
sklearn_model_mse = mean_squared_error(Y_test, sklearn_model_predictions)


# Plot dos dados de teste e das linhas de regressão para ambos os modelos
plt.scatter(X_test, Y_test, label='Test Data')
plt.plot(X_test, sklearn_model_predictions, color='blue', label='Sklearn')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
print(f'Sklearn - Mean Squared Error on Test Data: {sklearn_model_mse}')

plt.scatter(X_test, Y_test, label='Test Data')
plt.plot(X_test, model_predictions, color='red', label='LinearRegression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
print(f'LinearRegression - Mean Squared Error on Test Data: {model_mse}')

#Diferentças dos Modelos
difference = sklearn_model_mse - model_mse
print(f'The difference between our LinearRegression Model and Sklearn Model, for this data test is: {difference}')
