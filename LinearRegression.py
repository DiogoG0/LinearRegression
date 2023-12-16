import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, max_iterations=10000, min_delta_iterations=0.0001):
        """
        Simple Linear Regression model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate (between 0.0 and 1.0).
        max_iterations : int, optional
            The number of maximum training iterations.
        min_delta_iterations : float, optional
            The minimal change between delta iterations (between 0.0 and 1.0).
            
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_delta_iterations = min_delta_iterations
        self.coefficient = 0
        self.intercept = 0
        self.mean_squared_error = None
        self.reason = None

    def fit(self, X, Y):
        
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The training data.
        Y : array-like, shape (n_samples,)
            The target values.
        """
        self.fit_gradient_descent(X, Y)

    def fit_gradient_descent(self, X, Y):
        
        
        """"
        Fit the model using Gradient Descent.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The training data.
        Y : array-like, shape (n_samples,)
            The target values.
        """
        Y_mean = np.mean(Y)
        sum_squared_total = np.sum((Y - Y_mean) ** 2)
        n = float(len(X))

        for i in range(self.max_iterations + 1):
            Y_pred = self.coefficient * X + self.intercept
            dif = Y - Y_pred
            gradient_coefficient = (-2 / n) * np.sum(X * dif)
            gradient_intercept = (-2 / n) * np.sum(dif)
            
            old_error = np.sum(((X * self.coefficient + self.intercept) - Y) ** 2)
            
            self.coefficient -= self.learning_rate * gradient_coefficient
            self.intercept -= self.learning_rate * gradient_intercept
            
            new_error = np.sum(((X * self.coefficient + self.intercept) - Y) ** 2)
            self.mean_squared_error = new_error / n
            
            delta_iterations = ((new_error - old_error) / old_error)
            
            if abs(delta_iterations) < self.min_delta_iterations:
                self.reason = (f'Stopped with a delta_iterations of {round(delta_iterations, 5)} ')
                break
            elif i == self.max_iterations:
                self.reason = (f'Stopped with the max_iterations of {self.max_iterations}')

    def R2(self, Y):
        """
        Returns R squared.

        Parameters
        ----------
        Y : array-like, shape (n_samples,)
            The target values.
        """
        sum_squared_total = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (self.mean_squared_error / sum_squared_total)
        print(f'{round((r_squared) * 100, 2)}% of the DataSet variability is explained by the constructed model')

    def predict(self, X):
        """
        Make predictions for input matrix X.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The input data.

        Returns
        -------
        None
        """
        if not self.coefficient or not self.intercept:
            print("The model has not been trained yet!")
            return

        predictions = X * self.coefficient + self.intercept
        rounded_predictions = np.round(predictions, 2)  # Arredondar cada valor para 2 casas decimais
          
        return predictions
        
             
    def predict_input(self):
        """
        Predict the output of a given X provided by a prompt(input) of X.

        The input only accepts numbers, but there is a possibility to exit the loop with a word in the list:
        ['quit', 'sair', 'leave', 'exit', 'parar', 'stop','ok', ' ']

        Returns
        -------
        None
        """
        
        while True:
            user_input = input('What is the value of X? ')
            
            if user_input.lower() in ['quit', 'sair', 'leave', 'exit', 'parar', 'stop', 'ok','']:
                break
            else:
                try:
                    user_input = float(user_input)
                    prediction = self.coefficient * user_input + self.intercept
                    
                    if prediction is not None:
                        print(f'For an X of {user_input}, the expected output is {round(prediction, 2)}')
                        break
                except ValueError:
                    print('A number must be selected, or exit the function, please try again!')
                    
