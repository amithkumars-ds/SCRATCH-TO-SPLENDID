import math

class LinearRegression:
    def __init__(self):
        self.theta_0 = 0
        self.theta_1 = 0
        self.training_status = False
        self.model = {}

    def fit(self, X:list, y:list):
        if len(X) != len(y):
            raise ValueError("X & y must be of same length...")
        
        if len(X) == 0:
            raise ValueError("Training data cannot be empty...")
        
        print('[INFO] model training started...')

        self._gradient_descent(X,y,
                               theta_0=self.theta_0,
                               theta_1=self.theta_1,
                               lr=0.001,
                               iterations=10000)

        print('[INFO] model training complete...')


    def _gradient_descent(self, X:list, y:list, theta_0:float, theta_1:float, lr:float, iterations:int):
        m = len(y) # length of data for computations
        for _ in range(iterations):
            theta_0_update = theta_0 - (lr/m)*(theta_0*m
                                               + sum(theta_1 * x for x in X)
                                               - sum(y))
            theta_1_update = theta_1 - (lr/m)*(sum(theta_0*x for x in X)
                                               + sum(theta_1*(x**2) for x in X)
                                               - sum(xi*yi for xi, yi in zip(X,y)))
            
            theta_0 = theta_0_update
            theta_1 = theta_1_update

        self.coef_ = theta_1
        self.intercept_ = theta_0
        self.training_status = True

        self.model = {
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'training status': self.training_status
        }

        return None

    def predict(self, new_input:int|float):
        if not self.training_status:
            raise ValueError("Model is not trained yet...")
        
        predicted_output = self.intercept_ + self.coef_*new_input
        return predicted_output

    def evaluate(self, X, y_test:list):
        if not self.training_status:
            raise ValueError("Model is not trained yet...")

        if len(X) != len(y_test):
            raise ValueError("X & y must be of same length...")
        y_pred = []
        for i in X:
            predictions = self.predict(i)
            y_pred.append(predictions)
        
        mse_score = self.mse(y_test, y_pred)
        rmse_score = math.sqrt(mse_score)

        evaluation_metrics = {
            'mse_score':mse_score, 
            'rmse_score':rmse_score
            }

        return f"Evaluation Metrics: {evaluation_metrics}"

    def mse(self, y_test:list, y_pred:list):
        m = len(y_test)

        mean_squared_error = [(y_act_i - y_pred_i)**2 for y_act_i, y_pred_i in zip(y_test,y_pred)]
        mse_score = (1/m)*sum(mean_squared_error)
        return round(mse_score,2)
    
    def __repr__(self):
        return f"{self.model}"



if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('sample.csv')

    X = data['YearsExperience'].to_list()
    y = data['Salary'].to_list()

    lr_model = LinearRegression()
    lr_model.fit(X,y)
    print(lr_model.predict(1.6))
    print(lr_model.evaluate(X,y))
