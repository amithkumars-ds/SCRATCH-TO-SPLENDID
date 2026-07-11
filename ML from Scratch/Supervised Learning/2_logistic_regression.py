import math

class LogisticRegression:
    def __init__(self):
        self.alpha_0 = 0
        self.alpha_1 = 0
        self.training_status = False
        self.model = {}
        self.threshold = 0.5

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X & y must be of same length...")
        
        if len(X) == 0:
            raise ValueError("Training Data cannot be empty")
        
        print('[INFO] model training started...')

        self._gradientDescent(X, y,
                              lr=0.01,
                              alpha_0=self.alpha_0,
                              alpha_1=self.alpha_1,
                              iterations=10000)
        
        print('[INFO] model training complete...')


    def _gradientDescent(self, X, y, lr, alpha_0, alpha_1, iterations):
        m = len(y)
        for _ in range(iterations):
            alpha_0_update = alpha_0 - (lr/m)*(- sum(y)
                                              + sum(1 / (1 + math.exp(-(alpha_0 + alpha_1 * x))) for x in X))

            alpha_1_update = alpha_1 - (lr / m) * (sum(x * (1 / (1 + math.exp(-(alpha_0 + alpha_1 * x)))) for x in X)
                                                       - sum(xi * yi for xi, yi in zip(X, y)))

            alpha_0 = alpha_0_update
            alpha_1 = alpha_1_update
        
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.training_status = True

        self.model = {
            'alpha_0': self.alpha_0,
            'alpha_1': self.alpha_1,
            'threshold': self.threshold,
            'training_status': self.training_status
        }

        return None
    
    def predict(self, new_input:int|float):
        probabilities = self.predict_proba(new_input)
        class_1_proba = probabilities.get(1,0)
        if class_1_proba >= self.threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        return predicted_class
    
    def predict_proba(self, new_input: int|float):
        z = self.alpha_0 + self.alpha_1*float(new_input)
        sig_z = self._sigmoid(z)
        probabilites = {0:1-sig_z, 1:sig_z}

        return probabilites

    def _sigmoid(self, z):
        sig_z = 1/(1+math.exp(-z))
        return sig_z
    
    def evaluate(self, X:list[int], y_test:list[int]):
        if len(X) != len(y_test):
            raise ValueError("X & y must be of same length...")
        
        if not self.training_status:
            raise ValueError("Model not trained on data...")
        
        y_pred = []
        for i in X:
            prediction = self.predict(i)
            y_pred.append(prediction)
        accuracy_score = self._accuracy(y_test,y_pred)

        evaluation_metrics = {
            'accuracy': accuracy_score
        }

        return evaluation_metrics

    def _accuracy(self, y_test:list[int], y_pred:list[int]):
        m = len(y_test)
        score = sum(yt == yp for yt, yp in zip(y_test, y_pred))
        accuracy = score/m

        return accuracy
    
    def __repr__(self):
        return f"LogisticRegression Model({self.model})"






if __name__ == '__main__':
    X = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    y = [0,0,0,0,0,1,1,1,1,1,1]
    logreg_model = LogisticRegression()
    logreg_model.fit(X,y)
    print(logreg_model.evaluate(X,y))
    print(logreg_model.predict(44))
    print(logreg_model.predict_proba(44))
