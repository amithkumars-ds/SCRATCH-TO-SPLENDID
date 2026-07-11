import math

class KNearestRegressor:
    def __init__(self):
        self.k = 3
        self.model = {}
        self.model['training_status'] = False

    def fit(self, X: list[int | float], y: list[int | float],
            distance_type: str,
            k: int = 3):

        if len(X) != len(y):
            raise ValueError("X & y must be of same length")

        if len(X) == 0:
            raise ValueError("Training Data cannot be empty...")
        
        if k > len(X):
            raise ValueError("K cannot be larger than Training Data...")
        
        for i in y:
            if type(i) not in (int,float):
                raise ValueError(f"Invalid datatype found in {y}")

        if distance_type not in ('euclidean', 'manhattan'):
            raise ValueError("distance_type must be 'euclidean' or 'manhattan'")

        print('[INFO] model training started...')

        self.distance_type = distance_type
        self.k = k

        self.model['X'] = X
        self.model['y'] = y
        self.model['training_status'] = True
        self.model['distance_type'] = self.distance_type
        self.model['k'] = self.k


        print('[INFO] model training complete...')

        return self.model

    def predict(self, new_input: int | float):
        if not self.model['training_status']:
            raise ValueError("Model is not trained on data...")

        output = self._knn_reg(new_input)
        return output

    def _knn_reg(self, new_input: int | float):
        # computing the distances between P and all other data samples
        distances = self.compute_distance(
            X=self.model['X'],
            new_input=new_input
        )

        # getting sorted distances
        sorted_distance_indices = sorted(
            range(len(distances)),
            key=lambda i: distances[i]
        )

        # sorting the target vales wrt the distances
        sorted_target_values = [
            self.model['y'][i]
            for i in sorted_distance_indices
        ]

        # choosing top K entries
        top_k_values = sorted_target_values[:self.k]

        # getting the average of the top-K values from the target
        predicted_output = (1/self.k)*(sum(top_k_values))

        return round(predicted_output,2)

    def compute_distance(self, X: list[int], new_input:int):
        x1 = new_input

        if self.distance_type == 'euclidean':
            distances = [math.sqrt((x2i - x1) ** 2) for x2i in X]
        else:
            distances = [abs(x2i - x1) for x2i in X]

        return distances

    def evaluate(self, X_test, y_test):
        if len(X_test) != len(y_test):
            raise ValueError('X & y must be of same length...')

        if len(y_test) == 0:
            raise ValueError('Testing data cannot be empty...')

        y_pred = []
        for point in X_test:
            prediction = self.predict(point)
            y_pred.append(prediction)

        mse, rmse = self._evaluation_metrics(y_test,y_pred)

        evaluation_metrics = {
            'mse':round(mse,2),
            'rmse':round(rmse,2)
        }

        return evaluation_metrics

    def _evaluation_metrics(self, y_test, y_pred):
        m = len(y_test)
        errors = ((yt-yp)**2 for yt, yp in zip(y_test,y_pred))
        mse = (1/m)*(sum(errors))
        rmse = math.sqrt(mse)

        return mse, rmse
    
    def __repr__(self):
        return f"KNearestRegressor(Model: {self.model})"


if __name__ == '__main__':
    X = [1, 2, 3, 6, 7, 8]
    y = [1, 2, 1, 7, 8, 7]

    knn_regressor = KNearestRegressor()
    knn_regressor.fit(X, y, 'euclidean', 3)
    print(knn_regressor.predict(44))

    X_test = [3, 5 ,7]
    y_test = [4, 5 ,6]
    print(knn_regressor.evaluate(X_test, y_test))
    print(knn_regressor)