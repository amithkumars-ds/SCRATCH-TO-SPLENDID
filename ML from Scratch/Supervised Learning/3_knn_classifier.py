import math

class KNearestClassifier:
    def __init__(self):
        self.k = 3
        self.model = {}
        self.model['training_status'] = False

    def fit(self, X: list[int | float], y: list[int | float],
            class_labels: list[int | str],
            distance_type: str,
            k: int = 3):

        if len(X) != len(y):
            raise ValueError("X & y must be of same length")

        if len(X) != len(class_labels):
            raise ValueError("X & class_labels must be of same length")

        if len(X) == 0:
            raise ValueError("Training Data cannot be empty...")

        if distance_type not in ('euclidean', 'manhattan'):
            raise ValueError("distance_type must be 'euclidean' or 'manhattan'")

        print('[INFO] model training started...')

        self.model['X'] = X
        self.model['y'] = y
        self.model['class'] = class_labels
        self.model['training_status'] = True

        self.distance_type = distance_type
        self.k = k

        print('[INFO] model training complete...')

        return self.model

    def predict(self, new_input: tuple[int | float]):
        if not self.model['training_status']:
            raise ValueError("Model is not trained on data...")

        output = self._knn(new_input)
        return output

    def _knn(self, new_input: tuple[int | float]):
        # computing the distances between P and all other data samples
        distances = self.compute_distance(
            X=self.model['X'],
            y=self.model['y'],
            new_input=new_input
        )

        # getting sorted distances
        sorted_distance_indices = sorted(
            range(len(distances)),
            key=lambda i: distances[i]
        )

        # sorting the class labels wrt the distances
        sorted_class_labels = [
            self.model['class'][i]
            for i in sorted_distance_indices
        ]

        # choosing top K entries
        top_k_labels = sorted_class_labels[:self.k]

        # create a frequency map
        freq_map = {}
        for i in top_k_labels:
            freq_map[i] = freq_map.get(i, 0) + 1

        # choose the most frequent class label
        most_frequent_label = max(freq_map, key=freq_map.get)

        return most_frequent_label

    def compute_distance(self, X: list[int], y: list[int], new_input):
        x1 = new_input[0]
        y1 = new_input[1]

        if self.distance_type == 'euclidean':
            distances = [
                math.sqrt((x2i - x1) ** 2 + (y2i - y1) ** 2)
                for x2i, y2i in zip(X, y)
            ]
        else:
            distances = [
                abs(x2i - x1) + abs(y2i - y1)
                for x2i, y2i in zip(X, y)
            ]

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

        accuracy = self._accuracy_score(y_test, y_pred)
        accuracy = round(accuracy,2)

        evaluation_metrics = {
            'accuracy': accuracy
        }

        return evaluation_metrics

    def _accuracy_score(self, y_test, y_pred):
        m = len(y_test)
        scores = sum((yt == yp) for yt, yp in zip(y_test, y_pred))
        accuracy = (1 / m) * scores

        return accuracy


if __name__ == '__main__':
    X = [1, 2, 3, 6, 7, 8]
    y = [1, 2, 1, 7, 8, 7]
    class_labels = ["A", "A", "A", "B", "B", "B"]

    knn_classifier = KNearestClassifier()
    knn_classifier.fit(X, y, class_labels, 'euclidean', 3)
    print(knn_classifier.predict((44, 43)))

    X_test = [(2, 2), (7, 7), (4, 4)]
    y_test = ["A", "B", "B"]
    print(knn_classifier.evaluate(X_test, y_test))