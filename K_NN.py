class KNN:
    def __init__(self, k=3):
        self.k = k

    @staticmethod
#Definicion de distancia euclidiana
    def _euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return distance**0.5

#Obtencion de neighbors
    def _get_neighbors(self, train, test_row):
        distances = list()
        for train_row in train:
            dist = self._euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

#Prediccion 
    def predict(self, train, test_row):
        neighbors = self._get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

# Definimos nuestro conjunto de datos
dataset = [
    [158, 58, 'M'],
    [158, 63, 'M'],
    [160, 59, 'M'],
    [160, 60, 'M'],
    [163, 60, 'M'],
    [163, 61, 'M'],
    [160, 64, 'L'],
    [163, 64, 'L'],
    [165, 61, 'L'],
    [165, 62, 'L'],
    [165, 65, 'L'],
    [168, 62, 'L'],
    [168, 63, 'L'],
    [168, 66, 'L'],
]

# Creamos una instancia de nuestro clasificador kNN
knn = KNN(k=3)

#nueva_instancia = [170, 68] #da L
nueva_instancia = [159, 68] #da M
print(knn.predict(dataset, nueva_instancia))  