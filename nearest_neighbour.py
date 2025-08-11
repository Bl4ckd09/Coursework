def minkowski_distance(x, y, p):
    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1 / p)


def nearest_neighbour_classification(training_features, training_labels, testing_features, p):
    predictions = []
    for test_point in testing_features:
        distances = [minkowski_distance(test_point, train_point, p) for train_point in training_features]
        nearest_index = distances.index(min(distances))
        predictions.append(training_labels[nearest_index])
    return predictions
