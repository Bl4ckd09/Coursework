from nearest_neighbour import nearest_neighbour_classification

def test_nearest_neighbour_classification():
    training_features = [[0.0, 0.0], [1.0, 1.0]]
    training_labels = ["A", "B"]
    query = [[0.1, 0.1]]
    prediction = nearest_neighbour_classification(training_features, training_labels, query, p=2)
    assert prediction == ["A"]
