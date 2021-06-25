import numpy as np
from torchzq.metric import Metrics


def test_metric():
    metrics = Metrics()
    metrics.add_metric("test_min", [], "min")
    metrics.add_metric("test_max", [], "max")
    values = np.random.randint(0, 100, 1000)
    for i, value in enumerate(values, 1):
        metrics(dict(test_min=value, test_max=value))
        print(metrics.to_dict())
        assert metrics.to_dict()["test_min/best_score"] == values[:i].min()
        assert metrics.to_dict()["test_max/best_score"] == values[:i].max()
