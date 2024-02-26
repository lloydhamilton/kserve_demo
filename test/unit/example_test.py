from custom_predictor.dummy_model import DummyModel


class TestCustomPredictor:
    def test_custom_predictor(self):
        pass

    def test_dummy_model(self):
        model = DummyModel()
        assert isinstance(model, DummyModel)
