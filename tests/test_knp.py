import keras_neural_processes as knp


def test_CNP():
    model = knp.CNP()
    assert model


def test_NP():
    model = knp.NP()
    assert model


def test_ANP():
    model = knp.ANP()
    assert model
