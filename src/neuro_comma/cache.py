import threading

from neuro_comma.predict import Predictor


class ModelCache(object):
    """ """
    __shared_state = {
        "_model": None,
        "_lock": threading.Lock()
    }

    def __init__(self):
        self.__dict__ = self.__shared_state

    def load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # загружаем модель
                    self._model = Predictor('repunct-model', model_weights='weights_ep4_9910.pt')

    @property
    def model(self) -> Predictor:
        self.load_model()
        return self._model
