import numpy as np
import abc
from copy import deepcopy
from sklearn.decomposition.base import _BasePCA


class TransformerInterface(abc.ABCMeta):
    @abc.abstractmethod
    def inverse_transform(self, X):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def transform(self, X, y=None):
        pass


class DomainAdapter:
    def __init__(self, transformer: TransformerInterface, ref_img: np.ndarray):
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))

    @staticmethod
    def flatten(img):
        return img.reshape(-1, 3) / 255.

    @staticmethod
    def reconstruct(pixels, h, w):
        return (np.clip(pixels, 0, 1).reshape(h, w, -1) * 255).astype('uint8')

    @staticmethod
    def _pca_sign(x: _BasePCA):
        return np.sign(np.trace(x.components_))

    def __call__(self, image: np.ndarray):
        h, w, _ = image.shape
        pixels = self.flatten(image)
        self.source_transformer.fit(pixels)

        if isinstance(self.target_transformer, _BasePCA):
            # dirty hack to make sure colors are not inverted
            self.source_transformer: _BasePCA  # keep IDE cool
            if self._pca_sign(self.target_transformer) != self._pca_sign(self.source_transformer):
                self.target_transformer.components_ *= -1

        representation = self.source_transformer.transform(pixels)
        result = self.target_transformer.inverse_transform(representation)
        return self.reconstruct(result, h, w)
