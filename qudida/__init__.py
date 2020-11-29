import abc
from copy import deepcopy

import numpy as np
from sklearn.decomposition.base import _BasePCA
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d


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
    def __init__(self, transformer: TransformerInterface, ref_img: np.ndarray, kernel_size=1):
        self.kernel_size = kernel_size
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))

    def flatten(self, img):
        img = img.astype('float32') / 255.
        if self.kernel_size == 1:
            return img.reshape(-1, 3)
        patches = extract_patches_2d(img, (2, 2))
        pixels = patches.reshape(len(patches), -1)
        return pixels

    def reconstruct(self, pixels, h, w):
        pixels = (np.clip(pixels, 0, 1) * 255).astype('uint8')
        if self.kernel_size == 1:
            return pixels.reshape(h, w, 3)
        patches = pixels.reshape(-1, self.kernel_size, self.kernel_size, 3)
        return reconstruct_from_patches_2d(patches, (h, w, 3))

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
