from itertools import product

import cv2
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

from qudida import DomainAdapter


def params_combinations():
    return product(
        (QuantileTransformer(n_quantiles=255),
         StandardScaler(),
         MinMaxScaler(),
         PCA(n_components=2),
         ),
        ((None, None),
         (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR),
         (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
         ),
    )


@pytest.mark.parametrize('transformer,color_conversions',
                         params_combinations()
                         )
def test_transform(transformer, color_conversions):
    adapter = DomainAdapter(transformer=transformer,
                            ref_img=cv2.imread('target.png'),
                            color_conversions=color_conversions,
                            )
    source = cv2.imread('source.png')
    res = adapter(source)
    assert res.shape == source.shape
