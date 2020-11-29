import cv2
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

from qudida import DomainAdapter


@pytest.mark.parametrize('transformer', (QuantileTransformer(),
                                         StandardScaler(),
                                         MinMaxScaler(),
                                         PCA(n_components=2),
                                         )
                         )
def test_pca(transformer):
    adapter = DomainAdapter(transformer=transformer, ref_img=cv2.imread('target.png'))
    source = cv2.imread('source.png')
    result = adapter(source)
    cv2.imwrite('result.png', result)
