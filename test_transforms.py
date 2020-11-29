import cv2
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from itertools import product
from qudida import DomainAdapter


@pytest.mark.parametrize('transformer,kernel_size',
                         product((QuantileTransformer(),
                                  StandardScaler(),
                                  MinMaxScaler(),
                                  PCA(n_components=2),),
                                 (1, 2)
                                 )
                         )
def test_transform(transformer, kernel_size):
    adapter = DomainAdapter(transformer=transformer, ref_img=cv2.imread('target.png'), kernel_size=kernel_size)
    source = cv2.imread('source.png')
    result = adapter(source)
    cv2.imwrite('result.png', result)
