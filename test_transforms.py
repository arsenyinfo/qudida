import cv2
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from itertools import product
from qudida import DomainAdapter
from imageio import mimsave


def params_combinations():
    return product(
        (QuantileTransformer(n_quantiles=255),
         StandardScaler(),
         MinMaxScaler(),
         PCA(n_components=2),
         ),
        ((None, None), (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR)),
        (1, 2),
    )


@pytest.mark.parametrize('transformer,color_conversions,kernel_size',
                         params_combinations()
                         )
def test_transform(transformer, kernel_size, color_conversions):
    adapter = DomainAdapter(transformer=transformer,
                            ref_img=cv2.imread('target.png'),
                            kernel_size=kernel_size,
                            color_conversions=color_conversions,
                            )
    source = cv2.imread('source.png')
    result = adapter(source)
    cv2.imwrite('result.png', result)


def test_save_gif():
    frames = []
    for t, c, k in params_combinations():
        adapter = DomainAdapter(transformer=t,
                                ref_img=cv2.imread('target.png'),
                                kernel_size=k,
                                color_conversions=c,
                                )
        source = cv2.imread('source.png')
        result = adapter(source)
        result = cv2.putText(result, f'{t.__class__.__name__}, kernel size {k}, color {"RGB" if c[0] is None else "YCrCb"}', (20, 30),
                             fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0))
        frames.append(result)

    mimsave('result.gif', frames, fps=.5)
