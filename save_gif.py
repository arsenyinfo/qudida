import cv2
from imageio import mimsave
from tqdm import tqdm

from qudida import DomainAdapter
from test_transforms import params_combinations


def _get_colorspace_name(x):
    d = {None: 'RGB',
         cv2.COLOR_YCrCb2BGR: 'YCrCb',
         cv2.COLOR_HSV2BGR: 'HSV',
         }
    return d[x]


def save_gif():
    frames = []
    for t, c in tqdm(params_combinations()):
        adapter = DomainAdapter(transformer=t,
                                ref_img=cv2.imread('target.png'),
                                color_conversions=c,
                                )
        source = cv2.imread('source.png')
        result = adapter(source)
        result = cv2.putText(result,
                             f'{t.__class__.__name__}, colorspace {_get_colorspace_name(c[1])}',
                             (20, 30),
                             fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0))
        frames.append(result)

    mimsave('result.gif', frames, fps=1)


if __name__ == '__main__':
    save_gif()
