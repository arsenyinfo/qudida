import cv2
from imageio import mimsave

from qudida import DomainAdapter
from test_transforms import params_combinations


def save_gif():
    frames = []
    for t, c, k in params_combinations():
        adapter = DomainAdapter(transformer=t,
                                ref_img=cv2.imread('target.png'),
                                kernel_size=k,
                                color_conversions=c,
                                )
        source = cv2.imread('source.png')
        result = adapter(source)
        result = cv2.putText(result,
                             f'{t.__class__.__name__}, kernel size {k}, color {"RGB" if c[0] is None else "YCrCb"}',
                             (20, 30),
                             fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0))
        frames.append(result)

    mimsave('result.gif', frames, fps=1)


if __name__ == '__main__':
    save_gif()
