# QuDiDA (QUick and DIrty Domain Adaptation)

[![Upload release to PyPI](https://github.com/arsenyinfo/qudida/actions/workflows/upload_to_pypi.yml/badge.svg)](https://github.com/arsenyinfo/qudida/actions/workflows/upload_to_pypi.yml)

QuDiDA is a micro library for very naive though quick pixel level image domain adaptation via `scikit-learn` transformers.
Is assumed to be used as image augmentation technique, while was not tested in public benchmarks.

## Installation

```console
pip install qudida
```

or

```console
pip install git+https://github.com/arsenyinfo/qudida
```

## Usage

```python
import cv2

from sklearn.decomposition import PCA
from qudida import DomainAdapter

adapter = DomainAdapter(transformer=PCA(n_components=1), ref_img=cv2.imread('target.png'))
source = cv2.imread('source.png')
result = adapter(source)
cv2.imwrite('../result.png', result)
```

## Example

Source image:
![source](assets/source.png)
Target image (style donor):
![target](assets/target.png)
Result with various adaptations:
![result](assets/result.gif)
