# CLAHE
[![codecov](https://codecov.io/gh/ykszk/clahe/branch/main/graph/badge.svg?token=6288CW47HH)](https://codecov.io/gh/ykszk/clahe)

📊 Contrast Limited Adaptive Histogram Equalization

👽 When maximum pixel value of the input image is smaller than u16::MAX, output would be different from OpenCV.

⚠️ Lots of `unsafe` code was used since safe version was slower than unsafe version (current implementation).