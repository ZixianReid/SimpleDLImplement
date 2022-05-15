import numpy as np
import six


def generate_anchor_base(base_size=16, ratios=None, anchor_scale=None):
    if anchor_scale is None:
        anchor_scale = [8, 16, 32]
    if ratios is None:
        ratios = [0.5, 1, 2]

    py = base_size / 2
    px = base_size / 2

    anchor_base = np.zeros((len(ratios) * len(anchor_scale), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scale)):
            h = base_size * anchor_scale[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scale[j] * np.sqrt(ratios[i])

            index = i * len(anchor_scale) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


if __name__ == '__main__':
    aa = generate_anchor_base()
    bb = enumerate_shifted_anchor(aa, 16, 18, 25)
    pass