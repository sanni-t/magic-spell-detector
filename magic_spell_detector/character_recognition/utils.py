import cv2
import numpy as np
import itertools as it


def deskew(img: np.ndarray, cell_size: int):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * cell_size * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (cell_size, cell_size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def split2d(img: np.ndarray, cell_size: int, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def load_digits(img_file: str, cell_size: int, unique_samples: int):
    digits_img = cv2.imread(img_file, 0)
    digits = split2d(digits_img, (cell_size, cell_size))
    labels = np.repeat(np.arange(unique_samples), len(digits) / unique_samples)
    return digits, labels


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    output = it.zip_longest(fillvalue=fillvalue, *args)
    return output


def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = next(imgs)
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(list(map(np.hstack, rows)))


if __name__ == '__main__':
    pass
