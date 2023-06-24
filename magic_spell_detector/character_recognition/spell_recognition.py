import enum
import cv2
import os
# import sys
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from functools import lru_cache
from time import gmtime, strftime
from . import utils

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR.joinpath('new_data')
TRAINING_IMG_DIR = DATA_DIR.joinpath('training_images')
TEST_IMG_DIR = DATA_DIR.joinpath('test_images')
NEW_SPELLS_SVM_MODEL = DATA_DIR.joinpath('new_spells_svm.dat')

UNIT_IMG_SZ = 64
NUMBER_OF_SPELLS = 2


log = logging.getLogger(__name__)


@lru_cache(1)
def get_hog():
    win_size = (20, 20)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (10, 10)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1.
    histogram_normtype = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    use_signed_gradients = True

    return cv2.HOGDescriptor(win_size, block_size,
                             block_stride, cell_size,
                             nbins, deriv_aperture,
                             win_sigma, histogram_normtype,
                             l2_hys_threshold, gamma_correction,
                             nlevels, use_signed_gradients)


def get_svm_model(model_file: Optional[Path] = None):
    if model_file:
        return cv2.ml.SVM_load(str(model_file))
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    return svm


class Spells(int, enum.Enum):
    MUSIC_SPELL = 0
    LUMOS = 1
    NONE = 3


TRAINING_IMAGES = {
    Spells.MUSIC_SPELL: os.listdir(TRAINING_IMG_DIR.joinpath('MUSIC_SPELL')),
    Spells.LUMOS: os.listdir(TRAINING_IMG_DIR.joinpath('LUMOS')),
}

TEST_IMAGES = {
    Spells.MUSIC_SPELL: os.listdir(TEST_IMG_DIR.joinpath('MUSIC_SPELL')),
    Spells.LUMOS: os.listdir(TEST_IMG_DIR.joinpath('LUMOS')),
}


class SpellRecognition:
    def __init__(self):
        self._hog = get_hog()
        self._svm = get_svm_model(NEW_SPELLS_SVM_MODEL)

    def recognize(self, trace_image: np.ndarray) -> Spells:
        """ Crop, resize & predict spell using SVM. """
        # Resize
        trace = self._resize_crop_image(trace_image)
        # If not RPi
        # if not (sys.platform.startswith('linux') and os.uname()[1] == 'raspberrypi'):
        #     cv2.imshow("Cropped_resized", trace)
        possible_spell = self._recognize(trace)
        return Spells(possible_spell[0])

    def _recognize(self, trace) -> np.ndarray:
        """
        Deskew the resized trace, compute HOG descriptors, predict.
        """
        # Deskew trace
        deskewed_trace = utils.deskew(trace, cell_size=UNIT_IMG_SZ)
        # Compute HOG
        trace_desc = self._hog.compute(deskewed_trace)
        trace_desc2 = trace_desc.T
        # Get prediction
        prediction = self._svm.predict(trace_desc2)[1].ravel()
        log.debug(f"SVM model prediction: {prediction}")
        return prediction

    def _resize_crop_image(self, trace_image: np.ndarray) -> np.ndarray:
        """
        Crop to trace & pad the shorter side of the cropped trace so that
        the sides are equal. Then resize to a 64x64 array.
        """
        # Crop trace to remove irrelevant area
        cropped_trace = self._crop_trace(trace_image)

        # Make it a square image. Pad the smaller dimension with zeros
        diff = cropped_trace.shape[0] - cropped_trace.shape[1]
        padding = int(abs(diff) / 2)
        if diff > 0:
            """ Height > Width """
            cropped_trace = np.pad(cropped_trace, ((0, 0), (padding, padding)), 'constant')
        else:
            """ Width > Height"""
            cropped_trace = np.pad(cropped_trace, ((padding, padding), (0, 0)), 'constant')

        # Resize to set width x height
        return cv2.resize(cropped_trace, dsize=(UNIT_IMG_SZ, UNIT_IMG_SZ))

    def _crop_trace(self, trace_image: np.ndarray) -> np.ndarray:
        """
        Crop to edge-to-edge of the trace, and
        add 10 pixels padding to the longer of the sides.
        """
        row_has_trace = list(map(lambda row: row.any(), trace_image))
        col_has_trace = list(map(lambda col: col.any(), trace_image.T))  # Transpose for columns
        x0, x1 = self._get_first_and_last_indices(col_has_trace)
        y0, y1 = self._get_first_and_last_indices(row_has_trace)

        if x1 - x0 <= y1 - y0:
            # add margins to height
            y0 = max(0, y0 - 5)
            y1 = min(y1 + 5, trace_image.shape[0])
        else:
            # add margins to width
            x0 = max(0, x0 - 5)
            x1 = min(x1 + 5, trace_image.shape[1])
        cropped_trace = trace_image[y0:y1, x0:x1]
        return cropped_trace

    def _get_first_and_last_indices(self, has_trace_list: list) -> Tuple[int, int]:
        """ Get the first and last indices of trace occurrence in frame sub-array

        :param has_trace_list: list of whether a non-zero number exists in the subarray
        :return: location of first & last non-zero numbers in list
        """
        first = has_trace_list.index(True)
        last = len(has_trace_list) - list(has_trace_list.__reversed__()).index(True) - 1
        return first, last

    def save_trace_for_training(self, trace: np.ndarray,
                                loc: Optional[Path] = DATA_DIR):
        """ Save trace to file. """
        cropped_resized = self._resize_crop_image(trace)
        timestamp = strftime("%d_%b_%H_%M_%S", gmtime())
        cv2.imwrite(str(f"{loc}/img_{timestamp}.png"), cropped_resized)
        log.debug(f"Saved image img{timestamp}")


class SpellTrainer:
    def __init__(self):
        self._svm = get_svm_model()
        self._hog = get_hog()

    def prep_images(self, images: Dict[Spells, List[str]],
                    image_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Takes a dictionary of locations of training or test images keyed by spells,
         and does the following:
        - Read in the images to ndarrays
        - Deskews the images and adds them to a list
        - Creates labels corresponding to the images
        """
        deskewed_list = []
        labels_list = []
        for spell, spell_images in images.items():
            print(f"Getting images from: {spell_images}")
            for img_file in spell_images:
                if not img_file.endswith('.png'):
                    continue    # Skip any auto-generated system files
                log.debug(f"Adding image: {img_file}")
                img = cv2.imread(str(image_dir.joinpath(spell.name, img_file)), 0)
                deskewed_list.append(utils.deskew(img, cell_size=UNIT_IMG_SZ))
                labels_list.append(spell.value)
        log.debug(f"Labels for the images: {labels_list}")
        return deskewed_list, np.array(labels_list)

    def compute_hog(self, deskewed_images: List[np.ndarray]) -> np.ndarray:
        """ Get HOGs of deskewed images. """
        descriptors = []
        for img in deskewed_images:
            descriptors.append(self._hog.compute(img))
        squeezed_descriptors = np.squeeze(descriptors)
        return squeezed_descriptors

    def load(self, svm_file: Path):
        self._svm.load(str(svm_file))

    def save(self, filename: Path):
        self._svm.save(str(filename))

    def predict(self, samples: np.ndarray):
        return self._svm.predict(samples)[1].ravel()

    def train(self, samples: np.ndarray, labels: np.ndarray):
        self._svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)

    def evaluate_model(self, digits: list, hog_samples: np.ndarray, labels: List[int]):
        """ Run predictions on the input samples and:
        - Get the response matrix & confusion matrix
        - Log accuracy of prediction by comparing with provided labels
        - Visualize correct & incorrect predictions by showing them in
          white & red color respectively.
        """

        resp = self.predict(hog_samples)
        print(f"Response matrix: {resp}")
        err = (labels != resp).mean()
        print('Accuracy: %.2f %%' % ((1 - err) * 100))

        confusion = np.zeros((10, 10), np.int32)
        for i, j in zip(labels, resp):
            confusion[int(i), int(j)] += 1
        print(f'confusion matrix: {confusion}')

        vis = []
        for img, flag in zip(digits, resp == labels):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if not flag:
                img[..., :2] = 0

            vis.append(img)
        return utils.mosaic(25, vis)


if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)

    """ --------- Training ------- """
    trainer = SpellTrainer()
    deskewed_training_images, training_labels = trainer.prep_images(TRAINING_IMAGES,
                                                                    TRAINING_IMG_DIR)
    hog_descriptors = trainer.compute_hog(deskewed_training_images)
    trainer.train(hog_descriptors, training_labels)
    trainer.save(NEW_SPELLS_SVM_MODEL)
    print("Done training.")

    """ --------- Testing -------- """
    deskewed_test_images, test_labels = trainer.prep_images(TEST_IMAGES,
                                                            TEST_IMG_DIR)
    computed_hogs = trainer.compute_hog(deskewed_test_images)
    vis = trainer.evaluate_model(deskewed_test_images,
                                 computed_hogs,
                                 list(test_labels))
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)

