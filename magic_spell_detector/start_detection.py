import time
import cv2
import logging
from typing import Optional
from logging.config import dictConfig
from argparse import ArgumentParser
from pathlib import Path
from threading import Thread

import numpy as np
from utils import ControllerSettings
from wand_tracer.wand_tracer import WandTracer, TraceStatus
from character_recognition.spell_recognition import Spells, SpellRecognition


VERSION = "1.1.0"
PRINT_PROPS = True
SPELL_CHECK = SpellRecognition()
WAND_TRACER = WandTracer()
SETTINGS = ControllerSettings()
FRAME_WIDTH = 640
VID_RESOLUTION = (640, 480)
FRAME_RATE = 11
THIS_DIR = Path(__file__).parent
NEW_IMAGE_SAVE_DIR = THIS_DIR.joinpath("new_data")


def spellcheck_and_perform_spell(trace: np.ndarray) -> None:
    spell = SPELL_CHECK.recognize(trace)

    if spell == Spells.NONE:
        print("Invalid wand movement.")
    else:
        if spell == Spells.MUSIC_SPELL:
            # Add Music action here
            print("RECEIVED MUSIC SPELL!")
        elif spell == Spells.LUMOS:
            # Add Lumos action here
            print("RECEIVED LUMOS!")
        print(f"A spell has been cast: {spell}")


def build_arg_parser():
    arg_parser = ArgumentParser(
        description="Magical music box controller")
    arg_parser.add_argument("-s", "--save_traces",
                            default=SETTINGS.save_traces,
                            help='Turn ON spell training.')
    arg_parser.add_argument("-d", "--dir", required=False,
                            default=SETTINGS.traces_dir,
                            help="Location for saving traces.")
    return arg_parser


def configure_logger(name):
    logging.config.dictConfig({
        'version': 1,
        'propagate': 1,
        'formatters': {
            'default': {'format': '%(levelname)s - %(message)s'}
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'sys.stdout'
            },
        },
        'loggers': {
            'default': {
                'level': 'INFO',
                'handlers': ['console']
            }
        },
        'disable_existing_loggers': False,

    })
    return logging.getLogger(name)


log = configure_logger(__name__)


def save_trace_if_valid(trace_status: TraceStatus, trace: np.ndarray,
                        trace_dir: Optional[Path] = NEW_IMAGE_SAVE_DIR):
    if trace_status == TraceStatus.INVALID_TRACE:
        print("Invalid trace.")
        WAND_TRACER.erase_trace()
    elif trace_status == TraceStatus.READY_FOR_SPELLCHECK:
        print(f"Valid trace, trying to save to {NEW_IMAGE_SAVE_DIR}..")
        SPELL_CHECK.save_trace_for_training(trace, NEW_IMAGE_SAVE_DIR)
        WAND_TRACER.erase_trace()


def start_spell_detection(args):
    global PRINT_PROPS
    log.info("WAND TRACER STARTS")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if frame is None:
            time.sleep(0.05)
            continue

        (h, w) = frame.shape[:2]
        # calculate the ratio of the width and construct the
        # dimensions
        r = FRAME_WIDTH / float(w)
        dim = (FRAME_WIDTH, int(h * r))

        # resize the image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        resized_flipped = cv2.flip(resized, 1)
        gray = cv2.cvtColor(resized_flipped, cv2.COLOR_BGR2GRAY)

        if PRINT_PROPS:
            PRINT_PROPS = False
            log.info(gray.shape)

        WAND_TRACER.trace_wand(gray)
        trace_status = WAND_TRACER.get_trace_status()
        trace = WAND_TRACER.draw_frame

        if args.save_traces:
            save_trace_if_valid(trace_status, trace)
        else:
            if trace_status == TraceStatus.INVALID_TRACE:
                print("Invalid trace status")
                WAND_TRACER.erase_trace()
            elif trace_status == TraceStatus.READY_FOR_SPELLCHECK:
                # create a thread
                spell_performer_thread = Thread(
                    target=spellcheck_and_perform_spell,
                    args=(trace,)
                )
                # run the thread
                spell_performer_thread.start()

                # wait for the thread to finish
                print('Waiting for the thread...')
                spell_performer_thread.join()

                print("Done with the thread")
                WAND_TRACER.erase_trace()

        # Show trace
        # Superimpose wand trace
        final_frame = WAND_TRACER.add_wand_trace(gray)

        cv2.imshow("Wand Tracer", final_frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
        time.sleep(0.047)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    log.info(f"Starting Magic Spell Detector v{VERSION}")
    parser = build_arg_parser()
    args = parser.parse_args()
    log.info(f"Commandline args: {args.save_traces}, {args.dir}")
    start_spell_detection(args)
