import json
import os
from pathlib import Path
from typing import Optional, Union

HOME_DIR = Path(os.path.expanduser('~'))
THIS_DIR = Path(__file__).parent
DEFAULT_SETTINGS = THIS_DIR.joinpath("spell_detector_settings.txt")


class ControllerSettings:
    def __init__(self, settings_file: Optional[Path] = DEFAULT_SETTINGS):
        self._settings_file = settings_file
        self._settings = self.get_settings()

    def get_settings(self) -> dict:
        with open(self._settings_file, 'r') as fp:
            settings = json.load(fp)
            print(f"Settings: {settings}")
        return settings

    def _update_settings(self, setting: str, val: Union[bool, Path]):
        self._settings[setting] = val
        with open(self._settings_file, 'w') as fp:
            json.dump(self._settings, fp)

    @property
    def save_traces(self):
        return self._settings.get("save_traces")

    @save_traces.setter
    def save_traces(self, is_save):
        self._update_settings("save_traces", is_save)

    @property
    def traces_dir(self):
        return self._settings.get("traces_dir")

    @traces_dir.setter
    def traces_dir(self, directory):
        self._update_settings("traces_dir", directory)

    @property
    def show_video(self):
        return self._settings.get("show_video")

    @show_video.setter
    def show_video(self, is_show):
        self._update_settings("show_video", is_show)
