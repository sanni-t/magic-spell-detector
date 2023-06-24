# MAGIC SPELL DETECTOR

This package provides a tool to perform detection of spells cast using a wand whose tip can be detected reliably by a camera.
Example wand+camera systems are:
 - wand with LED tip or other concentrated, point source of light + any web camera
 - wand with an IR reflective tip + IR light source pointing towards you + a camera capable of seeing IR light

## Installation:

- Clone or download this repository to your local machine
- Install python v3.8+. 
- Recommended: [install pyenv](https://github.com/pyenv/pyenv#installation) so that you can run the project in virtual environment and control the dependencies' versions.
- [Install Poetry](https://python-poetry.org/docs/) - a python package management tool
- In terminal/ commandline app, navigate to the root of the project on your local machine and enter `poetry install`. This should
install all the dependencies for the project

## Usage:

- Run the spell detector using `poetry run python start_detection.py`
