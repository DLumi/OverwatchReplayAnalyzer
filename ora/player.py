import cv2
import numpy as np
import skimage
# from skimage import measure
from skimage.metrics import structural_similarity
from dataclasses import dataclass

# from . import overwatch as OW
from .utils import image as ImageUtils

from .hero import Hero


@dataclass(frozen=True)
class KFPlayer:
    team: int
    hero: Hero

    def to_dict(self):
        d = {
            'team': self.team,
            'hero': self.hero,
        }
        return d
