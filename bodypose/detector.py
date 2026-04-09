import numpy as np
import torch

from . import util
from .model import BodyPose25
from .dtype import Skeleton

_try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BodyPoseDetector:
    def __init__(self, weights_path: str):
        self.model = BodyPose25()
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.model.eval()

    def __call__(self, img: np.ndarray, device: torch.device = _try_cuda) -> list[Skeleton]:
        threshold_joint = 0.1
        threshold_limb = 0.05

        with torch.no_grad():
            self.model.to(device)
            x = util.img2tsr(img).to(device)
            y = self.model(x)

            heatmap, paf = util.postprocess_output(y, img.shape[:2])
            joints = util.nms_heatmap(heatmap, threshold_joint)
            limbs = util.match_limbs(joints, paf, threshold_limb)
            skeletons = util.rebuild_skeletons(limbs)
        return skeletons
