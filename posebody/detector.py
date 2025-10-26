import numpy as np
import torch
from .model import PoseBody25
from .dtype import Skeleton
from . import util

_try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseBody25Detector:
    def __init__(self, weights_path: str):
        self.num_joints = 25
        self.num_heatmap = 26
        self.num_paf = 52

        self._model = PoseBody25()
        self._model.load_state_dict(torch.load(weights_path, weights_only=True))
        self._model.eval()

    def __call__(self, ori_img: np.ndarray, show_heatmap_paf: bool = False, device: torch.device = _try_cuda) -> list[Skeleton]:
        threshold_joint = 0.1
        threshold_limb = 0.05

        with torch.no_grad():
            self._model.to(device)
            x = util.img2tsr(ori_img).to(device)
            heatmap, paf = self._model(x)

        heatmap, paf = util.postprocess_heatmap_paf(heatmap, paf, ori_img.shape[:2])
        joints = util.nms_heatmap(heatmap, threshold_joint)
        limbs = util.match_limbs(joints, paf, threshold_limb)
        skeletons = util.rebuild_skeletons(limbs)

        if show_heatmap_paf:
            util.show_heatmaps_paf(heatmap, paf)
        return skeletons