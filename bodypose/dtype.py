import numpy as np


class Joint:
    def __init__(self, x=-1, y=-1, score: float = 0.):
        self.data = np.array((x, y, score), dtype=np.float32)

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def score(self):
        return self.data[2]

    @property
    def xy(self):
        return self.data[:2]

    def get_image_coord(self):
        # 四舍五入再转整型，避免直接截断导致整体向左/上偏移
        return np.rint(self.data[:2]).astype(np.int32)


class Skeleton:
    def __init__(self, joints: list[Joint] = None, num_joints: int = 0, score: float = 0.):
        self.num_joints = num_joints
        self.score = score
        if joints is None:
            self.joints = [Joint()] * 25
        else:
            self.joints = joints

    def get_joints_coord(self) -> np.ndarray:
        return np.array([joint.xy for joint in self.joints])

    def resize(self, scale):
        # 支持统一缩放因子或 (scale_x, scale_y) 两个方向的缩放
        if isinstance(scale, (list, tuple, np.ndarray)):
            sx, sy = scale[0], scale[1]
            for joint in self.joints:
                joint.data[0] *= sx
                joint.data[1] *= sy
        else:
            for joint in self.joints:
                joint.data[:2] *= scale

    def __getitem__(self, item):
        return self.joints[item]

    def __setitem__(self, key, value: Joint):
        self.joints[key] = value

    def translate(self, dx: float, dy: float):
        for joint in self.joints:
            joint.data[0] += dx
            joint.data[1] += dy
