import numpy as np


class Joint:
    def __init__(self, x=0, y=0, score=0):
        self.array = np.array((x, y, score), dtype=np.float32)

    @property
    def x(self):
        return self.array[0]

    @property
    def y(self):
        return self.array[1]

    @property
    def score(self):
        return self.array[2]

    @property
    def xy(self):
        return self.array[:2]

    def get_image_coord(self):
        return self.array[:2].astype(np.int32)


class Skeleton:
    def __init__(self):
        self.num_joints = 0
        self.score = 0.
        self.joints = [Joint()] * 25

    def numpy(self) -> np.ndarray:
        return np.array([joint.array for joint in self.joints])

    def resize(self, scale):
        for joint in self.joints:
            joint.array[:2] *= scale

    def __getitem__(self, item):
        return self.joints[item]

    def __setitem__(self, key, value: Joint):
        self.joints[key] = value

    def translate(self, offset):
        for joint in self.joints:
            joint.array[:2] += offset
