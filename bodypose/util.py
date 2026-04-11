import math
import time

import cv2
import numpy as np
import torch
from torch.nn.functional import max_pool2d
from torchvision.transforms.functional import gaussian_blur

from .dtype import Joint, Skeleton

# 遍历时到关节点和paf_xy的映射
# 从颈部向四肢和头部关节点的拓扑排序，保证遍历时，当前连接不可能指向已遍历过的关节点
map2joints = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
              [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19], [19, 20], [14, 21],
              [11, 22], [22, 23], [11, 24]]

map2paf = [[0, 1], [14, 15], [22, 23], [16, 17], [18, 19], [24, 25], [26, 27], [6, 7], [2, 3], [4, 5], [8, 9], [10, 11],
           [12, 13], [30, 31], [32, 33], [36, 37], [34, 35], [38, 39], [20, 21], [28, 29], [40, 41], [42, 43], [44, 45],
           [46, 47], [48, 49], [50, 51]]

map2str = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
           "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
           "RBigToe", "RSmallToe", "RHeel", "Background"]


def draw_body_pose(img: np.ndarray, skeletons: list[Skeleton]) -> np.ndarray:
    stick_width = 4
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 0], [255, 255, 85], [255, 255, 170],
              [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255]]

    res = img.copy()
    for skel in skeletons:
        for limb, color in zip(map2joints, colors):
            joint0 = skel[limb[0]]
            joint1 = skel[limb[1]]
            if joint0.score < 0.1 or joint1.score < 0.1:
                continue
            x0, y0 = joint0.get_image_coord()
            x1, y1 = joint1.get_image_coord()
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            polygon = cv2.ellipse2Poly(((x0 + x1) // 2, (y0 + y1) // 2), (int(length / 2), stick_width), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(res, polygon, color)

    res = cv2.addWeighted(res, 0.4, res, 0.6, 0)

    for skel in skeletons:
        for joint, color in zip(skel.joints, colors):
            if joint.score < 0.1:
                continue
            cv2.circle(res, joint.get_image_coord(), 4, color, thickness=-1)

    return res


def pad_down_right_corner(img: np.ndarray) -> np.ndarray:
    stride = 16
    pad_val = 128

    h, w, _ = img.shape
    pad_d = stride - (h & (stride - 1))
    pad_r = stride - (w & (stride - 1))

    if pad_d == 0 and pad_r == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_d, 0, pad_r, cv2.BORDER_CONSTANT, value=(pad_val, pad_val, pad_val))


def img2tsr(img: np.ndarray, wh) -> torch.Tensor:
    resized = cv2.resize(img, wh)
    padded = pad_down_right_corner(resized)
    data = np.transpose(np.float32(padded), (2, 0, 1)) / 256 - 0.5  # H, W, C -> C, H, W
    data = np.ascontiguousarray(data)  # 内存连续
    return torch.from_numpy(data).unsqueeze_(0)


def postprocess_output(output, hw):
    result = torch.nn.functional.interpolate(output, hw, mode='bilinear').squeeze_(0)
    return result[:26], result[26:]


# 返回所有关节点的候选点列表
# return (NUM_JOINTS, matched_joints)
def nms_heatmap(heatmaps: torch.Tensor, threshold: float):
    smooth = gaussian_blur(heatmaps[:25], kernel_size=[5, 5])
    pool = max_pool2d(smooth, kernel_size=3, stride=1, padding=1)
    mask = (smooth > threshold) & (smooth == pool)

    indices = torch.nonzero(mask, as_tuple=True)
    scores = heatmaps[indices]

    indices = tuple(idx.cpu().numpy() for idx in indices)
    scores = scores.cpu().numpy()

    joints = [[] for _ in range(25)]
    for i in range(len(scores)):
        j = indices[0][i]
        y = indices[1][i]
        x = indices[2][i]
        s = scores[i]
        joints[j].append(Joint(x, y, s))

    return joints


class Limb:
    def __init__(self, joint0, joint1, score):
        self.joint0 = joint0
        self.joint1 = joint1
        self.score = score

    def __iter__(self):
        yield self.joint0
        yield self.joint1
        yield self.score


# return (NUM_LIMBS, matched_limbs)
def match_limbs(joints, paf: torch.Tensor, threshold):
    num_dx = 10
    limbs = []
    paf = paf.to("cpu", non_blocking=True).numpy()

    # 用关节点向量和paf匹配，得到候选（candidate）躯干
    for joints_idx, paf_idx in zip(map2joints, map2paf):
        paf_xy = paf[paf_idx]
        cand_joint0 = joints[joints_idx[0]]
        cand_joint1 = joints[joints_idx[1]]

        if not cand_joint0 or not cand_joint1:
            limbs.append([])
        else:
            cand_limbs = []
            for joint0 in cand_joint0:
                for joint1 in cand_joint1:
                    # 计算单位向量
                    vec = np.subtract(joint1.xy, joint0.xy)
                    norm = np.linalg.norm(vec).item()
                    if norm == 0.:
                        continue
                    vec = np.divide(vec, norm)

                    # 积分
                    bound = zip(np.linspace(joint0.x, joint1.x, num=num_dx),
                                np.linspace(joint0.y, joint1.y, num=num_dx))
                    vec_paf = np.array([paf_xy[:, int(round(y)), int(round(x))] for x, y in bound])
                    cos_vec = np.multiply(vec_paf, vec).sum(axis=1)  # cos <vec, vec_paf>
                    integral = cos_vec.mean().item()

                    score = integral  # + min(ori_img_w / 2. / norm - 1., 0.)
                    if score > 0. and np.sum(cos_vec > threshold) > 0.8 * num_dx:
                        cand_limbs.append(Limb(joint0, joint1, score))

            cand_limbs = sorted(cand_limbs, key=lambda x: x.score, reverse=True)
            matched_limbs = []
            matched_joints = []
            for limb in cand_limbs:
                if id(limb.joint0) not in matched_joints and id(limb.joint1) not in matched_joints:
                    matched_limbs.append(limb)
                    matched_joints.append(id(limb.joint0))
                    matched_joints.append(id(limb.joint1))

            limbs.append(matched_limbs)

    return limbs


def rebuild_skeletons(limbs) -> list[Skeleton]:
    cand_skels = []

    # 躯干尝试搭建骨架
    for (joint0_idx, joint1_idx), cand_limbs in zip(map2joints, limbs):
        if not cand_limbs:
            continue
        for joint0, joint1, score in cand_limbs:
            for cand_skel in cand_skels:
                if cand_skel[joint0_idx] is joint0:
                    # 因为拓扑排序，只可能是id0出现重复，
                    # 而一般id1不可能出现在之前的骨架中，
                    # 除非允许两个骨架的同一个关节位置共用一个关节点
                    cand_skel[joint1_idx] = joint1
                    cand_skel.num_joints += 1
                    cand_skel.score += score
                    break
            else:
                cand_skel = Skeleton()
                cand_skel[joint0_idx] = joint0
                cand_skel[joint1_idx] = joint1
                cand_skel.num_joints = 2
                cand_skel.score = joint0.score + joint1.score + score
                cand_skels.append(cand_skel)

    skeletons = []
    for cand_skel in cand_skels:
        if cand_skel.score >= 4. and cand_skel.num_joints / cand_skel.score >= 0.4:
            skeletons.append(cand_skel)

    return skeletons
