import cv2
import numpy as np
import math

from scipy.ndimage.filters import gaussian_filter
import torch

from .dtype import Joint, Skeleton

# 遍历时到关节点和paf_xy的映射
# 从颈部向四肢和头部关节点的拓扑排序，保证遍历时，当前连接不可能指向已遍历过的关节点
map2joints = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
              [13, 14], [1, 0], [0, 15], [15, 17],
              [0, 16], [16, 18], [2, 17], [5, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]

map2paf = [[0, 1], [14, 15], [22, 23], [16, 17], [18, 19], [24, 25], [26, 27], [6, 7], [2, 3], [4, 5], [8, 9], [10, 11],
           [12, 13], [30, 31], [32, 33],
           [36, 37], [34, 35], [38, 39], [20, 21], [28, 29], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], [50, 51]]

map2str = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
           "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
           "RBigToe", "RSmallToe", "RHeel", "Background"]


def draw_body_pose(img: np.ndarray, skeletons: list[Skeleton]):
    stick_width = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 0], [255, 255, 85], [255, 255, 170],
              [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255]]
    for skel in skeletons:
        for joint, color in zip(skel.joints, colors):
            if joint.score < 0.1:
                continue
            cv2.circle(img, joint.get_image_coord(), 4, color, thickness=-1)

    for skel in skeletons:
        for limb, color in zip(map2joints, colors):
            joint0 = skel[limb[0]]
            joint1 = skel[limb[1]]
            if joint0.score < 0.1 or joint1.score < 0.1:
                continue
            cur_canvas = img.copy()
            x0, y0 = joint0.get_image_coord()
            x1, y1 = joint1.get_image_coord()
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            polygon = cv2.ellipse2Poly(((x0 + x1) // 2, (y0 + y1) // 2), (int(length / 2), stick_width), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)

    return img


def pad_down_right_corner(img: np.ndarray) -> np.ndarray:
    stride = 8
    pad_val = 128

    h, w, _ = img.shape
    pad_d = 0 if h % stride == 0 else stride - h % stride
    pad_r = 0 if w % stride == 0 else stride - w % stride

    if pad_d == 0 and pad_r == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_d, 0, pad_r, cv2.BORDER_CONSTANT, value=(pad_val, pad_val, pad_val))


def img2tsr(img: np.ndarray) -> torch.Tensor:
    padded_img = pad_down_right_corner(img)
    data = np.transpose(np.float32(padded_img), (2, 0, 1)) / 256 - 0.5  # H, W, C -> C, H, W
    data = np.ascontiguousarray(data)  # 内存连续

    tsr = torch.from_numpy(data)
    tsr.unsqueeze_(0)
    return tsr


def postprocess_heatmap_paf(heatmap: torch.Tensor, paf: torch.Tensor, hw: tuple[int, int]):
    def _process(_x: torch.Tensor):
        _y = torch.nn.functional.interpolate(_x, hw, mode='bilinear')
        _y = _y.cpu().numpy()
        return _y.squeeze()

    return _process(heatmap), _process(paf)


# 返回所有关节点的候选点列表
# return (NUM_JOINTS, matched_joints)
def nms_heatmap(heatmaps: np.ndarray, threshold: float) -> list[list[Joint]]:
    joints = []

    for heatmap in heatmaps[:25]:
        smooth_heatmap = gaussian_filter(heatmap, sigma=3)

        map_left = np.zeros(smooth_heatmap.shape)
        map_left[1:, :] = smooth_heatmap[:-1, :]
        map_right = np.zeros(smooth_heatmap.shape)
        map_right[:-1, :] = smooth_heatmap[1:, :]
        map_up = np.zeros(smooth_heatmap.shape)
        map_up[:, 1:] = smooth_heatmap[:, :-1]
        map_down = np.zeros(smooth_heatmap.shape)
        map_down[:, :-1] = smooth_heatmap[:, 1:]

        # 求热图中超过阈值的峰值点，作为关节点的候选点
        peaks_binary = np.logical_and.reduce(
            (smooth_heatmap > threshold, smooth_heatmap >= map_left, smooth_heatmap >= map_right,
             smooth_heatmap >= map_up, smooth_heatmap >= map_down))  # 逻辑与

        peaks = np.argwhere(peaks_binary)
        candidate_joints = [Joint(x, y, heatmap[y, x].item()) for y, x in peaks]
        joints.append(candidate_joints)

    return joints


class Limb:
    def __init__(self, joint0, joint1, score):
        self.j0 = joint0
        self.j1 = joint1
        self.score = score

    def __iter__(self):
        yield self.j0
        yield self.j1
        yield self.score


# return (NUM_LIMBS, matched_limbs)
def match_limbs(joints: list[list[Joint]], paf: np.ndarray, threshold) -> list[list[Limb]]:
    limbs = []

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
                    num_dx = 10
                    bound = zip(np.linspace(joint0.x, joint1.x, num=num_dx),
                                np.linspace(joint0.y, joint1.y, num=num_dx))
                    vec_paf = np.array([paf_xy[:, int(round(y)), int(round(x))] for x, y in bound])
                    cos_vec = np.multiply(vec_paf, vec).sum(axis=1) # cos <vec, vec_paf>
                    integral = cos_vec.mean().item()

                    score = integral # + min(ori_img_w / 2. / norm - 1., 0.)
                    if score > 0. and len(np.argwhere(cos_vec > threshold)) > 0.8 * num_dx:
                        cand_limbs.append(Limb(joint0, joint1, score))

            cand_limbs = sorted(cand_limbs, key=lambda x: x.score, reverse=True)
            matched_limbs = []
            matched_joints = []
            for limb in cand_limbs:
                if id(limb.j0) not in matched_joints and id(limb.j1) not in matched_joints:
                    matched_limbs.append(limb)
                    matched_joints.append(id(limb.j0))
                    matched_joints.append(id(limb.j1))

            limbs.append(matched_limbs)

    return limbs


def rebuild_skeletons(limbs: list[list[Limb]]) -> list[Skeleton]:
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


def visualize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    img = heatmap.clip(min=0, max=1)
    img *= 255
    return img.astype(np.uint8)


def visualize_paf(paf_x: np.ndarray, paf_y: np.ndarray) -> np.ndarray:
    h, w = paf_x.shape
    step = 8
    threshold = 0.1

    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            vx, vy = paf_x[y, x], paf_y[y, x]
            magnitude = np.sqrt(vx ** 2 + vy ** 2)

            if magnitude > threshold:
                x_end = int(x + vx * step)
                y_end = int(y + vy * step)
                cv2.arrowedLine(img, (x, y), (x_end, y_end), (255,), 1, tipLength=0.3)

    return img


def show_heatmaps_paf(heatmap: np.ndarray, paf: np.ndarray):
    import matplotlib.pyplot as plt

    heatmap_title_image = [(f'heatmap {_l}', visualize_heatmap(_h)) for _l, _h in zip(map2str, heatmap)]
    paf_title_image = [(f'paf {map2str[start]}-{map2str[end]}', visualize_paf(paf[_x], paf[_y])) for
                       (start, end), (_x, _y) in
                       zip(map2joints, map2paf)]

    flg, axes = plt.subplots(2, 26, figsize=(52, 4))

    for col in range(26):
        axes[0, col].imshow(heatmap_title_image[col][1])
        axes[0, col].set_title(heatmap_title_image[col][0])
        axes[1, col].imshow(paf_title_image[col][1])
        axes[1, col].set_title(paf_title_image[col][0])

    plt.tight_layout()
    plt.show()
