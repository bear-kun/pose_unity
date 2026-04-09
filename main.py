from bodypose import PoseBody25Detector
from bodypose.camera import RealSenseCamera
from bodypose.util import draw_body_pose
import cv2
import time

if __name__ == '__main__':
    camera = RealSenseCamera()
    detector = PoseBody25Detector("weights/pose_body25.pt")
    K = camera.get_intrinsics()

    while True:
        start_time = time.time()
        color_img, depth_img = camera.get_frame()

        if color_img is None:
            continue

        skels = detector(color_img)

        for skel in skels:
            for joint in skel.joints:
                x2d, y2d = int(joint.x), int(joint.y)
                if 0 <= x2d < depth_img.shape[1] and 0 <= y2d < depth_img.shape[0]:
                    z = depth_img[y2d, x2d]
                    if z <= 0:
                        continue  # 无效深度
                    X = (x2d - K[0, 2]) * z / K[0, 0]
                    Y = (y2d - K[1, 2]) * z / K[1, 1]

        draw_body_pose(color_img, skels)
        cv2.imshow("D435i + OpenPose", color_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        end_time = time.time()
        print("fps: ", 1. / (end_time - start_time))

    camera.stop()
    cv2.destroyAllWindows()