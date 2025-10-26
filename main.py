from posebody import PoseBody25Detector
from posebody.util import draw_body_pose
import cv2

detector = PoseBody25Detector('weights/pose_body25.pt')
img = cv2.imread('images/img0.png')
skels = detector(img)
draw_body_pose(img, skels)
cv2.imshow('test', img)
cv2.waitKey()