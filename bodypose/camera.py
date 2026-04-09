import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self,
                 width=640,
                 height=480,
                 fps=30,
                 enable_color=True,
                 enable_depth=True,
                 align_to_color=True):
        """
        Intel RealSense D435i 相机封装
        - 支持调整分辨率与帧率
        - 默认启用彩色+深度流
        - 不使用 IMU
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align_to_color = align_to_color
        self.enable_color = enable_color
        self.enable_depth = enable_depth

        # 配置流
        if enable_color:
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 启动相机
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # 深度值单位(米)

        # 深度对齐到彩色
        self.align = rs.align(rs.stream.color) if align_to_color else None

    def get_intrinsics(self):
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float32)

    def get_frame(self):
        """
        获取一帧 RGB + 深度数据
        返回: color_img (H×W×3), depth_img (H×W, 单位: 米), K (内参矩阵)
        """
        frames = self.pipeline.wait_for_frames()

        # 若启用对齐
        if self.align and self.enable_color and self.enable_depth:
            frames = self.align.process(frames)

        color_img = depth_img = None

        if self.enable_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                color_img = np.asanyarray(color_frame.get_data())

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data()) * self.depth_scale

        return color_img, depth_img

    def stop(self):
        """关闭相机"""
        self.pipeline.stop()
