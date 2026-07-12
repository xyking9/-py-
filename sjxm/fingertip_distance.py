# -*- coding: utf-8 -*-
"""
摄像头实时指尖测距程序

功能说明：
    1. 调用笔记本内置摄像头实时读取视频流
    2. 使用 MediaPipe Tasks HandLandmarker 识别左右手食指指尖
    3. 计算两指尖之间的像素距离，并通过标定换算为真实厘米距离
    4. 画面左上角实时显示像素距离、厘米距离、检测状态
    5. 按 q 退出，按 c 进入标定模式
    6. 摄像头不可用时，可选择进入画板模式：在背景图上用鼠标点击两点测距

依赖库：opencv-python、mediapipe、numpy（math/json/os/urllib 为 Python 标准库）
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import json
import os
import urllib.request
import numpy as np

CALIBRATION_FILE = "calibration.json"
KNOWN_LENGTH_CM = 15.0
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# 画板模式参数
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600


def download_model():
    """下载 MediaPipe 手部关键点检测模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"正在下载模型文件: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"模型下载完成: {MODEL_PATH}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            return False
    return True


class FingertipDistanceMeasurer:
    """指尖测距核心类，封装摄像头读取、手部识别、距离计算与可视化"""

    def __init__(self):
        self.pixels_per_cm = None
        self.calibrating = False
        self.calibration_points = []

        # 模型加载失败时不中断程序，画板模式仍可使用
        self.hand_landmarker = None
        try:
            if download_model():
                self.hand_landmarker = vision.HandLandmarker.create_from_model_path(MODEL_PATH)
            else:
                print("警告：模型未加载，摄像头模式不可用，仅可使用画板模式")
        except Exception as e:
            print(f"警告：模型加载失败: {e}")
            print("摄像头模式不可用，仅可使用画板模式")

        self.cap = None
        self.left_index_tip = None
        self.right_index_tip = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = cv2.getTickCount()

        # 画板模式状态
        self.canvas_points = []
        self.canvas_mode = False

        self._load_calibration()

    def _process_frame(self, frame):
        """处理单帧画面：检测手部关键点，提取左右手食指指尖坐标"""
        self.left_index_tip = None
        self.right_index_tip = None

        if self.hand_landmarker is None:
            return

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.hand_landmarker.detect(mp_image)

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_label = handedness[0].category_name

                if len(hand_landmarks) > 8:
                    tip_landmark = hand_landmarks[8]
                    x = int(tip_landmark.x * w)
                    y = int(tip_landmark.y * h)

                    if hand_label == 'Left':
                        self.left_index_tip = (x, y)
                    elif hand_label == 'Right':
                        self.right_index_tip = (x, y)

    def _load_calibration(self):
        """从 JSON 文件加载历史标定数据"""
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.pixels_per_cm = data.get('pixels_per_cm')
                    if self.pixels_per_cm is not None:
                        print(f"已加载标定数据: {self.pixels_per_cm:.2f} 像素/厘米")
            except Exception as e:
                print(f"加载标定文件失败: {e}")

    def _save_calibration(self):
        """将标定数据保存到 JSON 文件，下次启动自动加载"""
        if self.pixels_per_cm is not None:
            try:
                with open(CALIBRATION_FILE, 'w') as f:
                    json.dump({'pixels_per_cm': self.pixels_per_cm}, f)
                print(f"标定数据已保存: {self.pixels_per_cm:.2f} 像素/厘米")
            except Exception as e:
                print(f"保存标定文件失败: {e}")

    def start_calibration(self):
        """启动标定模式，提示用户点击已知长度物体的两个端点"""
        print("=" * 60)
        print("标定模式已启动")
        print(f"请在画面中点击 {KNOWN_LENGTH_CM} 厘米长物体的两个端点")
        print("按 ESC 键取消标定")
        print("=" * 60)
        self.calibrating = True
        self.calibration_points = []

    def _calibration_callback(self, event, x, y, flags, param):
        """鼠标回调函数：标定模式下采集两个端点并计算换算比例"""
        if self.calibrating and event == cv2.EVENT_LBUTTONDOWN:
            self.calibration_points.append((x, y))
            print(f"已选择点 {len(self.calibration_points)}: ({x}, {y})")

            if len(self.calibration_points) >= 2:
                pixel_distance = math.hypot(
                    self.calibration_points[1][0] - self.calibration_points[0][0],
                    self.calibration_points[1][1] - self.calibration_points[0][1]
                )
                self.pixels_per_cm = pixel_distance / KNOWN_LENGTH_CM
                print(f"\n标定完成！")
                print(f"像素距离: {pixel_distance:.2f} 像素")
                print(f"实际距离: {KNOWN_LENGTH_CM} 厘米")
                print(f"换算比例: {self.pixels_per_cm:.2f} 像素/厘米")
                self._save_calibration()
                self.calibrating = False
                self.calibration_points = []

    def _canvas_mouse_callback(self, event, x, y, flags, param):
        """画板模式鼠标回调：采集两个点用于测距，或标定点采集"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 标定模式优先
        if self.calibrating:
            self.calibration_points.append((x, y))
            print(f"已选择点 {len(self.calibration_points)}: ({x}, {y})")
            if len(self.calibration_points) >= 2:
                pixel_distance = math.hypot(
                    self.calibration_points[1][0] - self.calibration_points[0][0],
                    self.calibration_points[1][1] - self.calibration_points[0][1]
                )
                self.pixels_per_cm = pixel_distance / KNOWN_LENGTH_CM
                print(f"\n标定完成！换算比例: {self.pixels_per_cm:.2f} 像素/厘米")
                self._save_calibration()
                self.calibrating = False
                self.calibration_points = []
            return

        # 测距点采集：超过2个点时重新开始
        if len(self.canvas_points) >= 2:
            self.canvas_points = []
        self.canvas_points.append((x, y))
        print(f"画板测距点 {len(self.canvas_points)}: ({x}, {y})")

    def _create_canvas_background(self):
        """生成带网格的画板背景图"""
        # 浅灰底色
        img = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 240, dtype=np.uint8)
        # 绘制网格线，间距 50 像素
        grid_color = (200, 200, 200)
        for x in range(0, CANVAS_WIDTH, 50):
            cv2.line(img, (x, 0), (x, CANVAS_HEIGHT), grid_color, 1)
        for y in range(0, CANVAS_HEIGHT, 50):
            cv2.line(img, (0, y), (CANVAS_WIDTH, y), grid_color, 1)
        # 边框
        cv2.rectangle(img, (0, 0), (CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1), (100, 100, 100), 2)
        return img

    def _draw_canvas(self, img):
        """在画板上绘制测距点、连线、距离信息"""
        h, w, _ = img.shape

        # 标定模式绘制
        if self.calibrating:
            cv2.putText(img, f"标定模式: 点击 {KNOWN_LENGTH_CM}cm 物体两端点", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for i, point in enumerate(self.calibration_points):
                cv2.circle(img, point, 8, (0, 255, 0), -1)
                cv2.putText(img, str(i + 1), (point[0] + 10, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if len(self.calibration_points) < 2:
                cv2.putText(img, f"还需点击 {2 - len(self.calibration_points)} 个点", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, "操作: q-退出  c-取消标定", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return

        # 测距点绘制
        pixel_distance = 0
        cm_distance = 0
        status_text = "点击两个点进行测距"

        if len(self.canvas_points) == 1:
            cv2.circle(img, self.canvas_points[0], 10, (0, 0, 255), -1)
            cv2.putText(img, "1", (self.canvas_points[0][0] + 15, self.canvas_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            status_text = "已选第1点，请点击第2点"
        elif len(self.canvas_points) == 2:
            p1, p2 = self.canvas_points
            cv2.circle(img, p1, 10, (0, 0, 255), -1)
            cv2.circle(img, p2, 10, (255, 0, 0), -1)
            cv2.line(img, p1, p2, (0, 200, 0), 2)
            cv2.putText(img, "1", (p1[0] + 15, p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "2", (p2[0] + 15, p2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            pixel_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if self.pixels_per_cm is not None:
                cm_distance = pixel_distance / self.pixels_per_cm
                status_text = "测距完成 (按 r 重新测量)"
            else:
                status_text = "未标定，按 c 标定后显示厘米距离"

        # 左上角信息面板
        y_offset = 30
        cv2.putText(img, f"像素距离: {pixel_distance:.2f} px", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if self.pixels_per_cm is not None:
            cv2.putText(img, f"实际距离: {cm_distance:.2f} cm", (20, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        else:
            cv2.putText(img, "实际距离: 需标定 (按 c 键)", (20, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        cv2.putText(img, status_text, (20, y_offset + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        # 标定状态
        if self.pixels_per_cm is not None:
            cv2.putText(img, f"已标定: {self.pixels_per_cm:.2f} px/cm", (20, y_offset + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        # 底部操作提示
        cv2.putText(img, "操作: q-退出  c-标定  r-清除点重新测量", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    def _draw_visualization(self, frame):
        """在画面上绘制指尖标记、连线、距离信息和状态文字"""
        h, w, _ = frame.shape

        if self.calibrating:
            cv2.putText(frame, f"标定模式: 点击 {KNOWN_LENGTH_CM}cm 物体两端点", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for i, point in enumerate(self.calibration_points):
                cv2.circle(frame, point, 8, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (point[0] + 10, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return

        status_text = ""
        pixel_distance = 0
        cm_distance = 0

        has_left = self.left_index_tip is not None
        has_right = self.right_index_tip is not None

        if has_left:
            cv2.circle(frame, self.left_index_tip, 10, (0, 0, 255), -1)
            cv2.putText(frame, "左手食指", (self.left_index_tip[0] + 15, self.left_index_tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if has_right:
            cv2.circle(frame, self.right_index_tip, 10, (255, 0, 0), -1)
            cv2.putText(frame, "右手食指", (self.right_index_tip[0] + 15, self.right_index_tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if has_left and has_right:
            cv2.line(frame, self.left_index_tip, self.right_index_tip, (0, 255, 0), 2)

            pixel_distance = math.hypot(
                self.right_index_tip[0] - self.left_index_tip[0],
                self.right_index_tip[1] - self.left_index_tip[1]
            )

            if self.pixels_per_cm is not None:
                cm_distance = pixel_distance / self.pixels_per_cm
                status_text = "检测状态: 正常"
            else:
                status_text = "检测状态: 未标定"
        else:
            if not has_left and not has_right:
                status_text = "请同时伸出左右食指"
            elif not has_left:
                status_text = "未检测到左手食指"
            else:
                status_text = "未检测到右手食指"

        y_offset = 30
        cv2.putText(frame, f"像素距离: {pixel_distance:.2f} px", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.pixels_per_cm is not None:
            cv2.putText(frame, f"实际距离: {cm_distance:.2f} cm", (20, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "实际距离: 需标定 (按 c 键)", (20, y_offset + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, status_text, (20, y_offset + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"帧率: {self.fps:.1f} FPS", (20, y_offset + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "操作: q-退出  c-标定", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _find_camera(self):
        """自动查找可用摄像头设备，尝试多种后端"""
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for backend in backends:
            for i in range(5):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.release()
                        return i
                    cap.release()
        return -1

    def _run_canvas_mode(self):
        """画板模式：在背景图上用鼠标点击两点测距"""
        self.canvas_mode = True
        print("=" * 60)
        print("已进入画板模式")
        print("操作说明：")
        print("  鼠标左键点击 - 选择测距点（每次2个点）")
        print("  c - 进入标定模式")
        print("  r - 清除已标记点，重新测量")
        print("  q - 退出程序")
        print("=" * 60)

        cv2.namedWindow("画板测距")
        cv2.setMouseCallback("画板测距", self._canvas_mouse_callback)

        try:
            while True:
                # 每帧重新生成背景，避免残留绘制
                img = self._create_canvas_background()
                self._draw_canvas(img)
                cv2.imshow("画板测距", img)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    print("退出程序")
                    break

                if key == ord('c'):
                    if self.calibrating:
                        print("取消标定")
                        self.calibrating = False
                        self.calibration_points = []
                    else:
                        self.start_calibration()

                if key == ord('r'):
                    self.canvas_points = []
                    self.calibration_points = []
                    self.calibrating = False
                    print("已清除标记点")

                if key == 27 and self.calibrating:
                    print("取消标定")
                    self.calibrating = False
                    self.calibration_points = []

        except Exception as e:
            print(f"画板模式运行错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cv2.destroyAllWindows()
            if self.hand_landmarker is not None:
                self.hand_landmarker.close()
            print("资源已释放")

    def _run_camera_mode(self, camera_index):
        """摄像头实时测距模式"""
        print(f"找到摄像头，设备索引: {camera_index}")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print("错误：无法打开摄像头！")
            return

        print("摄像头已打开，按 q 键退出，按 c 键进行标定")

        cv2.namedWindow("指尖测距")
        cv2.setMouseCallback("指尖测距", self._calibration_callback)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("警告：无法读取摄像头帧")
                    continue

                frame = cv2.flip(frame, 1)

                self._process_frame(frame)
                self._draw_visualization(frame)

                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    end_time = cv2.getTickCount()
                    elapsed_time = (end_time - self.start_time) / cv2.getTickFrequency()
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = cv2.getTickCount()

                cv2.imshow("指尖测距", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("退出程序")
                    break

                if key == ord('c'):
                    if self.calibrating:
                        print("取消标定")
                        self.calibrating = False
                        self.calibration_points = []
                    else:
                        self.start_calibration()

                if key == 27 and self.calibrating:
                    print("取消标定")
                    self.calibrating = False
                    self.calibration_points = []

        except Exception as e:
            print(f"运行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.hand_landmarker is not None:
                self.hand_landmarker.close()
            print("资源已释放")

    def run(self):
        """程序入口：优先尝试摄像头模式，不可用时询问是否进入画板模式"""
        camera_index = self._find_camera()

        if camera_index == -1:
            print("未检测到可用摄像头！")
            print("=" * 60)
            print("可能的原因：")
            print("  1. Windows隐私设置限制（设置 -> 隐私和安全性 -> 相机）")
            print("  2. 摄像头被其他应用占用（Teams、Zoom、微信等）")
            print("  3. 摄像头硬件问题或驱动异常")
            print("=" * 60)
            print("\n是否进入画板模式？")
            print("  画板模式：在背景图上用鼠标点击两点进行测距")
            choice = input("请输入 y 进入画板模式，其他键退出: ").strip().lower()
            if choice == 'y':
                self._run_canvas_mode()
            else:
                print("已退出")
            return

        self._run_camera_mode(camera_index)


if __name__ == "__main__":
    measurer = FingertipDistanceMeasurer()
    measurer.run()
