import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtCore import QSettings, QTimer, Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class Mp4Viewer(QMainWindow):
    def __init__(self, initial_video: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("MP4 Video Viewer")
        self.resize(1200, 850)

        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.current_rgb = None
        self.current_frame = -1
        self.frame_count = 0
        self.fps = 25.0
        self.duration_seconds = 0.0
        self.prediction_mask = None
        self.onnx_session = None
        self.onnx_model_path: Path | None = None
        self.settings = QSettings("2D-Image-Segmentation", "Mp4Viewer")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_next_frame)

        self.video_label = QLabel("点击“打开 MP4”选择视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #111; color: #ddd; font-size: 16px;")

        self.path_label = QLabel("未打开视频")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setStyleSheet("padding: 6px; background: #f2f2f2; border: 1px solid #ccc;")

        open_button = QPushButton("打开 MP4")
        open_button.clicked.connect(self.choose_video)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        self.pseudocolor_button = QPushButton("伪彩色")
        self.pseudocolor_button.setCheckable(True)
        self.pseudocolor_button.toggled.connect(self.refresh_current_frame)
        self.onnx_button = QPushButton("ONNX 分割")
        self.onnx_button.setCheckable(True)
        self.onnx_button.toggled.connect(self.toggle_onnx)
        previous_button = QPushButton("上一帧")
        previous_button.clicked.connect(lambda: self.step_frame(-1))
        next_button = QPushButton("下一帧")
        next_button.clicked.connect(lambda: self.step_frame(1))

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderMoved.connect(self.seek_frame)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setMinimumWidth(180)
        self.frame_label = QLabel("帧 0 / 0")
        self.frame_label.setMinimumWidth(120)
        self.resolution_label = QLabel("分辨率 --×--")
        self.resolution_label.setMinimumWidth(130)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.setMaximumWidth(160)
        self.alpha_slider.valueChanged.connect(self.refresh_current_frame)
        self.alpha_label = QLabel("55%")
        self.alpha_slider.valueChanged.connect(lambda value: self.alpha_label.setText(f"{value}%"))

        top_row = QHBoxLayout()
        top_row.addWidget(open_button)
        top_row.addWidget(self.path_label, 1)

        controls = QHBoxLayout()
        controls.addWidget(previous_button)
        controls.addWidget(self.play_button)
        controls.addWidget(next_button)
        controls.addWidget(self.pseudocolor_button)
        controls.addWidget(self.onnx_button)
        controls.addWidget(QLabel("分割透明度："))
        controls.addWidget(self.alpha_slider)
        controls.addWidget(self.alpha_label)
        controls.addWidget(self.position_slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(self.frame_label)
        controls.addWidget(self.resolution_label)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.video_label, 1)
        layout.addLayout(controls)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.step_frame(1))
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_video)

        remembered = Path(self.settings.value("last_video", ""))
        startup_video = initial_video or (remembered if remembered.is_file() else None)
        if startup_video is not None:
            self.open_video(startup_video)

    def choose_video(self) -> None:
        remembered_dir = Path(self.settings.value("last_directory", str(Path.cwd())))
        start_dir = str(self.video_path.parent if self.video_path else remembered_dir)
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择 MP4 视频", start_dir, "MP4 视频 (*.mp4);;所有视频 (*.avi *.mov *.mkv *.mp4)"
        )
        if selected:
            self.open_video(Path(selected))

    def open_video(self, path: Path) -> None:
        self.release_video()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            QMessageBox.critical(self, "打开失败", f"无法读取视频：\n{path}")
            return

        self.capture = capture
        self.video_path = path.resolve()
        self.settings.setValue("last_video", str(self.video_path))
        self.settings.setValue("last_directory", str(self.video_path.parent))
        self.settings.sync()
        self.frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.fps = detected_fps if detected_fps > 0 else 25.0
        self.duration_seconds = self.frame_count / self.fps if self.frame_count else 0.0
        self.current_frame = -1

        self.path_label.setText(str(self.video_path))
        self.resolution_label.setText(f"分辨率 {frame_width}×{frame_height}")
        self.position_slider.setRange(0, max(0, self.frame_count - 1))
        self.position_slider.setEnabled(self.frame_count > 0)
        self.play_button.setEnabled(True)
        self.timer.setInterval(max(1, round(1000 / self.fps)))
        self.read_next_frame()
        self.statusBar().showMessage(
            f"已打开：{self.video_path.name} · {frame_width}×{frame_height} · "
            f"{self.frame_count} 帧 · {self.fps:.2f} FPS", 5000
        )

    def toggle_playback(self) -> None:
        if self.capture is None:
            return
        if self.timer.isActive():
            self.pause()
        else:
            if self.current_frame >= self.frame_count - 1:
                self.seek_frame(0)
            self.timer.start()
            self.play_button.setText("暂停")

    def pause(self) -> None:
        self.timer.stop()
        self.play_button.setText("播放")

    def read_next_frame(self) -> None:
        if self.capture is None:
            return
        success, frame = self.capture.read()
        if not success:
            self.pause()
            return
        self.current_frame = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.current_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.prediction_mask = None
        if self.onnx_button.isChecked():
            try:
                self.predict_current_frame()
            except Exception as error:
                self.pause()
                self.onnx_button.setChecked(False)
                QMessageBox.critical(self, "ONNX 分割失败", str(error))
                return
        self.show_current_frame()
        self.update_position_labels()

    def seek_frame(self, frame_index: int) -> None:
        if self.capture is None or self.frame_count == 0:
            return
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.read_next_frame()

    def step_frame(self, offset: int) -> None:
        if self.capture is None:
            return
        self.pause()
        self.seek_frame(self.current_frame + offset)

    def show_current_frame(self) -> None:
        if self.current_rgb is None:
            return
        display_rgb = self.compose_display_frame()
        height, width, channels = display_rgb.shape
        qimage = QImage(
            display_rgb.data,
            width,
            height,
            width * channels,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def refresh_current_frame(self) -> None:
        self.show_current_frame()

    @staticmethod
    def pseudocolor(gray: np.ndarray) -> np.ndarray:
        gray = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=0.5))
        anchors = [
            (0, (0, 0, 0)), (45, (22, 4, 1)), (100, (76, 15, 3)),
            (160, (163, 55, 7)), (215, (245, 139, 24)), (255, (255, 232, 135)),
        ]
        lookup = np.zeros((256, 3), dtype=np.uint8)
        for start, end in zip(anchors[:-1], anchors[1:]):
            values = np.arange(start[0], end[0] + 1)
            ratio = (values - start[0]) / (end[0] - start[0])
            lookup[values] = np.asarray(start[1]) + ratio[:, None] * (
                np.asarray(end[1]) - np.asarray(start[1])
            )
        return lookup[gray]

    @staticmethod
    def processed_gray(frame_rgb: np.ndarray) -> np.ndarray:
        """Match training preprocessing by using grayscale pixels inside the fixed OCT ROI."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        yy, xx = np.ogrid[:height, :width]
        radius = min(height, width) * 0.475
        roi = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2 <= radius**2
        cleaned = gray.copy()
        cleaned[~roi] = 0
        return cleaned

    def compose_display_frame(self) -> np.ndarray:
        gray = self.processed_gray(self.current_rgb)
        display = self.pseudocolor(gray) if self.pseudocolor_button.isChecked() else np.repeat(gray[..., None], 3, axis=2)
        if self.prediction_mask is not None:
            colors = np.asarray([(0, 0, 0), (255, 0, 0), (0, 120, 255), (0, 255, 0)], dtype=np.uint8)
            foreground = self.prediction_mask != 0
            alpha = self.alpha_slider.value() / 100
            display[foreground] = (
                (1 - alpha) * display[foreground] + alpha * colors[self.prediction_mask[foreground]]
            ).astype(np.uint8)
        return np.ascontiguousarray(display)

    def toggle_onnx(self, enabled: bool) -> None:
        if not enabled:
            self.prediction_mask = None
            self.onnx_button.setText("ONNX 分割")
            self.show_current_frame()
            return
        self.onnx_button.setText("关闭 ONNX 分割")
        if self.current_rgb is not None:
            try:
                self.predict_current_frame()
                self.show_current_frame()
            except Exception as error:
                self.onnx_button.blockSignals(True)
                self.onnx_button.setChecked(False)
                self.onnx_button.blockSignals(False)
                self.onnx_button.setText("ONNX 分割")
                QMessageBox.critical(self, "ONNX 分割失败", str(error))

    def predict_current_frame(self) -> None:
        import onnxruntime as ort
        from predict_single_onnx import logits_to_mask, newest_onnx, prepare_input

        model_path = newest_onnx()
        if self.onnx_session is None or model_path != self.onnx_model_path:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
            self.onnx_model_path = model_path

        gray = self.processed_gray(self.current_rgb)
        input_meta = self.onnx_session.get_inputs()[0]
        tensor, _, _ = prepare_input(gray, input_meta.shape, 704)
        output = self.onnx_session.run(None, {input_meta.name: tensor})[0]
        prediction = logits_to_mask(output)
        self.prediction_mask = cv2.resize(
            prediction, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        self.statusBar().showMessage(
            f"ONNX：{model_path.name} · {self.onnx_session.get_providers()[0]}", 3000
        )

    def update_position_labels(self) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(max(0, self.current_frame))
        self.position_slider.blockSignals(False)
        current_seconds = max(0, self.current_frame) / self.fps
        self.time_label.setText(
            f"{self.format_time(current_seconds)} / {self.format_time(self.duration_seconds)}"
        )
        self.frame_label.setText(f"帧 {self.current_frame + 1} / {self.frame_count}")

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining = seconds - minutes * 60
        return f"{minutes:02d}:{remaining:06.3f}"

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self.show_current_frame()

    def release_video(self) -> None:
        self.pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.video_path is not None:
            self.settings.setValue("last_video", str(self.video_path))
            self.settings.setValue("last_directory", str(self.video_path.parent))
        self.settings.sync()
        self.release_video()
        super().closeEvent(event)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qt MP4 video viewer")
    parser.add_argument("video", type=Path, nargs="?", help="optional MP4 path to open on startup")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    app = QApplication(sys.argv[:1])
    viewer = Mp4Viewer(args.video)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
