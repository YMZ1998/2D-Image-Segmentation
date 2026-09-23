import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PyQt5.QtCore import QSettings, QTimer, Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSlider,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from inference_utils import (
    clean_circular_roi,
    create_pseudocolor,
    newest_onnx,
    onnx_output_to_mask,
    prepare_onnx_input,
)
from segmentation_config import CLASS_COLORS, CLASS_NAMES, IMAGE_SIZE


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
        self.class_visibility_boxes: dict[int, QCheckBox] = {}
        self.progress_dragging = False
        self.onnx_session = None
        self.onnx_model_path: Path | None = None
        self.trt_engine = None
        self.trt_context = None
        self.trt_cuda = None
        self.trt_input_name: str | None = None
        self.trt_output_name: str | None = None
        self.trt_engine_path: Path | None = None
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
        self.backend_box = QComboBox()
        self.backend_box.addItems(("ONNX", "TensorRT"))
        self.backend_box.setCurrentText(self.settings.value("inference_backend", "ONNX"))
        self.backend_box.currentTextChanged.connect(self.change_inference_backend)
        self.onnx_button = QPushButton("AI 分割")
        self.onnx_button.setCheckable(True)
        self.onnx_button.toggled.connect(self.toggle_onnx)
        previous_button = QPushButton("上一帧")
        previous_button.clicked.connect(lambda: self.step_frame(-1))
        next_button = QPushButton("下一帧")
        next_button.clicked.connect(lambda: self.step_frame(1))
        self.export_button = QPushButton("导出分割视频")
        self.export_button.clicked.connect(self.export_segmented_video)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        self.position_slider.setMinimumHeight(24)
        self.position_slider.setToolTip("拖动跳转视频帧")
        self.position_slider.sliderPressed.connect(self.begin_progress_drag)
        self.position_slider.sliderReleased.connect(self.finish_progress_drag)
        self.position_slider.valueChanged.connect(self.update_progress_preview)

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
        self.show_labels_button = QPushButton("标签显示")
        self.show_labels_button.setCheckable(True)
        self.show_labels_button.setChecked(True)
        self.show_labels_button.toggled.connect(self.toggle_label_controls)
        self.label_controls = QWidget()
        label_controls_layout = QHBoxLayout(self.label_controls)
        label_controls_layout.setContentsMargins(0, 0, 0, 0)
        label_controls_layout.setSpacing(8)
        for class_id, class_name in enumerate(CLASS_NAMES):
            if class_id == 0:
                continue
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(
                self.settings.value(f"mp4_class_visible_{class_id}", True, type=bool)
            )
            checkbox.toggled.connect(self.refresh_current_frame)
            label_controls_layout.addWidget(checkbox)
            self.class_visibility_boxes[class_id] = checkbox
        self.toggle_label_controls(self.show_labels_button.isChecked())

        top_row = QHBoxLayout()
        top_row.addWidget(open_button)
        top_row.addWidget(self.path_label, 1)

        controls = QHBoxLayout()
        controls.addWidget(previous_button)
        controls.addWidget(self.play_button)
        controls.addWidget(next_button)
        controls.addWidget(self.pseudocolor_button)
        controls.addWidget(QLabel("后端："))
        controls.addWidget(self.backend_box)
        controls.addWidget(self.onnx_button)
        controls.addWidget(QLabel("分割透明度："))
        controls.addWidget(self.alpha_slider)
        controls.addWidget(self.alpha_label)
        controls.addWidget(self.show_labels_button)
        controls.addWidget(self.label_controls)
        controls.addWidget(self.export_button)
        controls.addWidget(self.resolution_label)

        progress_row = QHBoxLayout()
        progress_row.addWidget(QLabel("播放进度："))
        progress_row.addWidget(self.position_slider, 1)
        progress_row.addWidget(self.time_label)
        progress_row.addWidget(self.frame_label)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.video_label, 1)
        layout.addLayout(progress_row)
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
                QMessageBox.critical(self, "AI 分割失败", str(error))
                return
        self.show_current_frame()
        self.update_position_labels()

    def seek_frame(self, frame_index: int) -> None:
        if self.capture is None or self.frame_count == 0:
            return
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.read_next_frame()

    def begin_progress_drag(self) -> None:
        self.progress_dragging = True

    def finish_progress_drag(self) -> None:
        if not self.progress_dragging:
            return
        self.progress_dragging = False
        self.seek_frame(self.position_slider.value())

    def update_progress_preview(self, frame_index: int) -> None:
        if not self.progress_dragging:
            return
        frame_index = max(0, min(int(frame_index), max(0, self.frame_count - 1)))
        current_seconds = frame_index / self.fps if self.fps > 0 else 0.0
        self.time_label.setText(
            f"{self.format_time(current_seconds)} / {self.format_time(self.duration_seconds)}"
        )
        self.frame_label.setText(f"帧 {frame_index + 1} / {self.frame_count}")

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
    def processed_gray(frame_rgb: np.ndarray) -> np.ndarray:
        """Match training preprocessing by using grayscale pixels inside the fixed OCT ROI."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return clean_circular_roi(gray)

    def compose_display_frame(self) -> np.ndarray:
        return self.compose_frame(self.current_rgb, self.prediction_mask)

    def compose_frame(
        self,
        frame_rgb: np.ndarray,
        prediction_mask: np.ndarray | None,
    ) -> np.ndarray:
        gray = self.processed_gray(frame_rgb)
        display = (
            create_pseudocolor(gray)
            if self.pseudocolor_button.isChecked()
            else np.repeat(gray[..., None], 3, axis=2)
        )
        if prediction_mask is None:
            return np.ascontiguousarray(display)

        colors = np.asarray(CLASS_COLORS, dtype=np.uint8)
        foreground = np.zeros_like(prediction_mask, dtype=bool)
        for class_id, checkbox in self.class_visibility_boxes.items():
            if checkbox.isChecked():
                foreground |= prediction_mask == class_id
        alpha = self.alpha_slider.value() / 100
        display[foreground] = (
            (1 - alpha) * display[foreground]
            + alpha * colors[prediction_mask[foreground]]
        ).astype(np.uint8)
        return np.ascontiguousarray(display)

    def export_segmented_video(self) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "提示", "请先打开视频")
            return

        range_dialog = QDialog(self)
        range_dialog.setWindowTitle("选择预测时间段")
        range_form = QFormLayout(range_dialog)
        start_spin = QDoubleSpinBox()
        end_spin = QDoubleSpinBox()
        for spin in (start_spin, end_spin):
            spin.setRange(0.0, self.duration_seconds)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(" 秒")
        start_spin.setValue(0.0)
        end_spin.setValue(self.duration_seconds)
        range_form.addRow("开始时间", start_spin)
        range_form.addRow("结束时间", end_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(range_dialog.accept)
        buttons.rejected.connect(range_dialog.reject)
        range_form.addRow(buttons)
        if range_dialog.exec_() != QDialog.Accepted:
            return

        start_frame = max(0, int(start_spin.value() * self.fps))
        end_frame = min(
            self.frame_count,
            math.ceil(end_spin.value() * self.fps),
        )
        if end_frame <= start_frame:
            QMessageBox.warning(self, "时间段无效", "结束时间必须晚于开始时间")
            return

        default_path = self.video_path.with_name(
            f"{self.video_path.stem}_{start_spin.value():.3f}-{end_spin.value():.3f}s_segmented.mp4"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "导出分割视频",
            str(default_path),
            "MP4 视频 (*.mp4)",
        )
        if not selected:
            return
        output_path = Path(selected)
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        if output_path.resolve() == self.video_path.resolve():
            QMessageBox.warning(self, "导出失败", "输出路径不能覆盖原视频")
            return

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            QMessageBox.critical(self, "导出失败", "无法重新读取原视频")
            return

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        export_frame_count = end_frame - start_frame
        export_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if export_fps <= 0:
            export_fps = self.fps

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            export_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            QMessageBox.critical(self, "导出失败", f"无法创建视频文件：\n{output_path}")
            return

        progress = QProgressDialog(
            "正在逐帧预测并导出…",
            "取消",
            0,
            export_frame_count,
            self,
        )
        progress.setWindowTitle("导出分割视频")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        self.export_button.setEnabled(False)
        was_playing = self.timer.isActive()
        self.pause()
        completed = 0
        try:
            while completed < export_frame_count:
                if progress.wasCanceled():
                    break
                success, frame_bgr = capture.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gray = self.processed_gray(frame_rgb)
                if self.use_tensorrt():
                    self.ensure_tensorrt_session()
                    from scripts.inference.predict_single_tensorrt import (
                        execute,
                        output_to_mask,
                        prepare_input,
                        tensor_shape,
                    )

                    input_shape = tuple(
                        dim if isinstance(dim, int) and dim > 0 else IMAGE_SIZE
                        for dim in tensor_shape(
                            self.trt_engine, self.trt_context, self.trt_input_name
                        )
                    )
                    tensor, _, _ = prepare_input(gray, input_shape, IMAGE_SIZE)
                    output, _ = execute(
                        self.trt_engine,
                        self.trt_context,
                        self.trt_cuda,
                        self.trt_input_name,
                        self.trt_output_name,
                        tensor,
                    )
                    prediction = output_to_mask(output)
                else:
                    import onnxruntime as ort

                    model_path = newest_onnx(PROJECT_ROOT / "save_weights")
                    if self.onnx_session is None or model_path != self.onnx_model_path:
                        providers = ["CPUExecutionProvider"]
                        if "CUDAExecutionProvider" in ort.get_available_providers():
                            providers.insert(0, "CUDAExecutionProvider")
                        self.onnx_session = ort.InferenceSession(
                            str(model_path), providers=providers
                        )
                        self.onnx_model_path = model_path
                    input_meta = self.onnx_session.get_inputs()[0]
                    tensor, _, _ = prepare_onnx_input(
                        gray, input_meta.shape, IMAGE_SIZE
                    )
                    output = self.onnx_session.run(
                        None, {input_meta.name: tensor}
                    )[0]
                    prediction = onnx_output_to_mask(output)

                prediction = cv2.resize(
                    prediction, (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)
                display_rgb = self.compose_frame(frame_rgb, prediction)
                writer.write(cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
                completed += 1
                progress.setValue(completed)
                QApplication.processEvents()

            canceled = progress.wasCanceled()
            progress.close()
            if canceled:
                writer.release()
                capture.release()
                output_path.unlink(missing_ok=True)
                QMessageBox.information(
                    self, "导出已取消", f"已处理 {completed} 帧，未保留不完整文件。"
                )
            else:
                QMessageBox.information(
                    self, "导出完成", f"已导出 {completed} 帧：\n{output_path}"
                )
        except Exception as error:
            writer.release()
            capture.release()
            output_path.unlink(missing_ok=True)
            QMessageBox.critical(self, "导出失败", str(error))
        finally:
            writer.release()
            capture.release()
            progress.close()
            self.export_button.setEnabled(True)
            if was_playing:
                self.timer.start()
                self.play_button.setText("暂停")

    def toggle_label_controls(self, visible: bool) -> None:
        self.label_controls.setVisible(visible)
        self.show_labels_button.setText("隐藏标签" if visible else "标签显示")
        self.refresh_current_frame()

    def toggle_onnx(self, enabled: bool) -> None:
        if not enabled:
            self.prediction_mask = None
            self.onnx_button.setText("AI 分割")
            self.show_current_frame()
            return
        self.onnx_button.setText("关闭 AI 分割")
        if self.current_rgb is not None:
            try:
                self.predict_current_frame()
                self.show_current_frame()
            except Exception as error:
                self.onnx_button.blockSignals(True)
                self.onnx_button.setChecked(False)
                self.onnx_button.blockSignals(False)
                self.onnx_button.setText("AI 分割")
                QMessageBox.critical(self, "AI 分割失败", str(error))

    def change_inference_backend(self, backend: str) -> None:
        self.settings.setValue("inference_backend", backend)
        self.prediction_mask = None
        self.onnx_session = None
        self.onnx_model_path = None
        self.trt_engine = None
        self.trt_context = None
        self.trt_cuda = None
        self.trt_input_name = None
        self.trt_output_name = None
        self.trt_engine_path = None
        if self.onnx_button.isChecked() and self.current_rgb is not None:
            try:
                self.predict_current_frame()
            except Exception as error:
                self.onnx_button.setChecked(False)
                QMessageBox.critical(self, "AI 分割失败", str(error))
        self.show_current_frame()

    def use_tensorrt(self) -> bool:
        return self.backend_box.currentText() == "TensorRT"

    def predict_current_frame(self) -> None:
        if self.use_tensorrt():
            self.predict_current_frame_tensorrt()
        else:
            self.predict_current_frame_onnx()

    def predict_current_frame_onnx(self) -> None:
        import onnxruntime as ort

        model_path = newest_onnx(PROJECT_ROOT / "save_weights")
        if self.onnx_session is None or model_path != self.onnx_model_path:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
            self.onnx_model_path = model_path

        gray = self.processed_gray(self.current_rgb)
        input_meta = self.onnx_session.get_inputs()[0]
        tensor, _, _ = prepare_onnx_input(gray, input_meta.shape, IMAGE_SIZE)
        output = self.onnx_session.run(None, {input_meta.name: tensor})[0]
        prediction = onnx_output_to_mask(output)
        self.prediction_mask = cv2.resize(
            prediction, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        self.statusBar().showMessage(
            f"ONNX：{model_path.name} · {self.onnx_session.get_providers()[0]}", 3000
        )

    def ensure_tensorrt_session(self) -> Path:
        from scripts.inference.predict_single_tensorrt import (
            binding_names,
            build_engine_from_onnx,
            import_trt_runtime,
            is_input,
            load_engine,
            matching_onnx_for_engine,
            newest_engine,
        )

        engine_path = newest_engine(PROJECT_ROOT / "save_weights")
        if self.trt_engine is None or engine_path != self.trt_engine_path:
            trt, cuda = import_trt_runtime()
            try:
                engine = load_engine(engine_path, trt)
            except RuntimeError:
                engine = build_engine_from_onnx(
                    onnx_path=matching_onnx_for_engine(engine_path),
                    engine_path=engine_path,
                    trt=trt,
                    fp16=False,
                    workspace_gb=2.0,
                )

            context = engine.create_execution_context()
            names = binding_names(engine)
            input_names = [name for name in names if is_input(engine, name)]
            output_names = [name for name in names if not is_input(engine, name)]
            if len(input_names) != 1 or len(output_names) < 1:
                raise RuntimeError(
                    f"Expected one TensorRT input and at least one output, got {input_names}, {output_names}"
                )

            self.trt_engine = engine
            self.trt_context = context
            self.trt_cuda = cuda
            self.trt_input_name = input_names[0]
            self.trt_output_name = output_names[0]
            self.trt_engine_path = engine_path
        return engine_path

    def predict_current_frame_tensorrt(self) -> None:
        from scripts.inference.predict_single_tensorrt import (
            execute,
            output_to_mask,
            prepare_input,
            tensor_shape,
        )

        engine_path = self.ensure_tensorrt_session()
        gray = self.processed_gray(self.current_rgb)
        input_shape = tuple(
            dim if isinstance(dim, int) and dim > 0 else IMAGE_SIZE
            for dim in tensor_shape(self.trt_engine, self.trt_context, self.trt_input_name)
        )
        tensor, _, _ = prepare_input(gray, input_shape, IMAGE_SIZE)
        output, elapsed_ms = execute(
            self.trt_engine,
            self.trt_context,
            self.trt_cuda,
            self.trt_input_name,
            self.trt_output_name,
            tensor,
        )
        prediction = output_to_mask(output)
        self.prediction_mask = cv2.resize(
            prediction, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        self.statusBar().showMessage(
            f"TensorRT：{engine_path.name} · {elapsed_ms:.1f} ms", 3000
        )

    def update_position_labels(self) -> None:
        if not self.progress_dragging:
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
            self.settings.setValue("inference_backend", self.backend_box.currentText())
        for class_id, checkbox in self.class_visibility_boxes.items():
            self.settings.setValue(f"mp4_class_visible_{class_id}", checkbox.isChecked())
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
