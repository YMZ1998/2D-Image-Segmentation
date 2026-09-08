import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QEvent, QSettings, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, \
    QPushButton, QSlider, QVBoxLayout, QWidget

from inference_utils import create_pseudocolor as create_pseudocolor_array
from inference_utils import newest_onnx, onnx_output_to_mask, prepare_onnx_input
from segmentation_config import CLASS_COLORS, CLASS_DISPLAY_VALUES, CLASS_NAMES, IMAGE_SIZE

DEFAULT_IMAGE_DIR = Path(r"D:\data\OCT")

# OCT 圆环参数
OCT_OUTER_RADIUS_RATIO = 0.49


class OverlayViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCT ONNX Inference Viewer")
        self.resize(1500, 900)

        self.settings = QSettings("2D-Image-Segmentation", "OverlayViewer")
        saved_dir = Path(self.settings.value("image_dir", str(DEFAULT_IMAGE_DIR)))
        self.image_dir = saved_dir if saved_dir.is_dir() else (
            DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.is_dir() else Path.cwd())
        self.output_dir = self.image_dir / "overlays"

        self.source_image: Image.Image | None = None
        self.pseudocolor_image: Image.Image | None = None
        self.prediction_mask: Image.Image | None = None
        self.prediction_rendered: Image.Image | None = None

        self.onnx_session = None
        self.onnx_model_path: Path | None = None
        self._source_bytes: bytes | None = None
        self._prediction_bytes: bytes | None = None

        self.directory_label = QLabel(str(self.image_dir.resolve()))
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setStyleSheet("padding:5px;background:#f2f2f2;border:1px solid #d0d0d0;")

        choose_button = QPushButton("选择图片目录")
        choose_button.clicked.connect(self.choose_image_dir)

        directory_layout = QHBoxLayout()
        directory_layout.addWidget(QLabel("图片目录："))
        directory_layout.addWidget(self.directory_label, 1)
        directory_layout.addWidget(choose_button)

        self.sample_box = QComboBox()
        self.sample_box.currentIndexChanged.connect(self.load_selected_sample)

        reload_button = QPushButton("刷新图片")
        reload_button.clicked.connect(self.refresh_samples)

        self.pseudocolor_button = QPushButton("转换伪彩色")
        self.pseudocolor_button.setCheckable(True)
        self.pseudocolor_button.toggled.connect(self.toggle_pseudocolor)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.valueChanged.connect(self.render_prediction)

        self.alpha_label = QLabel("55%")
        self.alpha_label.setMinimumWidth(48)

        self.predict_button = QPushButton("ONNX 预测")
        self.predict_button.clicked.connect(self.run_onnx_prediction)

        save_button = QPushButton("保存预测叠加图")
        save_button.clicked.connect(self.save_overlay)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("图片："))
        controls.addWidget(self.sample_box, 1)
        controls.addWidget(reload_button)
        controls.addWidget(self.pseudocolor_button)
        controls.addSpacing(16)
        controls.addWidget(QLabel("透明度："))
        controls.addWidget(self.alpha_slider, 1)
        controls.addWidget(self.alpha_label)
        controls.addWidget(self.predict_button)
        controls.addWidget(save_button)

        self.legend = QLabel(self.build_legend_text())
        self.legend.setStyleSheet("color:#555;padding:4px;")

        self.image_label = QLabel("目录中没有 PNG 图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background:#181818;color:#ddd;")
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        self.prediction_label = QLabel("点击“ONNX 预测”显示预测结果")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumSize(400, 400)
        self.prediction_label.setStyleSheet("background:#181818;color:#ddd;")

        source_title = QLabel("原始图像")
        source_title.setAlignment(Qt.AlignCenter)
        source_title.setStyleSheet("font-weight:bold;padding:4px;")

        self.prediction_title = QLabel("ONNX 预测")
        self.prediction_title.setAlignment(Qt.AlignCenter)
        self.prediction_title.setStyleSheet("font-weight:bold;padding:4px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(source_title)
        left_layout.addWidget(self.image_label, 1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.prediction_title)
        right_layout.addWidget(self.prediction_label, 1)

        image_layout = QHBoxLayout()
        image_layout.addLayout(left_layout, 1)
        image_layout.addLayout(right_layout, 1)

        self.pixel_label = QLabel("将鼠标悬浮在图片上查看像素值")
        self.pixel_label.setStyleSheet(
            "font-family:Consolas,'Microsoft YaHei';padding:6px 10px;background:#f2f2f2;border:1px solid #d0d0d0;")

        layout = QVBoxLayout()
        layout.addLayout(directory_layout)
        layout.addLayout(controls)
        layout.addWidget(self.legend)
        layout.addLayout(image_layout, 1)
        layout.addWidget(self.pixel_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.refresh_samples()

    @staticmethod
    def build_legend_text() -> str:
        items = []
        for class_id in range(1, len(CLASS_NAMES)):
            items.append(f"{CLASS_NAMES[class_id]} RGB={CLASS_COLORS[class_id]}")
        return "    ".join(items) if items else "无分割类别"

    def choose_image_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择包含 PNG 图片的目录", str(self.image_dir.resolve()))
        if not selected:
            return
        self.image_dir = Path(selected).resolve()
        self.output_dir = self.image_dir / "overlays"
        self.settings.setValue("image_dir", str(self.image_dir))
        self.directory_label.setText(str(self.image_dir))
        self.refresh_samples()
        self.statusBar().showMessage(f"已加载目录：{self.image_dir}", 4000)

    def refresh_samples(self) -> None:
        selected = self.sample_box.currentText()
        samples = [p.name for p in sorted(self.image_dir.glob("*.png")) if
                   p.is_file()] if self.image_dir.is_dir() else []

        self.sample_box.blockSignals(True)
        self.sample_box.clear()
        self.sample_box.addItems(samples)

        if selected in samples:
            self.sample_box.setCurrentText(selected)
        elif samples:
            self.sample_box.setCurrentIndex(0)

        self.sample_box.blockSignals(False)
        self.statusBar().showMessage(f"找到 {len(samples)} 张 PNG 图片", 3000)
        self.load_selected_sample()

    def load_selected_sample(self) -> None:
        name = self.sample_box.currentText()
        if not name:
            self.source_image = None
            self.pseudocolor_image = None
            self.clear_prediction()
            self.image_label.clear()
            self.image_label.setText("目录中没有 PNG 图片")
            return

        try:
            self.source_image = Image.open(self.image_dir / name).convert("RGBA")
            self.pseudocolor_image = self.create_pseudocolor(self.source_image)
            self.clear_prediction()
            self.update_source_image()
            self.statusBar().showMessage(f"已加载：{name}", 2000)
        except Exception as error:
            QMessageBox.critical(self, "加载失败", str(error))

    @staticmethod
    def create_pseudocolor(image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"))
        return Image.fromarray(create_pseudocolor_array(gray)).convert("RGBA")

    @staticmethod
    def mask_oct_region(gray: np.ndarray) -> np.ndarray:
        """只保留 OCT 圆环区域，外圆之外和中心无效圆全部置 0。"""
        h, w = gray.shape
        cx = w / 2.0
        cy = h / 2.0

        base_size = min(w, h)
        outer_radius = base_size * OCT_OUTER_RADIUS_RATIO

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (dist2 <= outer_radius ** 2)

        result = np.zeros_like(gray)
        result[mask] = gray[mask]
        return result

    def toggle_pseudocolor(self, enabled: bool) -> None:
        self.pseudocolor_button.setText("恢复原始图" if enabled else "转换伪彩色")
        self.update_source_image()
        if self.prediction_mask is not None:
            self.render_prediction()

    def get_base_image(self) -> Image.Image:
        if self.pseudocolor_button.isChecked() and self.pseudocolor_image is not None:
            return self.pseudocolor_image
        return self.source_image

    def update_source_image(self) -> None:
        if self.source_image is None:
            return
        self.set_label_image(self.image_label, self.get_base_image().convert("RGB"), True)

    def compose_overlay(self, mask: Image.Image) -> Image.Image:
        if self.source_image is None:
            raise RuntimeError("没有加载图片")

        result = self.get_base_image().convert("RGBA").copy()
        alpha = round(255 * self.alpha_slider.value() / 100)

        for class_id in range(1, len(CLASS_NAMES)):
            class_values = CLASS_DISPLAY_VALUES[class_id]
            rgb = CLASS_COLORS[class_id]
            lookup = [alpha if value in class_values else 0 for value in range(256)]
            class_alpha = mask.point(lookup)

            color_layer = Image.new("RGBA", result.size, (*rgb, 0))
            color_layer.putalpha(class_alpha)
            result = Image.alpha_composite(result, color_layer)

        return result.convert("RGB")

    def render_prediction(self) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value()}%")
        self.update_source_image()

        if self.prediction_mask is None:
            return

        self.prediction_rendered = self.compose_overlay(self.prediction_mask)
        self.set_label_image(self.prediction_label, self.prediction_rendered, False)

    def set_label_image(self, label: QLabel, image: Image.Image, source: bool = False) -> None:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")

        if source:
            self._source_bytes = data
        else:
            self._prediction_bytes = data

        qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def clear_prediction(self) -> None:
        self.prediction_mask = None
        self.prediction_rendered = None

        if hasattr(self, "prediction_label"):
            self.prediction_label.clear()
            self.prediction_label.setText("点击“ONNX 预测”显示预测结果")

        if hasattr(self, "prediction_title"):
            self.prediction_title.setText("ONNX 预测")

    def run_onnx_prediction(self) -> None:
        if self.source_image is None:
            return

        self.predict_button.setEnabled(False)
        self.predict_button.setText("预测中…")
        QApplication.processEvents()

        try:
            import onnxruntime as ort

            model_path = newest_onnx()

            if self.onnx_session is None or model_path != self.onnx_model_path:
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")

                self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
                self.onnx_model_path = model_path

            input_meta = self.onnx_session.get_inputs()[0]

            # 原始灰度图
            gray = np.asarray(self.source_image.convert("L"))

            # =================================================
            # 关键：预测前过滤圆外和中心无效区域
            # =================================================
            gray = self.mask_oct_region(gray)

            tensor, _, _ = prepare_onnx_input(gray, input_meta.shape, IMAGE_SIZE)

            start = time.perf_counter()
            output = self.onnx_session.run(None, {input_meta.name: tensor})[0]
            elapsed_ms = (time.perf_counter() - start) * 1000

            prediction = onnx_output_to_mask(output)
            prediction = np.asarray(
                Image.fromarray(prediction).resize(self.source_image.size, Image.Resampling.NEAREST))

            # =================================================
            # 再把模型输出的圆外区域强制清零
            # 防止 resize 或模型产生圆外预测
            # =================================================
            prediction = self.mask_prediction_region(prediction)

            self.prediction_mask = Image.fromarray(prediction.astype(np.uint8), mode="L")
            self.render_prediction()

            provider = self.onnx_session.get_providers()[0]
            self.prediction_title.setText(f"ONNX 预测 · {provider} · {elapsed_ms:.1f} ms")
            self.statusBar().showMessage(f"预测完成：{model_path.name}", 5000)

        except Exception as error:
            QMessageBox.critical(self, "ONNX 预测失败", str(error))

        finally:
            self.predict_button.setEnabled(True)
            self.predict_button.setText("ONNX 预测")

    @staticmethod
    def mask_prediction_region(mask: np.ndarray) -> np.ndarray:
        """预测结果同样只保留 OCT 圆环区域。"""
        h, w = mask.shape
        cx = w / 2.0
        cy = h / 2.0

        base_size = min(w, h)
        outer_radius = base_size * OCT_OUTER_RADIUS_RATIO

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        valid = (dist2 <= outer_radius ** 2)

        result = np.zeros_like(mask)
        result[valid] = mask[valid]
        return result

    def eventFilter(self, watched, event):
        if watched is self.image_label:
            if event.type() == QEvent.MouseMove:
                self.show_pixel_value(event.pos().x(), event.pos().y())
            elif event.type() == QEvent.Wheel:
                self.change_sample(-1 if event.angleDelta().y() > 0 else 1)
                event.accept()
                return True
            elif event.type() == QEvent.Leave:
                self.pixel_label.setText("将鼠标悬浮在图片上查看像素值")

        return super().eventFilter(watched, event)

    def change_sample(self, step: int) -> None:
        count = self.sample_box.count()
        if count < 2:
            return

        index = (self.sample_box.currentIndex() + step) % count
        self.sample_box.setCurrentIndex(index)
        self.statusBar().showMessage(f"图片 {index + 1}/{count}：{self.sample_box.currentText()}", 1500)

    def show_pixel_value(self, mouse_x: int, mouse_y: int) -> None:
        pixmap = self.image_label.pixmap()
        if pixmap is None or self.source_image is None:
            return

        offset_x = (self.image_label.width() - pixmap.width()) / 2
        offset_y = (self.image_label.height() - pixmap.height()) / 2
        local_x = mouse_x - offset_x
        local_y = mouse_y - offset_y

        if not (0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height()):
            self.pixel_label.setText("鼠标位于图片显示区域之外")
            return

        image_x = min(self.source_image.width - 1, int(local_x * self.source_image.width / pixmap.width()))
        image_y = min(self.source_image.height - 1, int(local_y * self.source_image.height / pixmap.height()))

        rgb = self.source_image.getpixel((image_x, image_y))[:3]
        gray = self.source_image.convert("L").getpixel((image_x, image_y))

        text = f"坐标: ({image_x}, {image_y})    灰度: {gray}    RGB: {rgb}"

        if self.prediction_mask is not None:
            value = self.prediction_mask.getpixel((image_x, image_y))
            class_name = "background"

            for class_id in range(1, len(CLASS_NAMES)):
                if value in CLASS_DISPLAY_VALUES[class_id]:
                    class_name = CLASS_NAMES[class_id]
                    break

            text += f"    Predict: {value} ({class_name})"

        self.pixel_label.setText(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if self.source_image is not None:
            self.update_source_image()

        if self.prediction_rendered is not None:
            self.set_label_image(self.prediction_label, self.prediction_rendered, False)

    def save_overlay(self) -> None:
        if self.prediction_rendered is None:
            QMessageBox.information(self, "提示", "请先执行 ONNX 预测")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(self.sample_box.currentText()).stem
        default_path = self.output_dir / f"{stem}_prediction.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存预测叠加图",
            str(default_path.resolve()),
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)"
        )

        if path:
            self.prediction_rendered.save(path)
            self.statusBar().showMessage(f"已保存：{path}", 5000)


def main() -> None:
    app = QApplication(sys.argv)
    viewer = OverlayViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
