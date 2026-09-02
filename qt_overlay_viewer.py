import sys
from pathlib import Path

from PIL import Image, ImageOps
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


IMAGE_DIR = Path("data/merged/images")
MASK_DIR = Path("data/merged/masks")
OUTPUT_DIR = Path("data/merged/overlays")

# Compatible with class-index and display-scaled grayscale masks.
CLASSES = {
    "plaque": ({1, 127}, (255, 0, 0)),
    "Stent": ({3, 192}, (0, 120, 255)),
    "Calcification": ({2, 244, 255}, (0, 255, 0)),
}


class OverlayViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mask Overlay Transparency Debugger")
        self.resize(1100, 900)

        self.source_image: Image.Image | None = None
        self.pseudocolor_image: Image.Image | None = None
        self.mask_image: Image.Image | None = None
        self.rendered_image: Image.Image | None = None
        self._image_bytes: bytes | None = None

        self.sample_box = QComboBox()
        self.sample_box.currentIndexChanged.connect(self.load_selected_sample)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.valueChanged.connect(self.render_overlay)

        self.alpha_label = QLabel("55%")
        self.alpha_label.setMinimumWidth(48)

        reload_button = QPushButton("刷新样本")
        reload_button.clicked.connect(self.refresh_samples)
        self.pseudocolor_button = QPushButton("转换伪彩色")
        self.pseudocolor_button.setCheckable(True)
        self.pseudocolor_button.toggled.connect(self.toggle_pseudocolor)
        save_button = QPushButton("保存叠加图")
        save_button.clicked.connect(self.save_overlay)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("样本："))
        controls.addWidget(self.sample_box, 1)
        controls.addWidget(reload_button)
        controls.addWidget(self.pseudocolor_button)
        controls.addSpacing(16)
        controls.addWidget(QLabel("透明度："))
        controls.addWidget(self.alpha_slider, 1)
        controls.addWidget(self.alpha_label)
        controls.addWidget(save_button)

        legend = QLabel(
            "红色 = plaque    蓝色 = Stent    绿色 = Calcification    "
            f"图片目录：{IMAGE_DIR}    Mask 目录：{MASK_DIR}"
        )
        legend.setStyleSheet("color: #555; padding: 4px;")

        self.image_label = QLabel("没有找到匹配的图片和 mask")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background: #181818; color: #ddd;")
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        self.pixel_label = QLabel("将鼠标悬浮在图片上查看像素值")
        self.pixel_label.setStyleSheet(
            "font-family: Consolas, 'Microsoft YaHei'; padding: 6px 10px; "
            "background: #f2f2f2; border: 1px solid #d0d0d0;"
        )

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(legend)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.pixel_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.refresh_samples()

    def refresh_samples(self) -> None:
        selected = self.sample_box.currentText()
        samples = [path.name for path in sorted(MASK_DIR.glob("*.png")) if (IMAGE_DIR / path.name).exists()]

        self.sample_box.blockSignals(True)
        self.sample_box.clear()
        self.sample_box.addItems(samples)
        if selected in samples:
            self.sample_box.setCurrentText(selected)
        self.sample_box.blockSignals(False)
        self.load_selected_sample()

    def load_selected_sample(self) -> None:
        name = self.sample_box.currentText()
        if not name:
            self.source_image = None
            self.mask_image = None
            self.image_label.setText("没有找到匹配的图片和 mask")
            return

        try:
            self.source_image = Image.open(IMAGE_DIR / name).convert("RGBA")
            self.pseudocolor_image = self.create_pseudocolor(self.source_image)
            self.mask_image = Image.open(MASK_DIR / name).convert("L")
            if self.source_image.size != self.mask_image.size:
                raise ValueError(
                    f"图片尺寸 {self.source_image.size} 与 mask 尺寸 {self.mask_image.size} 不一致"
                )
            self.render_overlay()
        except Exception as error:
            QMessageBox.critical(self, "加载失败", str(error))

    @staticmethod
    def create_pseudocolor(image: Image.Image) -> Image.Image:
        gray = ImageOps.autocontrast(image.convert("L"), cutoff=0.5)
        # OCT-style warm palette: black -> dark brown -> orange -> gold -> pale yellow.
        anchors = [
            (0, (0, 0, 0)),
            (45, (22, 4, 1)),
            (100, (76, 15, 3)),
            (160, (163, 55, 7)),
            (215, (245, 139, 24)),
            (255, (255, 232, 135)),
        ]
        channels = [[], [], []]
        for value in range(256):
            for index in range(len(anchors) - 1):
                start_value, start_color = anchors[index]
                end_value, end_color = anchors[index + 1]
                if start_value <= value <= end_value:
                    ratio = (value - start_value) / (end_value - start_value)
                    for channel in range(3):
                        channels[channel].append(
                            round(start_color[channel] + ratio * (end_color[channel] - start_color[channel]))
                        )
                    break
        return gray.convert("RGB").point(channels[0] + channels[1] + channels[2]).convert("RGBA")

    def toggle_pseudocolor(self, enabled: bool) -> None:
        self.pseudocolor_button.setText("恢复原始图" if enabled else "转换伪彩色")
        self.render_overlay()

    def render_overlay(self) -> None:
        alpha_percent = self.alpha_slider.value()
        self.alpha_label.setText(f"{alpha_percent}%")
        if self.source_image is None or self.mask_image is None:
            return

        base_image = self.pseudocolor_image if self.pseudocolor_button.isChecked() else self.source_image
        result = base_image.copy()
        alpha = round(255 * alpha_percent / 100)
        for class_values, rgb in CLASSES.values():
            lookup = [alpha if value in class_values else 0 for value in range(256)]
            class_alpha = self.mask_image.point(lookup)
            color_layer = Image.new("RGBA", result.size, (*rgb, 0))
            color_layer.putalpha(class_alpha)
            result = Image.alpha_composite(result, color_layer)

        self.rendered_image = result.convert("RGB")
        self.show_rendered_image()

    def show_rendered_image(self) -> None:
        if self.rendered_image is None:
            return
        rgba = self.rendered_image.convert("RGBA")
        self._image_bytes = rgba.tobytes("raw", "RGBA")
        qimage = QImage(
            self._image_bytes,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format_RGBA8888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def eventFilter(self, watched, event):  # noqa: N802 (Qt API name)
        if watched is self.image_label:
            if event.type() == QEvent.MouseMove:
                self.show_pixel_value(event.pos().x(), event.pos().y())
            elif event.type() == QEvent.Leave:
                self.pixel_label.setText("将鼠标悬浮在图片上查看像素值")
        return super().eventFilter(watched, event)

    def show_pixel_value(self, mouse_x: int, mouse_y: int) -> None:
        pixmap = self.image_label.pixmap()
        if pixmap is None or self.source_image is None or self.mask_image is None:
            return

        # QLabel centers the aspect-ratio-preserving pixmap, leaving letterbox margins.
        offset_x = (self.image_label.width() - pixmap.width()) / 2
        offset_y = (self.image_label.height() - pixmap.height()) / 2
        local_x = mouse_x - offset_x
        local_y = mouse_y - offset_y
        if not (0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height()):
            self.pixel_label.setText("鼠标位于图片显示区域之外")
            return

        image_x = min(self.source_image.width - 1, int(local_x * self.source_image.width / pixmap.width()))
        image_y = min(self.source_image.height - 1, int(local_y * self.source_image.height / pixmap.height()))
        source_rgb = self.source_image.getpixel((image_x, image_y))[:3]
        display_rgb = self.rendered_image.getpixel((image_x, image_y)) if self.rendered_image else source_rgb
        mask_value = self.mask_image.getpixel((image_x, image_y))
        class_name = "background"
        for name, (values, _) in CLASSES.items():
            if mask_value in values:
                class_name = name
                break
        self.pixel_label.setText(
            f"坐标: ({image_x}, {image_y})    原图 RGB: {source_rgb}    "
            f"显示 RGB: {display_rgb}    Mask: {mask_value}    类别: {class_name}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self.show_rendered_image()

    def save_overlay(self) -> None:
        if self.rendered_image is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_path = OUTPUT_DIR / self.sample_box.currentText()
        path, _ = QFileDialog.getSaveFileName(
            self, "保存叠加图", str(default_path.resolve()), "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)"
        )
        if path:
            self.rendered_image.save(path)
            self.statusBar().showMessage(f"已保存：{path}", 5000)


def main() -> None:
    app = QApplication(sys.argv)
    viewer = OverlayViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
