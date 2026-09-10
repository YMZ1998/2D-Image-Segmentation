import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from PyQt5.QtCore import QEvent, QSettings, QSize, Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_utils import create_pseudocolor as create_pseudocolor_array
from inference_utils import newest_onnx, onnx_output_to_mask, prepare_onnx_input
from segmentation_config import (
    CLASS_COLORS,
    CLASS_DISPLAY_VALUES,
    CLASS_NAMES,
    IMAGE_SIZE,
)

DEFAULT_IMAGE_DIR = Path(r"D:\data\OCT")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# OCT 有效圆区域
OCT_OUTER_RADIUS_RATIO = 0.49


class OverlayViewer(QMainWindow):
    """
    OCT ONNX Inference Viewer
    ------------------------------------------------------------
    单图模式：
      - 左侧：数据 / 模型 / 显示设置
      - 中间：Original + Segmentation
      - 右侧：Classes / Quantification / Image Info
      - CPUExecutionProvider only
      - 无序列浏览、无自动播放、无导出
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("OCT ONNX Inference Viewer")
        self.resize(1600, 920)
        self.setMinimumSize(1280, 760)
        self.setAcceptDrops(True)

        self._init_state()
        self._build_ui()
        self._init_shortcuts()
        self._update_status_bar()

    # ==================================================================
    # State
    # ==================================================================
    def _init_state(self) -> None:
        self.settings = QSettings("2D-Image-Segmentation", "OCTCompactViewer")

        saved_dir = Path(self.settings.value("image_dir", str(DEFAULT_IMAGE_DIR)))
        self.image_dir = (
            saved_dir
            if saved_dir.is_dir()
            else (DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.is_dir() else Path.cwd())
        )

        saved_model = Path(self.settings.value("onnx_model", ""))
        self.selected_model_path = saved_model if saved_model.is_file() else None

        self.image_path: Path | None = None
        self.source_image: Image.Image | None = None
        self.pseudocolor_image: Image.Image | None = None
        self.prediction_mask: Image.Image | None = None
        self.prediction_rendered: Image.Image | None = None

        self.onnx_session = None
        self.onnx_model_path: Path | None = None

        self.zoom_factor = 1.0
        self.last_inference_ms: float | None = None

        # Keep image byte buffers alive for QImage safety.
        self._source_bytes: bytes | None = None
        self._prediction_bytes: bytes | None = None

    def _init_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_image_file)
        QShortcut(QKeySequence("F5"), self, activated=self.run_onnx_prediction)
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.choose_onnx_model)

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self) -> None:
        self.setStyleSheet(self._application_style())

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        workspace = QHBoxLayout()
        workspace.setContentsMargins(10, 10, 10, 8)
        workspace.setSpacing(10)

        workspace.addWidget(self._build_sidebar())

        center_layout = QHBoxLayout()
        center_layout.setSpacing(10)
        center_layout.addWidget(self._build_original_card(), 1)
        center_layout.addWidget(self._build_prediction_card(), 1)
        workspace.addLayout(center_layout, 1)

        workspace.addWidget(self._build_info_sidebar())

        root_layout.addLayout(workspace, 1)
        root_layout.addWidget(self._build_status_bar())

        container = QWidget()
        container.setObjectName("appRoot")
        container.setLayout(root_layout)
        self.setCentralWidget(container)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(62)

        row = QHBoxLayout(header)
        row.setContentsMargins(18, 8, 14, 8)
        row.setSpacing(8)

        logo = QLabel("◎")
        logo.setObjectName("logo")
        logo.setFixedWidth(34)
        row.addWidget(logo)

        brand = QVBoxLayout()
        brand.setSpacing(0)

        title = QLabel("OCT ONNX Inference Viewer")
        title.setObjectName("appTitle")

        subtitle = QLabel("AI-Powered Intravascular OCT Analysis")
        subtitle.setObjectName("appSubtitle")

        brand.addWidget(title)
        brand.addWidget(subtitle)
        row.addLayout(brand)

        row.addStretch(1)

        open_btn = self._header_button(
            "打开图像",
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
            self.choose_image_file,
            "Ctrl+O",
        )
        row.addWidget(open_btn)

        run_btn = self._header_button(
            "运行推理",
            self.style().standardIcon(QStyle.SP_MediaPlay),
            self.run_onnx_prediction,
            "F5",
            primary=True,
        )
        self.header_run_button = run_btn
        row.addWidget(run_btn)

        settings_btn = self._header_button(
            "设置",
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            self.choose_onnx_model,
            "Ctrl+M",
        )
        row.addWidget(settings_btn)

        return header

    def _header_button(
        self,
        text: str,
        icon,
        slot,
        shortcut: str = "",
        primary: bool = False,
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("headerPrimaryButton" if primary else "headerButton")
        button.setIcon(icon)
        button.setIconSize(QSize(15, 15))
        button.setFixedHeight(32)
        if shortcut:
            button.setToolTip(f"{text}    {shortcut}")
        button.clicked.connect(slot)
        return button

    # ------------------------------------------------------------------
    # Left sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QFrame:
        sidebar = self._card("sidebar")
        sidebar.setFixedWidth(255)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # ---- Data Input ----
        layout.addWidget(self._section_title("数据输入", "Data Input"))

        choose_image_btn = QPushButton("选择图像文件…")
        choose_image_btn.setObjectName("primaryButton")
        choose_image_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        choose_image_btn.setIconSize(QSize(14, 14))
        choose_image_btn.clicked.connect(self.choose_image_file)
        layout.addWidget(choose_image_btn)

        self.image_path_label = QLabel("未选择图像")
        self.image_path_label.setObjectName("pathLabel")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.image_path_label)

        layout.addWidget(self._divider())

        # ---- Model ----
        layout.addWidget(self._section_title("模型设置", "Model Settings"))

        model_caption = QLabel("ONNX 模型")
        model_caption.setObjectName("captionLabel")
        layout.addWidget(model_caption)

        self.model_info_label = QLabel(
            str(self.selected_model_path)
            if self.selected_model_path
            else "自动加载 save_weights 中最新 ONNX 模型"
        )
        self.model_info_label.setObjectName("pathLabel")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.model_info_label)

        model_button = QPushButton("选择 ONNX 模型…")
        model_button.clicked.connect(self.choose_onnx_model)
        layout.addWidget(model_button)

        device_row = QHBoxLayout()
        device_row.setSpacing(8)
        device_row.addWidget(QLabel("Device"))
        device_row.addStretch(1)
        device_value = QLabel("CPU")
        device_value.setObjectName("strongValue")
        device_row.addWidget(device_value)
        layout.addLayout(device_row)

        provider_row = QHBoxLayout()
        provider_row.setSpacing(8)
        provider_row.addWidget(QLabel("Provider"))
        provider_row.addStretch(1)
        self.provider_label = QLabel("CPUExecutionProvider")
        self.provider_label.setObjectName("mutedLabel")
        provider_row.addWidget(self.provider_label)
        layout.addLayout(provider_row)

        layout.addWidget(self._divider())

        # ---- Display ----
        layout.addWidget(self._section_title("显示设置", "Display"))

        self.contrast_slider, self.contrast_label = self._add_slider_row(
            layout, "对比度", 25, 200, 100
        )
        self.brightness_slider, self.brightness_label = self._add_slider_row(
            layout, "亮度", -100, 100, 0
        )

        self.contrast_slider.valueChanged.connect(self.update_display_adjustments)
        self.brightness_slider.valueChanged.connect(self.update_display_adjustments)

        reset_button = QPushButton("恢复默认值")
        reset_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reset_button.setIconSize(QSize(14, 14))
        reset_button.clicked.connect(self.reset_display_settings)
        layout.addWidget(reset_button)

        layout.addStretch(1)

        hint = QLabel("Ctrl+O 打开图像   ·   F5 运行推理")
        hint.setObjectName("footerHint")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        return sidebar

    # ------------------------------------------------------------------
    # Original viewer
    # ------------------------------------------------------------------
    def _build_original_card(self) -> QFrame:
        card = self._card("imageCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("原始图像")
        title.setObjectName("panelTitle")
        header.addWidget(title)

        subtitle = QLabel("Original")
        subtitle.setObjectName("panelSubtitle")
        header.addWidget(subtitle)

        header.addStretch(1)

        header.addWidget(self._viewer_tool_button("1:1", self.reset_zoom, 40))
        header.addWidget(self._viewer_tool_button("Fit", self.fit_to_window, 40))
        header.addWidget(self._viewer_tool_button("−", lambda: self.change_zoom(-0.1), 30))
        header.addWidget(self._viewer_tool_button("+", lambda: self.change_zoom(0.1), 30))
        header.addWidget(self._viewer_tool_button("⛶", self.fit_to_window, 30))

        layout.addLayout(header)

        self.image_label = QLabel("请选择 OCT 图像")
        self.image_label.setObjectName("imageCanvas")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 420)
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label, 1)
        # 注意:installEventFilter 必须在 addWidget 之后调用,
        # 否则 Qt 5.15 在将带事件过滤器的控件加入布局时会触发原生栈溢出崩溃 (0xc0000409 / BEX64)。
        self.image_label.installEventFilter(self)

        return card

    # ------------------------------------------------------------------
    # Segmentation viewer
    # ------------------------------------------------------------------
    def _build_prediction_card(self) -> QFrame:
        card = self._card("imageCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header1 = QHBoxLayout()
        header1.setSpacing(6)

        title = QLabel("AI 分割结果")
        title.setObjectName("panelTitle")
        header1.addWidget(title)

        subtitle = QLabel("Segmentation")
        subtitle.setObjectName("panelSubtitle")
        header1.addWidget(subtitle)

        header1.addStretch(1)

        header1.addWidget(self._viewer_tool_button("1:1", self.reset_zoom, 40))
        header1.addWidget(self._viewer_tool_button("Fit", self.fit_to_window, 40))
        header1.addWidget(self._viewer_tool_button("−", lambda: self.change_zoom(-0.1), 30))
        header1.addWidget(self._viewer_tool_button("+", lambda: self.change_zoom(0.1), 30))
        header1.addWidget(self._viewer_tool_button("⛶", self.fit_to_window, 30))

        layout.addLayout(header1)

        header2 = QHBoxLayout()
        header2.setSpacing(8)

        self.view_mode_box = QComboBox()
        self.view_mode_box.addItems(("Overlay", "Original", "Mask"))
        self.view_mode_box.setFixedWidth(94)
        self.view_mode_box.currentTextChanged.connect(self.render_prediction)
        header2.addWidget(self.view_mode_box)

        alpha_caption = QLabel("透明度")
        alpha_caption.setObjectName("captionLabel")
        header2.addWidget(alpha_caption)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(55)
        self.alpha_slider.setMinimumWidth(120)
        self.alpha_slider.valueChanged.connect(self.render_prediction)
        header2.addWidget(self.alpha_slider, 1)

        self.alpha_label = QLabel("55%")
        self.alpha_label.setObjectName("smallValue")
        self.alpha_label.setFixedWidth(40)
        self.alpha_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header2.addWidget(self.alpha_label)

        layout.addLayout(header2)

        self.prediction_label = QLabel("运行推理后显示 AI 分割结果")
        self.prediction_label.setObjectName("imageCanvas")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumSize(420, 420)
        self.prediction_label.setMouseTracking(True)
        layout.addWidget(self.prediction_label, 1)
        # 注意:installEventFilter 必须在 addWidget 之后调用(同 image_label 的修复原因)。
        self.prediction_label.installEventFilter(self)

        return card

    # ------------------------------------------------------------------
    # Right sidebar
    # ------------------------------------------------------------------
    def _build_info_sidebar(self) -> QFrame:
        sidebar = self._card("infoSidebar")
        sidebar.setFixedWidth(275)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # ---- Classes ----
        layout.addWidget(self._section_title("类别", "Classes"))

        self.class_visibility_boxes = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            row = QHBoxLayout()
            row.setSpacing(7)

            color = QLabel()
            color.setFixedSize(9, 9)
            color.setStyleSheet(
                f"background:rgb{CLASS_COLORS[class_id]};"
                "border:none;border-radius:4px;"
            )
            row.addWidget(color)

            name = QLabel(class_name)
            name.setObjectName("className")
            row.addWidget(name)

            row.addStretch(1)

            visible = QCheckBox()
            visible.setChecked(True)
            visible.setToolTip(f"显示 / 隐藏 {class_name}")
            visible.toggled.connect(self.render_prediction)
            self.class_visibility_boxes[class_id] = visible
            row.addWidget(visible)

            layout.addLayout(row)

        layout.addWidget(self._divider())

        # ---- Quantification ----
        layout.addWidget(self._section_title("定量结果", "Quantification"))

        table_header = QHBoxLayout()
        table_header.setSpacing(4)

        h1 = QLabel("类别")
        h1.setObjectName("tableHeader")
        table_header.addWidget(h1, 1)

        h2 = QLabel("像素数")
        h2.setObjectName("tableHeader")
        h2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h2.setFixedWidth(82)
        table_header.addWidget(h2)

        h3 = QLabel("占比")
        h3.setObjectName("tableHeader")
        h3.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h3.setFixedWidth(52)
        table_header.addWidget(h3)

        layout.addLayout(table_header)

        self.class_count_labels = {}
        self.class_ratio_labels = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            row = QHBoxLayout()
            row.setSpacing(4)

            name = QLabel(class_name)
            name.setObjectName("tableCell")
            row.addWidget(name, 1)

            count = QLabel("—")
            count.setObjectName("tableCellMuted")
            count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count.setFixedWidth(82)
            row.addWidget(count)

            ratio = QLabel("—")
            ratio.setObjectName("tableCellMuted")
            ratio.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            ratio.setFixedWidth(52)
            row.addWidget(ratio)

            self.class_count_labels[class_id] = count
            self.class_ratio_labels[class_id] = ratio

            layout.addLayout(row)

        layout.addWidget(self._divider())

        # ---- Image Info ----
        layout.addWidget(self._section_title("图像信息", "Image Info"))

        self.resolution_value = self._add_info_row(layout, "图像尺寸", "—")
        self.file_value = self._add_info_row(layout, "文件名", "—")
        self.model_value = self._add_info_row(layout, "模型", "—")
        self.device_value = self._add_info_row(layout, "设备", "CPU")
        self.inference_value = self._add_info_row(layout, "推理时间", "—")

        self.pixel_label = QLabel("鼠标悬浮影像查看像素与分类")
        self.pixel_label.setObjectName("pixelInfo")
        self.pixel_label.setWordWrap(True)
        self.pixel_label.setMinimumHeight(46)
        layout.addWidget(self.pixel_label)

        layout.addStretch(1)

        return sidebar

    # ------------------------------------------------------------------
    # Bottom status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("bottomBar")
        bar.setFixedHeight(32)

        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(10)

        ready_dot = QLabel("●")
        ready_dot.setObjectName("readyDot")
        row.addWidget(ready_dot)

        self.status_ready = QLabel("Ready")
        self.status_ready.setObjectName("statusText")
        row.addWidget(self.status_ready)

        row.addWidget(self._status_separator())

        self.status_model = QLabel("Model: —")
        self.status_model.setObjectName("statusText")
        row.addWidget(self.status_model)

        row.addWidget(self._status_separator())

        self.status_device = QLabel("Device: CPU")
        self.status_device.setObjectName("statusText")
        row.addWidget(self.status_device)

        row.addWidget(self._status_separator())

        self.status_image = QLabel("Image: —")
        self.status_image.setObjectName("statusText")
        row.addWidget(self.status_image)

        row.addWidget(self._status_separator())

        self.status_size = QLabel("Size: —")
        self.status_size.setObjectName("statusText")
        row.addWidget(self.status_size)

        row.addWidget(self._status_separator())

        self.status_inference = QLabel("Inference: —")
        self.status_inference.setObjectName("statusText")
        row.addWidget(self.status_inference)

        row.addStretch(1)

        return bar

    # ==================================================================
    # UI helpers
    # ==================================================================
    @staticmethod
    def _card(object_name: str) -> QFrame:
        card = QFrame()
        card.setObjectName(object_name)
        return card

    @staticmethod
    def _section_title(chinese: str, english: str) -> QLabel:
        label = QLabel(f"{chinese}    {english}")
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _divider() -> QFrame:
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        return divider

    @staticmethod
    def _status_separator() -> QLabel:
        sep = QLabel("|")
        sep.setObjectName("statusSeparator")
        return sep

    def _viewer_tool_button(self, text: str, slot, width: int) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("viewerToolButton")
        button.setFixedSize(width, 26)
        button.clicked.connect(slot)
        return button

    def _add_slider_row(
        self,
        layout: QVBoxLayout,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
    ):
        row = QHBoxLayout()
        row.setSpacing(7)

        title_label = QLabel(title)
        title_label.setFixedWidth(42)
        row.addWidget(title_label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        row.addWidget(slider, 1)

        value_label = QLabel(f"{value}%")
        value_label.setObjectName("smallValue")
        value_label.setFixedWidth(38)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label)

        layout.addLayout(row)
        return slider, value_label

    def _add_info_row(self, layout: QVBoxLayout, title: str, value: str) -> QLabel:
        row = QHBoxLayout()
        row.setSpacing(8)

        key = QLabel(title)
        key.setObjectName("infoKey")
        key.setFixedWidth(58)
        row.addWidget(key)

        val = QLabel(value)
        val.setObjectName("infoValue")
        val.setWordWrap(True)
        val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(val, 1)

        layout.addLayout(row)
        return val

    # ==================================================================
    # File loading
    # ==================================================================
    def choose_image_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 OCT 图像",
            str(self.image_dir.resolve()),
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not selected:
            return

        self.load_image(Path(selected).resolve())

    def load_image(self, path: Path) -> None:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            QMessageBox.warning(self, "提示", "请选择有效的 OCT 图像文件")
            return

        try:
            self.image_path = path
            self.image_dir = path.parent
            self.settings.setValue("image_dir", str(self.image_dir))

            self.source_image = Image.open(path).convert("RGBA")
            self.pseudocolor_image = self.create_pseudocolor(self.source_image)

            self.clear_prediction()
            self.zoom_factor = 1.0
            self.last_inference_ms = None

            self.image_path_label.setText(str(path))
            self.resolution_value.setText(
                f"{self.source_image.width} × {self.source_image.height}"
            )
            self.file_value.setText(path.name)

            self.update_source_image()
            self._update_status_bar()

            self.status_ready.setText("Ready")

        except Exception as error:
            QMessageBox.critical(self, "图像加载失败", str(error))

    def choose_onnx_model(self) -> None:
        start = (
            self.selected_model_path.parent
            if self.selected_model_path
            else PROJECT_ROOT / "save_weights"
        )

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型",
            str(start),
            "ONNX 模型 (*.onnx)",
        )
        if not selected:
            return

        self.selected_model_path = Path(selected).resolve()
        self.settings.setValue("onnx_model", str(self.selected_model_path))

        self.model_info_label.setText(str(self.selected_model_path))
        self.model_value.setText(self.selected_model_path.name)

        self.onnx_session = None
        self.onnx_model_path = None

        self._update_status_bar()

    def resolve_onnx_model(self) -> Path:
        if self.selected_model_path and self.selected_model_path.is_file():
            return self.selected_model_path
        return newest_onnx(PROJECT_ROOT / "save_weights")

    # ==================================================================
    # Drag/drop
    # ==================================================================
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if not event.mimeData().hasUrls():
            return

        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                event.acceptProposedAction()
                return

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile()).resolve()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                self.load_image(path)
                event.acceptProposedAction()
                return

    # ==================================================================
    # Display
    # ==================================================================
    @staticmethod
    def create_pseudocolor(image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"))
        return Image.fromarray(create_pseudocolor_array(gray)).convert("RGBA")

    def adjusted_gray(self) -> np.ndarray:
        if self.source_image is None:
            raise RuntimeError("没有加载图像")

        gray = np.asarray(self.source_image.convert("L"), dtype=np.float32)
        contrast = self.contrast_slider.value() / 100.0
        brightness = self.brightness_slider.value() * 2.55

        adjusted = (gray - 127.5) * contrast + 127.5 + brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def get_base_image(self) -> Image.Image:
        gray = self.adjusted_gray()
        return Image.fromarray(gray, mode="L").convert("RGBA")

    def update_display_adjustments(self) -> None:
        self.contrast_label.setText(f"{self.contrast_slider.value()}%")
        self.brightness_label.setText(f"{self.brightness_slider.value()}%")

        if self.source_image is not None:
            self.update_source_image()

        if self.prediction_mask is not None:
            self.render_prediction()

    def reset_display_settings(self) -> None:
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)

    def update_source_image(self) -> None:
        if self.source_image is None:
            return

        self.set_label_image(
            self.image_label,
            self.get_base_image().convert("RGB"),
            source=True,
        )

    # ------------------------------------------------------------------
    # Prediction rendering modes
    # ------------------------------------------------------------------
    def compose_overlay(self, mask: Image.Image) -> Image.Image:
        if self.source_image is None:
            raise RuntimeError("没有加载图像")

        result = self.get_base_image().convert("RGBA").copy()
        alpha = round(255 * self.alpha_slider.value() / 100)

        for class_id in range(1, len(CLASS_NAMES)):
            box = self.class_visibility_boxes.get(class_id)
            if box is not None and not box.isChecked():
                continue

            class_values = CLASS_DISPLAY_VALUES[class_id]
            rgb = CLASS_COLORS[class_id]

            lookup = [alpha if value in class_values else 0 for value in range(256)]
            class_alpha = mask.point(lookup)

            color_layer = Image.new("RGBA", result.size, (*rgb, 0))
            color_layer.putalpha(class_alpha)
            result = Image.alpha_composite(result, color_layer)

        return result.convert("RGB")

    def compose_mask_only(self, mask: Image.Image) -> Image.Image:
        if self.source_image is None:
            raise RuntimeError("没有加载图像")

        output = Image.new("RGB", self.source_image.size, (0, 0, 0))

        for class_id in range(len(CLASS_NAMES)):
            box = self.class_visibility_boxes.get(class_id)
            if box is not None and not box.isChecked():
                continue

            class_values = CLASS_DISPLAY_VALUES[class_id]
            rgb = CLASS_COLORS[class_id]

            lookup = [255 if value in class_values else 0 for value in range(256)]
            alpha = mask.point(lookup)

            layer = Image.new("RGB", output.size, rgb)
            output.paste(layer, mask=alpha)

        return output

    def render_prediction(self, *_args) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value()}%")

        if self.source_image is None:
            return

        mode = self.view_mode_box.currentText()

        if mode == "Original":
            rendered = self.get_base_image().convert("RGB")

        elif self.prediction_mask is None:
            self.prediction_label.clear()
            self.prediction_label.setText("运行推理后显示 AI 分割结果")
            return

        elif mode == "Mask":
            rendered = self.compose_mask_only(self.prediction_mask)

        else:
            rendered = self.compose_overlay(self.prediction_mask)

        self.prediction_rendered = rendered
        self.set_label_image(
            self.prediction_label,
            rendered,
            source=False,
        )

    # ------------------------------------------------------------------
    # Image scaling
    # ------------------------------------------------------------------
    def set_label_image(
        self,
        label: QLabel,
        image: Image.Image,
        source: bool = False,
    ) -> None:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")

        if source:
            self._source_bytes = data
        else:
            self._prediction_bytes = data

        qimage = QImage(
            data,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format_RGBA8888,
        ).copy()

        available = label.size()

        base_pixmap = QPixmap.fromImage(qimage)
        fit_pixmap = base_pixmap.scaled(
            available,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        if self.zoom_factor != 1.0:
            zoomed_size = QSize(
                max(1, round(fit_pixmap.width() * self.zoom_factor)),
                max(1, round(fit_pixmap.height() * self.zoom_factor)),
            )
            fit_pixmap = base_pixmap.scaled(
                zoomed_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

        label.setPixmap(fit_pixmap)

    def change_zoom(self, delta: float) -> None:
        self.zoom_factor = min(
            2.5,
            max(0.5, round(self.zoom_factor + delta, 2)),
        )

        if self.source_image is not None:
            self.update_source_image()

        if self.prediction_rendered is not None:
            self.set_label_image(
                self.prediction_label,
                self.prediction_rendered,
                source=False,
            )

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0

        if self.source_image is not None:
            self.update_source_image()

        if self.prediction_rendered is not None:
            self.set_label_image(
                self.prediction_label,
                self.prediction_rendered,
                source=False,
            )

    def fit_to_window(self) -> None:
        self.reset_zoom()

    def clear_prediction(self) -> None:
        self.prediction_mask = None
        self.prediction_rendered = None
        self.last_inference_ms = None

        if hasattr(self, "prediction_label"):
            self.prediction_label.clear()
            self.prediction_label.setText("运行推理后显示 AI 分割结果")

        if hasattr(self, "inference_value"):
            self.inference_value.setText("—")

        if hasattr(self, "class_count_labels"):
            for label in self.class_count_labels.values():
                label.setText("—")

        if hasattr(self, "class_ratio_labels"):
            for label in self.class_ratio_labels.values():
                label.setText("—")

        self._update_status_bar()

    # ==================================================================
    # OCT region masking
    # ==================================================================
    @staticmethod
    def mask_oct_region(gray: np.ndarray) -> np.ndarray:
        """只保留 OCT 外圆内部区域。"""
        h, w = gray.shape
        cx = w / 2.0
        cy = h / 2.0
        outer_radius = min(w, h) * OCT_OUTER_RADIUS_RATIO

        yy, xx = np.ogrid[:h, :w]
        valid = (xx - cx) ** 2 + (yy - cy) ** 2 <= outer_radius ** 2

        result = np.zeros_like(gray)
        result[valid] = gray[valid]
        return result

    @staticmethod
    def mask_prediction_region(mask: np.ndarray) -> np.ndarray:
        """预测结果只保留 OCT 外圆内部区域。"""
        h, w = mask.shape
        cx = w / 2.0
        cy = h / 2.0
        outer_radius = min(w, h) * OCT_OUTER_RADIUS_RATIO

        yy, xx = np.ogrid[:h, :w]
        valid = (xx - cx) ** 2 + (yy - cy) ** 2 <= outer_radius ** 2

        result = np.zeros_like(mask)
        result[valid] = mask[valid]
        return result

    # ==================================================================
    # ONNX Runtime CPU inference
    # ==================================================================
    def run_onnx_prediction(self) -> bool:
        if self.source_image is None:
            QMessageBox.information(self, "提示", "请先选择 OCT 图像")
            return False

        self.header_run_button.setEnabled(False)
        self.header_run_button.setText("推理中…")
        self.status_ready.setText("Inferencing")
        QApplication.processEvents()

        try:
            import onnxruntime as ort

            model_path = self.resolve_onnx_model()

            if self.onnx_session is None or model_path != self.onnx_model_path:
                available = ort.get_available_providers()

                if "CPUExecutionProvider" not in available:
                    raise RuntimeError("当前 ONNX Runtime 不支持 CPUExecutionProvider")

                self.onnx_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                self.onnx_model_path = model_path

            input_meta = self.onnx_session.get_inputs()[0]

            gray = np.asarray(self.source_image.convert("L"))
            gray = self.mask_oct_region(gray)

            tensor, _, _ = prepare_onnx_input(
                gray,
                input_meta.shape,
                IMAGE_SIZE,
            )

            start = time.perf_counter()
            output = self.onnx_session.run(
                None,
                {input_meta.name: tensor},
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            prediction = onnx_output_to_mask(output)
            prediction = np.asarray(
                Image.fromarray(prediction).resize(
                    self.source_image.size,
                    Image.Resampling.NEAREST,
                )
            )

            prediction = self.mask_prediction_region(prediction)

            self.prediction_mask = Image.fromarray(
                prediction.astype(np.uint8),
                mode="L",
            )

            self.last_inference_ms = elapsed_ms

            self.render_prediction()

            # Stats
            counts = np.bincount(
                prediction.ravel(),
                minlength=len(CLASS_NAMES),
            )
            total = max(1, prediction.size)

            for class_id in range(len(CLASS_NAMES)):
                count = int(counts[class_id])
                ratio = count / total * 100.0

                self.class_count_labels[class_id].setText(f"{count:,}")
                self.class_ratio_labels[class_id].setText(f"{ratio:.1f}%")

            provider = self.onnx_session.get_providers()[0]

            self.provider_label.setText(provider)
            self.model_info_label.setText(str(model_path))
            self.model_value.setText(model_path.name)
            self.device_value.setText("CPU")
            self.inference_value.setText(f"{elapsed_ms:.1f} ms")

            self.status_ready.setText("Ready")
            self._update_status_bar()
            return True

        except Exception as error:
            self.status_ready.setText("Error")
            QMessageBox.critical(self, "ONNX 推理失败", str(error))
            return False

        finally:
            self.header_run_button.setEnabled(True)
            self.header_run_button.setText("运行推理")

    # ==================================================================
    # Mouse interaction
    # ==================================================================
    def eventFilter(self, watched, event):
        if watched in (self.image_label, self.prediction_label):
            if event.type() == QEvent.MouseMove:
                self.show_pixel_value(
                    watched,
                    event.pos().x(),
                    event.pos().y(),
                )
            elif event.type() == QEvent.Wheel:
                delta = 0.1 if event.angleDelta().y() > 0 else -0.1
                self.change_zoom(delta)
                event.accept()
                return True
            elif event.type() == QEvent.Leave:
                self.pixel_label.setText("鼠标悬浮影像查看像素与分类")

        return super().eventFilter(watched, event)

    def show_pixel_value(
        self,
        label: QLabel,
        mouse_x: int,
        mouse_y: int,
    ) -> None:
        pixmap = label.pixmap()

        if pixmap is None or self.source_image is None:
            return

        offset_x = (label.width() - pixmap.width()) / 2
        offset_y = (label.height() - pixmap.height()) / 2

        local_x = mouse_x - offset_x
        local_y = mouse_y - offset_y

        if not (
            0 <= local_x < pixmap.width()
            and 0 <= local_y < pixmap.height()
        ):
            self.pixel_label.setText("鼠标位于影像区域之外")
            return

        image_x = min(
            self.source_image.width - 1,
            int(local_x * self.source_image.width / pixmap.width()),
        )
        image_y = min(
            self.source_image.height - 1,
            int(local_y * self.source_image.height / pixmap.height()),
        )

        gray = self.source_image.convert("L").getpixel((image_x, image_y))

        text = f"({image_x}, {image_y})   Gray: {gray}"

        if self.prediction_mask is not None:
            value = self.prediction_mask.getpixel((image_x, image_y))
            class_name = CLASS_NAMES[0]

            for class_id in range(len(CLASS_NAMES)):
                if value in CLASS_DISPLAY_VALUES[class_id]:
                    class_name = CLASS_NAMES[class_id]
                    break

            text += f"\nClass: {class_name}   Value: {value}"

        self.pixel_label.setText(text)

    # ==================================================================
    # Resize / status
    # ==================================================================
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if self.source_image is not None:
            self.update_source_image()

        if self.prediction_rendered is not None:
            self.set_label_image(
                self.prediction_label,
                self.prediction_rendered,
                source=False,
            )

    def _update_status_bar(self) -> None:
        if not hasattr(self, "status_model"):
            return

        model_name = (
            self.selected_model_path.name
            if self.selected_model_path
            else (
                self.onnx_model_path.name
                if self.onnx_model_path
                else "—"
            )
        )

        image_name = self.image_path.name if self.image_path else "—"

        image_size = (
            f"{self.source_image.width} × {self.source_image.height}"
            if self.source_image is not None
            else "—"
        )

        inference = (
            f"{self.last_inference_ms:.1f} ms"
            if self.last_inference_ms is not None
            else "—"
        )

        self.status_model.setText(f"Model: {model_name}")
        self.status_device.setText("Device: CPU")
        self.status_image.setText(f"Image: {image_name}")
        self.status_size.setText(f"Size: {image_size}")
        self.status_inference.setText(f"Inference: {inference}")

    # ==================================================================
    # Style
    # ==================================================================
    @staticmethod
    def _application_style() -> str:
        return r"""
        /* ============================================================
           Root
           ============================================================ */
        QWidget#appRoot {
            background: #071624;
            color: #D5E2EC;
            font-family: "Microsoft YaHei UI";
            font-size: 11px;
        }

        QLabel {
            color: #D5E2EC;
            background: transparent;
        }

        /* ============================================================
           Header
           ============================================================ */
        QFrame#header {
            background: #0A1D2E;
            border: none;
            border-bottom: 1px solid #18354C;
        }

        QLabel#logo {
            color: #1597E8;
            font-size: 28px;
            font-weight: 600;
        }

        QLabel#appTitle {
            color: #F0F6FA;
            font-size: 16px;
            font-weight: 600;
        }

        QLabel#appSubtitle {
            color: #70899B;
            font-size: 9px;
        }

        QPushButton#headerButton,
        QPushButton#headerPrimaryButton {
            min-width: 92px;
            max-height: 32px;
            padding: 0 12px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }

        QPushButton#headerButton {
            background: transparent;
            border: 1px solid #1C3A52;
            color: #CBD8E2;
        }

        QPushButton#headerButton:hover {
            background: #102A40;
            border-color: #2D668D;
        }

        QPushButton#headerPrimaryButton {
            background: #0E7FCB;
            border: 1px solid #168FDF;
            color: white;
        }

        QPushButton#headerPrimaryButton:hover {
            background: #118CDB;
        }

        /* ============================================================
           Panels
           ============================================================ */
        QFrame#sidebar,
        QFrame#infoSidebar {
            background: #0B2032;
            border: 1px solid #173A53;
            border-radius: 6px;
        }

        QFrame#imageCard {
            background: #0A1D2E;
            border: 1px solid #173A53;
            border-radius: 6px;
        }

        QLabel#sectionTitle {
            color: #E8F0F5;
            font-size: 12px;
            font-weight: 600;
            padding: 2px 0 4px 0;
        }

        QLabel#panelTitle {
            color: #E8F0F5;
            font-size: 12px;
            font-weight: 600;
        }

        QLabel#panelSubtitle {
            color: #8AA2B5;
            font-size: 10px;
        }

        QLabel#captionLabel {
            color: #8BA2B4;
            font-size: 10px;
        }

        QLabel#strongValue {
            color: #E5F2FB;
            font-size: 11px;
            font-weight: 600;
        }

        QLabel#mutedLabel {
            color: #7F98AA;
            font-size: 10px;
        }

        QLabel#footerHint {
            color: #536D80;
            font-size: 9px;
        }

        QFrame#divider {
            background: #17364D;
            border: none;
            max-height: 1px;
            min-height: 1px;
            margin: 5px 0;
        }

        /* ============================================================
           Path labels
           ============================================================ */
        QLabel#pathLabel {
            color: #A9BCC9;
            background: #0D263A;
            border: 1px solid #1B425E;
            border-radius: 4px;
            padding: 6px 7px;
            font-size: 10px;
        }

        /* ============================================================
           Buttons
           ============================================================ */
        QPushButton {
            min-height: 27px;
            max-height: 29px;
            padding: 0 9px;
            color: #D5E2EC;
            background: #0F2A40;
            border: 1px solid #234B66;
            border-radius: 4px;
            font-size: 10px;
        }

        QPushButton:hover {
            background: #143650;
            border-color: #3274A0;
        }

        QPushButton:pressed {
            background: #0C2437;
        }

        QPushButton#primaryButton {
            background: #0F82D0;
            border-color: #168FDF;
            color: white;
            font-weight: 600;
        }

        QPushButton#primaryButton:hover {
            background: #118DDE;
        }

        QPushButton#viewerToolButton {
            min-height: 24px;
            max-height: 24px;
            padding: 0;
            background: #0B2235;
            border: 1px solid #23465F;
            color: #BED0DC;
            border-radius: 4px;
            font-size: 10px;
        }

        QPushButton#viewerToolButton:hover {
            background: #12304A;
            border-color: #3680AA;
        }

        /* ============================================================
           ComboBox
           ============================================================ */
        QComboBox {
            min-height: 26px;
            max-height: 26px;
            color: #D7E4ED;
            background: #0D263A;
            border: 1px solid #234B66;
            border-radius: 4px;
            padding: 0 7px;
            font-size: 10px;
        }

        QComboBox:hover {
            border-color: #337AA5;
        }

        QComboBox QAbstractItemView {
            background: #0D263A;
            color: #D7E4ED;
            border: 1px solid #234B66;
            selection-background-color: #126FAF;
        }

        /* ============================================================
           Slider
           ============================================================ */
        QSlider::groove:horizontal {
            height: 3px;
            background: #27465A;
            border-radius: 1px;
        }

        QSlider::sub-page:horizontal {
            background: #168DD9;
            border-radius: 1px;
        }

        QSlider::handle:horizontal {
            width: 10px;
            height: 10px;
            margin: -4px 0;
            background: #1B9BED;
            border: none;
            border-radius: 5px;
        }

        QLabel#smallValue {
            color: #B4C7D4;
            font-size: 9px;
        }

        /* ============================================================
           Checkbox
           ============================================================ */
        QCheckBox {
            spacing: 4px;
        }

        QCheckBox::indicator {
            width: 12px;
            height: 12px;
        }

        QCheckBox::indicator:unchecked {
            background: #0A1C2C;
            border: 1px solid #365A73;
            border-radius: 3px;
        }

        QCheckBox::indicator:checked {
            background: #168DDD;
            border: 1px solid #249CEB;
            border-radius: 3px;
        }

        /* ============================================================
           Viewer
           ============================================================ */
        QLabel#imageCanvas {
            background: #000000;
            color: #6F8494;
            border: none;
            border-radius: 4px;
        }

        /* ============================================================
           Right panel
           ============================================================ */
        QLabel#className {
            color: #D2E0E9;
            font-size: 10px;
        }

        QLabel#tableHeader {
            color: #7891A4;
            font-size: 9px;
            padding-bottom: 2px;
        }

        QLabel#tableCell {
            color: #C9D8E2;
            font-size: 10px;
            padding: 1px 0;
        }

        QLabel#tableCellMuted {
            color: #91A7B7;
            font-size: 10px;
            padding: 1px 0;
        }

        QLabel#infoKey {
            color: #7891A4;
            font-size: 9px;
        }

        QLabel#infoValue {
            color: #C7D7E1;
            font-size: 10px;
        }

        QLabel#pixelInfo {
            color: #8FA6B6;
            background: #0D263A;
            border: 1px solid #1B425E;
            border-radius: 4px;
            padding: 6px;
            font-size: 9px;
        }

        /* ============================================================
           Bottom status
           ============================================================ */
        QFrame#bottomBar {
            background: #081A29;
            border: none;
            border-top: 1px solid #18354C;
        }

        QLabel#readyDot {
            color: #2BD28A;
            font-size: 10px;
        }

        QLabel#statusText {
            color: #8FA5B5;
            font-size: 9px;
        }

        QLabel#statusSeparator {
            color: #29475C;
            font-size: 9px;
        }

        /* ============================================================
           Tooltips
           ============================================================ */
        QToolTip {
            background: #102A40;
            color: #E6EFF5;
            border: 1px solid #315B77;
            padding: 4px;
        }
        """


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    viewer = OverlayViewer()
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
