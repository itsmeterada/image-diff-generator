import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog,
                             QGroupBox, QSplitter, QLineEdit, QComboBox, QProgressBar,
                             QMessageBox, QTabWidget, QDialog, QTextEdit, QCheckBox)
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QDragEnterEvent, QDropEvent

from sam3_wrapper import get_sam3_wrapper, SAM3Wrapper


class DropLabel(QLabel):
    """Custom QLabel that accepts drag and drop for images"""
    image_dropped = pyqtSignal(str)

    def __init__(self, text=""):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.default_text = text
        self.has_image = False
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f8f8f8;
                padding: 20px;
                color: #666;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #f0f8f0;
            }
        """)
        self.setMinimumSize(300, 200)
        self.setScaledContents(False)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.image_dropped.emit(file_path)
            event.acceptProposedAction()

    def load_thumbnail(self, file_path):
        """Load and display thumbnail of the dropped image"""
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # Scale thumbnail to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size() - QRect(0, 0, 40, 40).size(),  # Account for padding
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.has_image = True
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    background-color: #e8f5e9;
                    padding: 10px;
                }
            """)

    def clear_thumbnail(self):
        """Clear the thumbnail and restore default text"""
        self.clear()
        self.setText(self.default_text)
        self.has_image = False
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f8f8f8;
                padding: 20px;
                color: #666;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #f0f8f0;
            }
        """)


class ImageDiffViewer(QLabel):
    """Custom widget to display layered images with transparency control"""

    def __init__(self):
        super().__init__()
        self.image1 = None
        self.image2 = None
        self.mask = None
        self.alpha1_2 = 1.0  # Alpha between image1 and image2
        self.alpha2_mask = 1.0  # Alpha between image2 and mask
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)

    def set_images(self, img1, img2, mask):
        """Set the three images to display"""
        self.image1 = img1
        self.image2 = img2
        self.mask = mask
        self.update_display()

    def set_alpha1_2(self, alpha):
        """Set transparency between image1 and image2"""
        self.alpha1_2 = alpha / 100.0
        self.update_display()

    def set_alpha2_mask(self, alpha):
        """Set transparency between image2 and mask"""
        self.alpha2_mask = alpha / 100.0
        self.update_display()

    def update_display(self):
        """Composite the images with the current alpha values"""
        if self.image1 is None or self.image2 is None or self.mask is None:
            return

        # Start with image1
        result = self.image1.copy()
        h, w = result.shape[:2]

        # Blend image1 with image2 using alpha1_2
        img2_resized = cv2.resize(self.image2, (w, h))
        result = cv2.addWeighted(result, 1 - self.alpha1_2, img2_resized, self.alpha1_2, 0)

        # Blend result with mask using alpha2_mask
        mask_resized = cv2.resize(self.mask, (w, h))
        # Convert mask to BGR if it's grayscale
        if len(mask_resized.shape) == 2:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 1 - self.alpha2_mask, mask_resized, self.alpha2_mask, 0)

        # Convert to QPixmap and display
        self.display_image(result)

    def display_image(self, img):
        """Convert OpenCV image to QPixmap and display"""
        if img is None:
            return

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class SAM3LoaderThread(QThread):
    """Thread for loading SAM3 model without blocking UI."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, sam3_wrapper: SAM3Wrapper, use_gpu: bool = True):
        super().__init__()
        self.sam3_wrapper = sam3_wrapper
        self.use_gpu = use_gpu

    def run(self):
        success, message = self.sam3_wrapper.load_model(
            use_gpu=self.use_gpu,
            progress_callback=lambda msg: self.progress.emit(msg)
        )
        self.finished.emit(success, message)


class SAM3MaskThread(QThread):
    """Thread for generating SAM3 masks without blocking UI."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, str)

    def __init__(self, sam3_wrapper: SAM3Wrapper, image: np.ndarray, text_prompt: str, threshold: float):
        super().__init__()
        self.sam3_wrapper = sam3_wrapper
        self.image = image
        self.text_prompt = text_prompt
        self.threshold = threshold

    def run(self):
        self.progress.emit(f"Generating mask for '{self.text_prompt}'...")
        combined_mask, individual_masks, scores, message = self.sam3_wrapper.generate_mask_from_text(
            self.image, self.text_prompt, self.threshold
        )
        self.finished.emit(combined_mask, individual_masks, scores, message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image1_path = None
        self.image2_path = None
        self.image1 = None
        self.image2 = None
        self.mask = None
        self.sam3_mask = None
        self.sam3_wrapper = get_sam3_wrapper()
        self.loader_thread = None
        self.mask_thread = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Difference Generator")
        self.setGeometry(100, 100, 1200, 800)

        # Set modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #3a3a3a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #3a8f42;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5CBF60;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        # Image 1 drop zone
        img1_group = QGroupBox("Image 1 (Base)")
        img1_layout = QVBoxLayout()
        self.drop_label1 = DropLabel("Drag and drop\nImage 1 here")
        self.drop_label1.image_dropped.connect(self.load_image1)
        img1_layout.addWidget(self.drop_label1)

        browse_btn1 = QPushButton("Browse...")
        browse_btn1.clicked.connect(self.browse_image1)
        img1_layout.addWidget(browse_btn1)
        img1_group.setLayout(img1_layout)
        left_layout.addWidget(img1_group)

        # Image 2 drop zone
        img2_group = QGroupBox("Image 2 (Compare)")
        img2_layout = QVBoxLayout()
        self.drop_label2 = DropLabel("Drag and drop\nImage 2 here")
        self.drop_label2.image_dropped.connect(self.load_image2)
        img2_layout.addWidget(self.drop_label2)

        browse_btn2 = QPushButton("Browse...")
        browse_btn2.clicked.connect(self.browse_image2)
        img2_layout.addWidget(browse_btn2)
        img2_group.setLayout(img2_layout)
        left_layout.addWidget(img2_group)

        # Process button
        self.process_btn = QPushButton("Generate Difference Mask")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)

        # SAM3 Controls Group
        sam3_group = QGroupBox("SAM3 Text Prompt Segmentation")
        sam3_layout = QVBoxLayout()

        # SAM3 Status label
        self.sam3_status_label = QLabel("Status: Checking SAM3...")
        sam3_layout.addWidget(self.sam3_status_label)

        # GPU info label
        self.gpu_info_label = QLabel("")
        sam3_layout.addWidget(self.gpu_info_label)

        # Load/Unload model buttons
        model_btn_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model (GPU)")
        self.load_model_btn.clicked.connect(lambda: self.load_sam3_model(use_gpu=True))
        model_btn_layout.addWidget(self.load_model_btn)

        self.load_model_cpu_btn = QPushButton("Load (CPU)")
        self.load_model_cpu_btn.clicked.connect(lambda: self.load_sam3_model(use_gpu=False))
        model_btn_layout.addWidget(self.load_model_cpu_btn)

        self.unload_model_btn = QPushButton("Unload")
        self.unload_model_btn.clicked.connect(self.unload_sam3_model)
        self.unload_model_btn.setEnabled(False)
        model_btn_layout.addWidget(self.unload_model_btn)
        sam3_layout.addLayout(model_btn_layout)

        # Install guide button
        self.install_guide_btn = QPushButton("Show Install Guide")
        self.install_guide_btn.clicked.connect(self.show_sam3_install_instructions)
        self.install_guide_btn.setStyleSheet("background-color: #2196F3;")
        sam3_layout.addWidget(self.install_guide_btn)

        # Progress bar
        self.sam3_progress = QProgressBar()
        self.sam3_progress.setTextVisible(True)
        self.sam3_progress.setRange(0, 0)  # Indeterminate
        self.sam3_progress.hide()
        sam3_layout.addWidget(self.sam3_progress)

        # Text prompt input
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("Text Prompt:")
        self.text_prompt_input = QLineEdit()
        self.text_prompt_input.setPlaceholderText("e.g., cat, person, car (comma-separated)")
        self.text_prompt_input.returnPressed.connect(self.generate_sam3_mask)
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.text_prompt_input)
        sam3_layout.addLayout(prompt_layout)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_value_label = QLabel("0.50")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v/100:.2f}")
        )
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        sam3_layout.addLayout(threshold_layout)

        # Target image selector
        target_layout = QHBoxLayout()
        target_label = QLabel("Apply to:")
        self.target_image_combo = QComboBox()
        self.target_image_combo.addItems(["Image 1", "Image 2"])
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_image_combo)
        sam3_layout.addLayout(target_layout)

        # Generate SAM3 mask button
        self.sam3_generate_btn = QPushButton("Generate SAM3 Mask")
        self.sam3_generate_btn.clicked.connect(self.generate_sam3_mask)
        self.sam3_generate_btn.setEnabled(False)
        sam3_layout.addWidget(self.sam3_generate_btn)

        # SAM3 result label
        self.sam3_result_label = QLabel("")
        sam3_layout.addWidget(self.sam3_result_label)

        sam3_group.setLayout(sam3_layout)
        left_layout.addWidget(sam3_group)

        # Transparency controls
        controls_group = QGroupBox("Layer Transparency Controls")
        controls_layout = QVBoxLayout()

        # Slider for Image1 <-> Image2
        slider1_label = QLabel("Image 1 ⟷ Image 2")
        self.slider1 = QSlider(Qt.Orientation.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(100)
        self.slider1.valueChanged.connect(self.update_alpha1_2)
        self.slider1_value = QLabel("100%")

        slider1_layout = QVBoxLayout()
        slider1_layout.addWidget(slider1_label)
        slider1_layout.addWidget(self.slider1)
        slider1_layout.addWidget(self.slider1_value, alignment=Qt.AlignmentFlag.AlignCenter)
        controls_layout.addLayout(slider1_layout)

        # Slider for Image2 <-> Mask
        slider2_label = QLabel("Image 2 ⟷ Mask")
        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(100)
        self.slider2.valueChanged.connect(self.update_alpha2_mask)
        self.slider2_value = QLabel("100%")

        slider2_layout = QVBoxLayout()
        slider2_layout.addWidget(slider2_label)
        slider2_layout.addWidget(self.slider2)
        slider2_layout.addWidget(self.slider2_value, alignment=Qt.AlignmentFlag.AlignCenter)
        controls_layout.addLayout(slider2_layout)

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        # Edge smoothing controls
        smoothing_group = QGroupBox("Mask Edge Smoothing")
        smoothing_layout = QVBoxLayout()

        self.smoothing_checkbox = QCheckBox("Enable Edge Smoothing")
        self.smoothing_checkbox.setChecked(False)
        self.smoothing_checkbox.stateChanged.connect(self.on_smoothing_changed)
        smoothing_layout.addWidget(self.smoothing_checkbox)

        smooth_strength_layout = QHBoxLayout()
        smooth_strength_label = QLabel("Strength:")
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 50)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.setEnabled(False)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        self.smoothing_value_label = QLabel("5")
        smooth_strength_layout.addWidget(smooth_strength_label)
        smooth_strength_layout.addWidget(self.smoothing_slider)
        smooth_strength_layout.addWidget(self.smoothing_value_label)
        smoothing_layout.addLayout(smooth_strength_layout)

        smoothing_group.setLayout(smoothing_layout)
        left_layout.addWidget(smoothing_group)

        # Save buttons
        save_btn_layout = QVBoxLayout()

        self.save_btn = QPushButton("Save Mask Image")
        self.save_btn.clicked.connect(self.save_mask)
        self.save_btn.setEnabled(False)
        save_btn_layout.addWidget(self.save_btn)

        self.save_with_alpha_btn = QPushButton("Save Image 2 with Mask Alpha")
        self.save_with_alpha_btn.clicked.connect(self.save_image_with_mask_alpha)
        self.save_with_alpha_btn.setEnabled(False)
        save_btn_layout.addWidget(self.save_with_alpha_btn)

        left_layout.addLayout(save_btn_layout)

        left_layout.addStretch()

        # Right panel for display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        display_group = QGroupBox("Result Preview (Layered View)")
        display_layout = QVBoxLayout()
        self.viewer = ImageDiffViewer()
        display_layout.addWidget(self.viewer)
        display_group.setLayout(display_layout)
        right_layout.addWidget(display_group)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # Initialize SAM3 status
        self.init_sam3_status()

    def init_sam3_status(self):
        """Initialize SAM3 status and update UI accordingly."""
        if self.sam3_wrapper.is_sam3_available():
            self.sam3_status_label.setText("Status: SAM3 available (model not loaded)")
            gpu_available, gpu_info = self.sam3_wrapper.check_gpu_available()
            self.gpu_info_label.setText(gpu_info)
            if not gpu_available:
                self.load_model_btn.setEnabled(False)
        else:
            error_msg = self.sam3_wrapper.get_import_error()
            if error_msg:
                self.sam3_status_label.setText(f"Status: SAM3 not installed\n({error_msg[:50]}...)")
            else:
                self.sam3_status_label.setText("Status: SAM3 not installed")
            self.sam3_status_label.setStyleSheet("color: #ff6666;")
            self.load_model_btn.setEnabled(False)
            self.load_model_cpu_btn.setEnabled(False)
            self.sam3_generate_btn.setEnabled(False)
            self.gpu_info_label.setText("Click 'Show Install Guide' for setup instructions")

    def show_sam3_install_instructions(self):
        """Show SAM3 installation instructions in a dialog."""
        instructions = self.sam3_wrapper.get_installation_instructions()

        dialog = QDialog(self)
        dialog.setWindowTitle("SAM3 Installation Guide")
        dialog.setMinimumSize(600, 500)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(instructions)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def load_sam3_model(self, use_gpu: bool = True):
        """Load SAM3 model in a background thread."""
        if self.loader_thread is not None and self.loader_thread.isRunning():
            return

        # Disable buttons during loading
        self.load_model_btn.setEnabled(False)
        self.load_model_cpu_btn.setEnabled(False)
        self.sam3_progress.show()
        self.sam3_status_label.setText("Status: Loading model...")

        # Create and start loader thread
        self.loader_thread = SAM3LoaderThread(self.sam3_wrapper, use_gpu)
        self.loader_thread.progress.connect(self.on_sam3_load_progress)
        self.loader_thread.finished.connect(self.on_sam3_load_finished)
        self.loader_thread.start()

    def on_sam3_load_progress(self, message: str):
        """Handle SAM3 loading progress updates."""
        self.sam3_status_label.setText(f"Status: {message}")

    def on_sam3_load_finished(self, success: bool, message: str):
        """Handle SAM3 loading completion."""
        self.sam3_progress.hide()

        if success:
            self.sam3_status_label.setText(f"Status: Model loaded")
            self.sam3_status_label.setStyleSheet("color: #66ff66;")
            self.unload_model_btn.setEnabled(True)
            self.update_sam3_generate_button()
        else:
            self.sam3_status_label.setText(f"Status: {message}")
            self.sam3_status_label.setStyleSheet("color: #ff6666;")
            self.load_model_btn.setEnabled(True)
            self.load_model_cpu_btn.setEnabled(True)

            # Show detailed error message
            QMessageBox.warning(self, "SAM3 Load Error", message)

    def unload_sam3_model(self):
        """Unload SAM3 model to free memory."""
        self.sam3_wrapper.unload_model()
        self.sam3_status_label.setText("Status: Model unloaded")
        self.sam3_status_label.setStyleSheet("color: #ffffff;")
        self.load_model_btn.setEnabled(True)
        self.load_model_cpu_btn.setEnabled(True)
        self.unload_model_btn.setEnabled(False)
        self.sam3_generate_btn.setEnabled(False)

    def update_sam3_generate_button(self):
        """Update SAM3 generate button enabled state."""
        has_image = self.image1 is not None or self.image2 is not None
        model_loaded = self.sam3_wrapper.is_model_loaded()
        self.sam3_generate_btn.setEnabled(has_image and model_loaded)

    def generate_sam3_mask(self):
        """Generate mask using SAM3 with text prompt."""
        if self.mask_thread is not None and self.mask_thread.isRunning():
            return

        # Get text prompt
        text_prompt = self.text_prompt_input.text().strip()
        if not text_prompt:
            QMessageBox.warning(self, "Input Error", "Please enter a text prompt.")
            return

        # Get target image
        target_idx = self.target_image_combo.currentIndex()
        if target_idx == 0:
            target_image = self.image1
        else:
            target_image = self.image2

        if target_image is None:
            QMessageBox.warning(self, "Input Error", "Please load the target image first.")
            return

        if not self.sam3_wrapper.is_model_loaded():
            QMessageBox.warning(self, "Model Error", "Please load the SAM3 model first.")
            return

        # Get threshold
        threshold = self.threshold_slider.value() / 100.0

        # Disable generate button and show progress
        self.sam3_generate_btn.setEnabled(False)
        self.sam3_progress.show()
        self.sam3_result_label.setText("Generating mask...")

        # Create and start mask generation thread
        self.mask_thread = SAM3MaskThread(
            self.sam3_wrapper, target_image, text_prompt, threshold
        )
        self.mask_thread.progress.connect(self.on_sam3_mask_progress)
        self.mask_thread.finished.connect(self.on_sam3_mask_finished)
        self.mask_thread.start()

    def on_sam3_mask_progress(self, message: str):
        """Handle SAM3 mask generation progress."""
        self.sam3_result_label.setText(message)

    def on_sam3_mask_finished(self, combined_mask, individual_masks, scores, message: str):
        """Handle SAM3 mask generation completion."""
        self.sam3_progress.hide()
        self.sam3_generate_btn.setEnabled(True)
        self.sam3_result_label.setText(message)

        if combined_mask is None:
            return

        # Store the SAM3 mask
        self.sam3_mask = combined_mask

        # Get target image for display
        target_idx = self.target_image_combo.currentIndex()
        if target_idx == 0:
            target_image = self.image1
        else:
            target_image = self.image2

        # Create colored mask (green for SAM3 segmentation)
        h, w = target_image.shape[:2]
        mask_resized = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[:, :, 1] = mask_resized  # Green channel

        self.mask = colored_mask
        self.binary_mask = mask_resized

        # If we have both images, show layered view
        if self.image1 is not None and self.image2 is not None:
            h1, w1 = self.image1.shape[:2]
            h2, w2 = self.image2.shape[:2]
            target_h = max(h1, h2)
            target_w = max(w1, w2)

            img1_resized = cv2.resize(self.image1, (target_w, target_h))
            img2_resized = cv2.resize(self.image2, (target_w, target_h))
            mask_display = cv2.resize(colored_mask, (target_w, target_h))

            self.viewer.set_images(img1_resized, img2_resized, mask_display)
        else:
            # Show single image with mask overlay
            overlay = self.sam3_wrapper.overlay_mask_on_image(
                target_image, mask_resized, color=(0, 255, 0), alpha=0.5
            )
            self.viewer.display_image(overlay)

        self.save_btn.setEnabled(True)
        self.save_with_alpha_btn.setEnabled(True)

    def browse_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 1", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_path:
            self.load_image1(file_path)

    def browse_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 2", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_path:
            self.load_image2(file_path)

    def load_image1(self, file_path):
        self.image1_path = file_path
        self.image1 = cv2.imread(file_path)
        if self.image1 is not None:
            self.drop_label1.load_thumbnail(file_path)
            self.check_ready_to_process()
            self.update_sam3_generate_button()

    def load_image2(self, file_path):
        self.image2_path = file_path
        self.image2 = cv2.imread(file_path)
        if self.image2 is not None:
            self.drop_label2.load_thumbnail(file_path)
            self.check_ready_to_process()
            self.update_sam3_generate_button()

    def check_ready_to_process(self):
        if self.image1 is not None and self.image2 is not None:
            self.process_btn.setEnabled(True)

    def process_images(self):
        if self.image1 is None or self.image2 is None:
            return

        # Resize images to the same size (use the larger dimensions)
        h1, w1 = self.image1.shape[:2]
        h2, w2 = self.image2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)

        img1_resized = cv2.resize(self.image1, (target_w, target_h))
        img2_resized = cv2.resize(self.image2, (target_w, target_h))

        # Compute absolute difference
        diff = cv2.absdiff(img1_resized, img2_resized)

        # Convert to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary mask
        _, binary_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Create colored mask (red for differences)
        self.mask = np.zeros_like(img1_resized)
        self.mask[:, :, 2] = binary_mask  # Red channel

        # Also store the binary mask for saving
        self.binary_mask = binary_mask

        # Update viewer
        self.viewer.set_images(img1_resized, img2_resized, self.mask)

        self.save_btn.setEnabled(True)
        self.save_with_alpha_btn.setEnabled(True)

    def update_alpha1_2(self, value):
        self.slider1_value.setText(f"{value}%")
        self.viewer.set_alpha1_2(value)

    def update_alpha2_mask(self, value):
        self.slider2_value.setText(f"{value}%")
        self.viewer.set_alpha2_mask(value)

    def on_smoothing_changed(self):
        """Handle smoothing checkbox or slider change."""
        enabled = self.smoothing_checkbox.isChecked()
        self.smoothing_slider.setEnabled(enabled)
        self.smoothing_value_label.setText(str(self.smoothing_slider.value()))

    def apply_edge_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """Apply edge smoothing to mask if enabled."""
        if not self.smoothing_checkbox.isChecked():
            return mask

        strength = self.smoothing_slider.value()
        # Ensure kernel size is odd
        kernel_size = strength * 2 + 1

        # Apply Gaussian blur for smooth edges
        smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        return smoothed

    def save_mask(self):
        if self.mask is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask Image", "difference_mask.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
        )

        if file_path:
            # Apply edge smoothing if enabled
            mask_to_save = self.apply_edge_smoothing(self.binary_mask.copy())

            # Save the binary mask with alpha channel
            # Create RGBA image with white foreground and alpha channel
            h, w = mask_to_save.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 255  # Red
            rgba[:, :, 1] = 255  # Green
            rgba[:, :, 2] = 255  # Blue
            rgba[:, :, 3] = mask_to_save  # Alpha

            # Convert RGBA to BGRA for OpenCV
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(file_path, bgra)

            self.statusBar().showMessage(f"Mask saved to: {file_path}", 5000)

    def save_image_with_mask_alpha(self):
        """Save Image 2 with mask applied as alpha channel."""
        if self.image2 is None or self.binary_mask is None:
            QMessageBox.warning(self, "Error", "Image 2 and mask are required.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image with Alpha", "image_with_mask.png",
            "PNG Image (*.png);;All Files (*.*)"
        )

        if file_path:
            # Resize mask to match image2 if needed
            h, w = self.image2.shape[:2]
            mask_h, mask_w = self.binary_mask.shape[:2]

            if (h, w) != (mask_h, mask_w):
                mask_resized = cv2.resize(self.binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = self.binary_mask.copy()

            # Apply edge smoothing if enabled
            mask_resized = self.apply_edge_smoothing(mask_resized)

            # Create BGRA image (Image 2 RGB + Mask Alpha)
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            bgra[:, :, 0] = self.image2[:, :, 0]  # Blue
            bgra[:, :, 1] = self.image2[:, :, 1]  # Green
            bgra[:, :, 2] = self.image2[:, :, 2]  # Red
            bgra[:, :, 3] = mask_resized  # Alpha from mask

            cv2.imwrite(file_path, bgra)
            self.statusBar().showMessage(f"Image with alpha saved to: {file_path}", 5000)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
