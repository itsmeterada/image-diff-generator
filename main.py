import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog,
                             QGroupBox, QSplitter)
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QDragEnterEvent, QDropEvent


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image1_path = None
        self.image2_path = None
        self.image1 = None
        self.image2 = None
        self.mask = None

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

        # Save button
        self.save_btn = QPushButton("Save Mask Image")
        self.save_btn.clicked.connect(self.save_mask)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)

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

    def load_image2(self, file_path):
        self.image2_path = file_path
        self.image2 = cv2.imread(file_path)
        if self.image2 is not None:
            self.drop_label2.load_thumbnail(file_path)
            self.check_ready_to_process()

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

    def update_alpha1_2(self, value):
        self.slider1_value.setText(f"{value}%")
        self.viewer.set_alpha1_2(value)

    def update_alpha2_mask(self, value):
        self.slider2_value.setText(f"{value}%")
        self.viewer.set_alpha2_mask(value)

    def save_mask(self):
        if self.mask is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask Image", "difference_mask.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
        )

        if file_path:
            # Save the binary mask with alpha channel
            # Create RGBA image with white foreground and alpha channel
            h, w = self.binary_mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 255  # Red
            rgba[:, :, 1] = 255  # Green
            rgba[:, :, 2] = 255  # Blue
            rgba[:, :, 3] = self.binary_mask  # Alpha

            # Convert RGBA to BGRA for OpenCV
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(file_path, bgra)

            self.statusBar().showMessage(f"Mask saved to: {file_path}", 5000)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
