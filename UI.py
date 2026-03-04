from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QCheckBox
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt

from CV_Module import DocumentScanner

from PIL import Image  # pip install pillow


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.file_paths: list[str] = []
        self.scanned_paths: list[str] = []

        # window setting
        self.setMinimumSize(600, 500)
        self.setWindowTitle("Document Scanner")
        self.setWindowIcon(QIcon("Icon.png"))

        # Style
        text_style = self.font()
        text_style.setBold(True)
        text_style.setPointSize(17)

        button_style = self.font()
        button_style.setPointSize(12)
        button_width = 200
        button_height = 35

        # ==== Element =====
        self.upload_label = QLabel("Upload your image(s)")
        self.upload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_label.setFont(text_style)

        self.image = QLabel("No image uploaded")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet("border: 2px dashed gray;")
        self.image.setMinimumHeight(250)

        self.upload_button = QPushButton("Upload (Multiple)")
        self.upload_button.setFont(button_style)
        self.upload_button.setFixedSize(button_width, button_height)

        self.scan_button = QPushButton("Scan")
        self.scan_button.setFont(button_style)
        self.scan_button.setFixedSize(button_width, button_height)

        # NEW: Checkbox
        self.export_pdf_checkbox = QCheckBox("Export scanned pages to a single PDF")
        self.export_pdf_checkbox.setChecked(False)

        self.result = QLabel("", alignment=Qt.AlignmentFlag.AlignHCenter)

        # Action
        self.upload_button.clicked.connect(self.upload_multiple)
        self.scan_button.clicked.connect(self.scan)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.upload_label)
        layout.addWidget(self.image, stretch=1)
        layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.export_pdf_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.scan_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result)

        centerWidget = QWidget()
        centerWidget.setLayout(layout)
        self.setCentralWidget(centerWidget)

    def upload_multiple(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not paths:
            return

        self.file_paths = paths
        self.scanned_paths = []

        # show the first image as preview
        first = self.file_paths[0]
        pixmap = QPixmap(first).scaled(
            self.image.width(),
            self.image.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image.setPixmap(pixmap)
        self.image.setText("")

        self.result.setText(f"Selected {len(self.file_paths)} image(s).")
        print("Selected files:", self.file_paths)

    def scan(self):
        if not self.file_paths:
            self.result.setText("Please upload image(s) first.")
            return

        scanner = DocumentScanner()
        self.scanned_paths = []

        for i, path in enumerate(self.file_paths, start=1):
            self.result.setText(f"Scanning {i}/{len(self.file_paths)}...")
            QApplication.processEvents()  # keep UI responsive

            scanned_path = scanner.scan(path, save=True)
            if scanned_path is None:
                self.result.setText(f"Failed on: {path}")
                return
            self.scanned_paths.append(scanned_path)

        # preview the last scanned result
        last_scanned = self.scanned_paths[-1]
        pixmap = QPixmap(last_scanned).scaled(
            self.image.width(),
            self.image.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image.setPixmap(pixmap)
        self.image.setText("")

        # if checked -> export to pdf
        if self.export_pdf_checkbox.isChecked():
            default_name = "scanned.pdf"
            pdf_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF",
                default_name,
                "PDF Files (*.pdf)"
            )
            if pdf_path:
                ok, err = self.save_images_to_pdf(self.scanned_paths, pdf_path)
                if not ok:
                    self.result.setText(f"PDF export failed: {err}")
                    return
                self.result.setText(f"Scan complete! PDF saved: {pdf_path}")
                return

        self.result.setText("Scan complete!")

    def save_images_to_pdf(self, image_paths: list[str], pdf_path: str) -> tuple[bool, str]:
        try:
            pil_images = []
            for p in image_paths:
                img = Image.open(p).convert("RGB")  # PDF needs RGB
                pil_images.append(img)

            if not pil_images:
                return False, "No images to export."

            first, rest = pil_images[0], pil_images[1:]
            first.save(pdf_path, save_all=True, append_images=rest)
            return True, ""
        except Exception as e:
            return False, str(e)


app = QApplication([])
window = Window()
window.show()
app.exec()