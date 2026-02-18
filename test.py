from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

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
        # Top Text
        self.upload_label = QLabel("Upload your image")
        self.upload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_label.setFont(text_style)

        # Upload image holder
        self.image = QLabel("No image uploaded")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet("border: 2px dashed gray;")
        self.image.setMinimumHeight(250)

        # Buttons
        self.upload_button = QPushButton("Upload")
        self.upload_button.setFont(button_style)
        self.upload_button.setFixedSize(button_width, button_height)
        self.scan_button = QPushButton("Scan")
        self.scan_button.setFont(button_style)
        self.scan_button.setFixedSize(button_width, button_height)

        # Result Image
        self.result = QLabel("", alignment=Qt.AlignmentFlag.AlignHCenter)

        # Action
        self.upload_button.clicked.connect(self.upload)
        self.scan_button.clicked.connect(self.scan)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.upload_label)
        layout.addWidget(self.image, stretch=1)
        layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.scan_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result)

        # Add to window
        centerWidget = QWidget()
        centerWidget.setLayout(layout)
        self.setCentralWidget(centerWidget)

    def upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(
                self.image.width(),
                self.image.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image.setPixmap(pixmap)
            self.image.setText("")
            print(file_path)


    def scan(self):
        if self.image.pixmap() is None:
            self.result.setText("Please Press Scan")
        else:
            self.result.setText("Scan complete!")

        # นะโมใส่ code นี่นะ รูปคือ file_path

app = QApplication([])
window = Window()

window.show()
app.exec()