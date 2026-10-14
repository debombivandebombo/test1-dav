import argparse
import sys
from PySide6.QtWidgets import QApplication
from .window import MainWindow


def main():
    parser = argparse.ArgumentParser(description='SOW1 ISAC measurement console')
    parser.add_argument('--demo', action='store_true', help='Connect synthetic instrument and show live plots')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('SOW1 Measurement Console')
    window = MainWindow(demo=args.demo)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
