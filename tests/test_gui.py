"""Hardware-free Qt workflow test; render with the platform's real font backend."""
import os
if os.name != 'nt' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from sow1_gui.window import MainWindow


class GuiWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = MainWindow()
        self.w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.errors = []
        self.w.worker.event.connect(self.record_error)
        c = self.w.config.copy()
        c.update(folder=self.tmp.name, nh=3, sweeps=2, live_points=101,
                 segments=[[1e9, 1.1e9, 5], [2e9, 2.1e9, 7]])
        self.w.load_form(c)
        self.w.show()

    def record_error(self, kind, payload):
        if kind == 'error':
            self.errors.append(payload)

    def until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), 'Timed out waiting for GUI/worker')

    def click(self, widget):
        self.assertTrue(widget.isEnabled(), widget.text())
        widget.click()
        self.until(lambda: not self.w.busy)
        self.assertEqual(self.errors, [])

    def tearDown(self):
        self.w.close()
        self.until(lambda: not self.w.worker.isRunning())
        self.app.processEvents()
        self.tmp.cleanup()

    def test_gui_sol_record_and_return_to_live(self):
        w = self.w
        self.click(w.connect_btn)
        self.click(w.cal_prepare)
        self.click(w.cal_begin)
        self.assertEqual(w.cal_step.count(), 3)
        for step in range(3):
            w.cal_step.setCurrentIndex(step)
            self.click(w.cal_measure)
        self.click(w.cal_apply)
        w.cal_note.setText('GUI simulation fixture')
        w.cal_ack.setChecked(True)
        self.click(w.cal_confirm)
        self.click(w.segment_btn)
        self.click(w.start_row_btn)
        for point in range(3):
            w.point.setValue(point)
            self.assertFalse(w.capture_btn.isEnabled())
            w.position_ack.setChecked(True)
            self.click(w.capture_btn)
        self.assertTrue(all(w.state['points']))
        self.assertIsNotNone(w.result)
        self.click(w.close_row_btn)
        self.click(w.time_btn)
        self.click(w.live_btn)
        self.until(lambda: w.last_frame is not None and w.last_frame['mode'] == 'time')
        self.assertEqual(len(w.last_frame['traces']), 6)
        # Applying ns limits must reach the engine in seconds.
        w.axes.cellWidget(0, 0).setCurrentText('ns')
        self.click(w.axis_btn)
        self.assertAlmostEqual(w.state['config']['axes'][0]['xmin'], 160 / 3e8, places=12)
        self.click(w.live_btn)
        self.click(w.disconnect_btn)
        self.assertFalse(w.state['connected'])


if __name__ == '__main__':
    unittest.main()
