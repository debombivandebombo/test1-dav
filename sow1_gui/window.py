"""Qt desktop console. Hardware I/O is confined to InstrumentWorker."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QSplitter, QScrollArea, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPlainTextEdit, QFileDialog, QMessageBox, QProgressBar, QCheckBox,
)
from .config import defaults, validate
from .storage import read_result
from .worker import InstrumentWorker

STYLE = '''
QMainWindow, QWidget { background: #edf2f6; color: #17283a; font-family: 'Segoe UI'; font-size: 12px; }
QLabel { background: transparent; }
QLabel#title { color: #122b41; font-size: 25px; font-weight: 700; }
QLabel#subtitle { color: #526b7c; }
QLabel#badge { background: #15384c; color: #b4f5df; border-radius: 6px; padding: 9px; font-weight: 600; }
QLabel#error { background: #fff0df; color: #8b360f; border: 1px solid #e3b88d; padding: 8px; border-radius: 5px; }
QGroupBox { background: #ffffff; border: 1px solid #d5e0e7; border-radius: 7px; margin-top: 15px; padding: 13px 9px 9px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background: #ffffff; border: 1px solid #c4d2dd; border-radius: 4px; padding: 5px; selection-background-color: #b8e6db; }
QPushButton { background: #ffffff; border: 1px solid #b8cad6; border-radius: 5px; padding: 8px 11px; font-weight: 600; }
QPushButton:hover { background: #dceeea; border-color: #218979; }
QPushButton:disabled { color: #8b9caa; background: #e7edf1; border-color: #d5e0e7; }
QPushButton#primary { background: #087b70; color: white; border-color: #087b70; }
QPushButton#primary:disabled { background: #e7edf1; color: #8b9caa; border-color: #d5e0e7; }
QPushButton#stop { background: #fff2e3; color: #9b4d12; border-color: #e7c298; }
QTabWidget::pane { border: 1px solid #cfdae2; border-radius: 5px; }
QTabBar::tab { padding: 9px 11px; background: #dfe8ee; }
QTabBar::tab:selected { background: #ffffff; color: #087b70; font-weight: 600; }
QTableWidget { background: white; gridline-color: #dce6ed; border: 1px solid #d3dfe7; }
QHeaderView::section { background: #e3edf2; padding: 5px; border: none; }
QProgressBar { border: none; border-radius: 4px; background: #dce6ec; text-align: center; min-height: 20px; }
QProgressBar::chunk { background: #43bba0; border-radius: 4px; }
'''


def number(value, minimum=0, maximum=1e9, decimals=3):
    w = QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setValue(value)
    w.setKeyboardTracking(False)
    return w


def integer(value, minimum=0, maximum=100000):
    w = QSpinBox()
    w.setRange(minimum, maximum)
    w.setValue(value)
    w.setKeyboardTracking(False)
    return w


def button(text, callback, role=''):
    w = QPushButton(text)
    w.setObjectName(role)
    w.clicked.connect(callback)
    return w


def note(text):
    w = QLabel(text)
    w.setWordWrap(True)
    w.setObjectName('subtitle')
    return w


def group(title):
    w = QGroupBox(title)
    layout = QFormLayout(w)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    return w, layout


def scroll_page(tabs, title):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    page = QWidget()
    layout = QVBoxLayout(page)
    scroll.setWidget(page)
    tabs.addTab(scroll, title)
    return layout


class MainWindow(QMainWindow):
    def __init__(self, demo=False):
        super().__init__()
        self.setWindowTitle('SOW1 · ISAC Measurement Console')
        self.resize(1520, 980)
        self.setStyleSheet(STYLE)
        self.config = defaults()
        self.state = dict(connected=False, mode='disconnected', row_active=False, calibrated=False, points=[], live=False)
        self.busy = False
        self.closing = False
        self.demo = demo
        self.demo_stage = 0
        self.last_frame = None
        self.result = None
        self.actions = []
        self.worker = InstrumentWorker()
        self.worker.event.connect(self.on_event)
        self.worker.finished.connect(self.on_worker_finished)
        self._build()
        self.load_form(self.config)
        self.worker.start()
        self.refresh_controls()
        if demo:
            QTimer.singleShot(100, self.connect_device)

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 12)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel('SOW1  /  Measurement Console')
        title.setObjectName('title')
        titles.addWidget(title)
        titles.addWidget(note('ISAC FIELD CAMPAIGN     •     Inspect the channel → restore calibration → acquire each position'))
        heading.addLayout(titles, 1)
        self.badge = QLabel('DISCONNECTED')
        self.badge.setObjectName('badge')
        heading.addWidget(self.badge)
        outer.addLayout(heading)
        connection = QHBoxLayout()
        self.device_type = QComboBox()
        self.device_type.addItems(['Simulation', 'Keysight VNA'])
        self.address = QLineEdit()
        self.address.setMinimumWidth(300)
        self.address.setPlaceholderText('VISA resource (leave blank to auto-detect Keysight)')
        self.connect_btn = button('Connect', self.connect_device)
        self.disconnect_btn = button('Disconnect', lambda: self.send('disconnect'))
        self.identity = note('Simulation opens no hardware connection.')
        for w in (self.device_type, self.address, self.connect_btn, self.disconnect_btn):
            connection.addWidget(w)
        connection.addWidget(self.identity, 1)
        outer.addLayout(connection)
        modes = QHBoxLayout()
        self.time_btn = button('1  Inspect · time domain', lambda: self.apply_mode('time'), 'primary')
        self.segment_btn = button('2  Restore · segmented', lambda: self.apply_mode('segmented'))
        self.live_btn = button('Start live view', self.toggle_live)
        self.stop_btn = button('Stop after current sweep', self.stop, 'stop')
        self.auto_btn = button('Autoscale VNA + plots', self.autoscale)
        for w in (self.time_btn, self.segment_btn, self.live_btn, self.auto_btn, self.stop_btn):
            modes.addWidget(w)
        outer.addLayout(modes)
        self.error_label = QLabel()
        self.error_label.setObjectName('error')
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        outer.addWidget(self.error_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 5, 0)
        self.settings = QTabWidget()
        left_layout.addWidget(self.settings)
        self._campaign_tab()
        self._time_tab()
        self._calibration_tab()
        self._metadata_tab()
        files = QHBoxLayout()
        files.addWidget(button('Save settings…', self.save_settings))
        files.addWidget(button('Load settings…', self.load_settings))
        left_layout.addLayout(files)
        splitter.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(5, 0, 0, 0)
        self.views = QTabWidget()
        right_layout.addWidget(self.views, 1)
        self._live_view()
        self._results_view()
        self._acquisition_panel(right_layout)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1200)
        self.log.setMaximumHeight(115)
        self.log.setPlaceholderText('Instrument actions, saved files, and errors appear here.')
        right_layout.addWidget(self.log)
        splitter.addWidget(right)
        splitter.setSizes([455, 1025])
        self.statusBar().showMessage('Choose simulation to explore the complete workflow without an instrument.')

    def _campaign_tab(self):
        layout = scroll_page(self.settings, 'Campaign')
        box, form = group('Setup & virtual array')
        self.folder = QLineEdit()
        browse = button('…', self.choose_folder)
        path = QHBoxLayout()
        path.addWidget(self.folder, 1)
        path.addWidget(browse)
        form.addRow('Output folder', path)
        self.fields = {}
        for key, label, val, low in [('setup', 'Setup number', 6, 0), ('row', 'Row j (zero based)', 0, 0),
                                      ('nh', 'Horizontal positions Nh', 10, 1), ('nv', 'Vertical rows Nv', 1, 1),
                                      ('va_vert_pos', 'Vertical antenna index', 0, 0)]:
            self.fields[key] = integer(val, low)
            form.addRow(label, self.fields[key])
        for key, label in [('dh', 'Horizontal spacing (mm)'), ('dv', 'Vertical spacing (mm)')]:
            self.fields[key] = number(10, 0.001)
            form.addRow(label, self.fields[key])
        form.addRow(note('Manual positioning, as in the outdoor notebook. No motor movements are issued.'))
        layout.addWidget(box)
        box, form = group('Segmented acquisition')
        self.fields['ifbw'] = number(10000, 1, 1e8, 0)
        self.fields['sweeps'] = integer(10, 1, 10000)
        self.fields['average'] = integer(1, 1, 10000)
        form.addRow('IF bandwidth (Hz)', self.fields['ifbw'])
        form.addRow('Sweeps / repeats', self.fields['sweeps'])
        form.addRow('Averaging count', self.fields['average'])
        form.addRow(note('Average = 1 saves each repeat separately. Average > 1 saves one result after the requested group of sweeps.'))
        powers = QGridLayout()
        self.powers = []
        for i in range(4):
            w = number(0, -100, 30, 1)
            w.setPrefix(f'P{i+1} ')
            self.powers.append(w)
            powers.addWidget(w, i // 2, i % 2)
        form.addRow('Power (dBm)', powers)
        self.sparams = QLineEdit()
        form.addRow('S-parameters in order', self.sparams)
        form.addRow(note('S21, S31, S41 at every position; remaining traces at i = 0 and i = Nh − 1.'))
        self.segments = QTableWidget(0, 3)
        self.segments.setHorizontalHeaderLabels(['Start (GHz)', 'Stop (GHz)', 'Points'])
        self.segments.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.segments.setMinimumHeight(215)
        form.addRow(self.segments)
        segment_buttons = QHBoxLayout()
        segment_buttons.addWidget(button('Add segment', lambda: self.segments.insertRow(self.segments.rowCount())))
        segment_buttons.addWidget(button('Remove selected', lambda: self.segments.removeRow(self.segments.currentRow())))
        form.addRow(segment_buttons)
        layout.addWidget(box)
        layout.addStretch()

    def _time_tab(self):
        layout = scroll_page(self.settings, 'Time view')
        box, form = group('Live sweep & transform')
        for key, label, val, lo, hi, decimals in [
            ('live_start', 'Start (GHz)', 10, 0.000001, 1000, 6),
            ('live_stop', 'Stop (GHz)', 15, 0.000001, 1000, 6),
            ('live_ifbw', 'IF bandwidth (Hz)', 10000, 1, 1e8, 0),
            ('interval', 'Delay between refreshes (s)', 0.5, 0.05, 60, 2)]:
            self.fields[key] = number(val, lo, hi, decimals)
            form.addRow(label, self.fields[key])
        self.fields['live_points'] = integer(10001, 2, 1000001)
        form.addRow('Frequency points', self.fields['live_points'])
        self.window = QComboBox()
        self.window.addItems(['hann', 'hamming', 'rectangle', 'kaiser', 'bohman'])
        form.addRow('Transform window', self.window)
        form.addRow(note('Plots show the VNA display data. Distance uses c·delay (c = 3×10⁸ m/s), matching the driver; it is not divided by two.'))
        layout.addWidget(box)
        box, form = group('Per-trace VNA axes')
        self.axes = QTableWidget(6, 5)
        self.axes.setHorizontalHeaderLabels(['Unit', 'X min', 'X max', 'Y min dB', 'Y max dB'])
        self.axes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.axes.setMinimumHeight(235)
        for i in range(6):
            unit = QComboBox()
            unit.addItems(['m', 'ns'])
            unit.currentTextChanged.connect(lambda value, row=i: self.convert_axis_units(row, value))
            self.axes.setCellWidget(i, 0, unit)
        form.addRow(self.axes)
        self.axis_btn = button('Apply axes to VNA + plots', self.apply_axes, 'primary')
        form.addRow(self.axis_btn)
        form.addRow(note('Apply sends set_Xscale / set_Yscale through the Python VNA module. Plot zoom and pan are local; use this table to change the instrument display.'))
        layout.addWidget(box)
        layout.addStretch()

    def _calibration_tab(self):
        layout = scroll_page(self.settings, 'Calibration')
        self.cal_prepare = button('Prepare segmented calibration', lambda: self.apply_mode('calibration'), 'primary')
        layout.addWidget(self.cal_prepare)
        layout.addWidget(note('Uses pseudo_preset to preserve stored calibration. No full preset is sent. A new frequency grid, power, IFBW or trace list requires a new confirmation.'))
        box, form = group('Guided SOL · repeat for each required port')
        self.cal_port = integer(1, 1, 4)
        self.connector = QComboBox()
        self.connector.addItems(['2.92', '1.85'])
        self.gender = QComboBox()
        self.gender.addItems(['f', 'm'])
        self.kit = QComboBox()
        self.kit.addItems(['85561A Databased', 'ZV-Z129', '85058E Databased'])
        for label, w in [('Port', self.cal_port), ('Connector (mm)', self.connector), ('Gender', self.gender), ('Calibration kit', self.kit)]:
            form.addRow(label, w)
        self.cal_begin = button('Initialize SOL for this port', self.begin_calibration)
        form.addRow(self.cal_begin)
        self.cal_step = QComboBox()
        self.cal_step.setMinimumContentsLength(12)
        self.cal_step.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addRow('Instrument step', self.cal_step)
        self.cal_desc = note('Initialize SOL to retrieve the required standards from the instrument.')
        form.addRow(self.cal_desc)
        self.cal_step.currentTextChanged.connect(self.cal_desc.setText)
        self.cal_measure = button('Acquire connected standard', lambda: self.send('measure_standard', step=self.cal_step.currentIndex() + 1))
        form.addRow(self.cal_measure)
        self.cal_name = QLineEdit('calset1')
        form.addRow('Instrument calset name', self.cal_name)
        self.cal_apply = button('Save SOL to calset', lambda: self.send('apply_calibration', name=self.cal_name.text()))
        self.cal_abort = button('Abort SOL', lambda: self.send('abort_calibration'))
        form.addRow(self.cal_apply, self.cal_abort)
        layout.addWidget(box)
        box, form = group('Verify calibration before recording')
        form.addRow(note('On the VNA, verify the active calset and coverage for every required port and segment. Saving a one-port SOL does not establish a full multiport calibration.'))
        self.cal_note = QLineEdit()
        self.cal_note.setPlaceholderText('Calset identifier / operator verification note')
        self.cal_ack = QCheckBox('I verified the active calibration and its coverage')
        form.addRow(self.cal_note)
        form.addRow(self.cal_ack)
        self.cal_confirm = button('Confirm current calibration', self.confirm_calibration)
        form.addRow(self.cal_confirm)
        layout.addWidget(box)
        box, form = group('Back-to-back through references')
        form.addRow(note('Restore segmented mode, connect each through path, then capture it separately. References are saved, not automatically divided into the channel data.'))
        row = QHBoxLayout()
        self.b2b_buttons = []
        for i, name in enumerate(('S21', 'S31', 'S41'), 1):
            b = button('Capture ' + name, lambda checked=False, trace=i: self.send('b2b', trace=trace))
            self.b2b_buttons.append(b)
            row.addWidget(b)
        form.addRow(row)
        layout.addWidget(box)
        layout.addStretch()

    def _metadata_tab(self):
        layout = scroll_page(self.settings, 'Metadata')
        layout.addWidget(note('Antenna metadata is saved with every row. Array dimensions and spacing for BS come from the Campaign tab. Coordinate units follow your campaign convention.'))
        self.antenna_fields = {}
        for key, label in [('bs_params', 'BS / virtual array'), ('ue0_params', 'UE 0'), ('ue1_params', 'UE 1'), ('ue2_params', 'UE 2')]:
            box, form = group(label)
            fields = {}
            for field, name in [('ant_name', 'Antenna name'), ('ant_type', 'Antenna type')]:
                fields[field] = QLineEdit()
                form.addRow(name, fields[field])
            for field, name, size in [('ant_coord', 'Coordinates [1, 2]', 2), ('ant_orientation', 'Orientation [θ, φ] (°)', 2)]:
                h = QHBoxLayout()
                fields[field] = [number(0, -1e9, 1e9, 5) for _ in range(size)]
                for w in fields[field]:
                    h.addWidget(w)
                form.addRow(name, h)
            fields['ant_height'] = number(0, -10000, 100000, 4)
            form.addRow('Height (m)', fields['ant_height'])
            for field, name in [('v_pol_vector', 'V polarization (JSON)'), ('h_pol_vector', 'H polarization (JSON)')]:
                fields[field] = QLineEdit()
                form.addRow(name, fields[field])
            self.antenna_fields[key] = fields
            layout.addWidget(box)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText('Scene, target positions, cable changes, weather, operator…')
        self.notes.setMinimumHeight(90)
        layout.addWidget(self.notes)
        layout.addStretch()

    def _live_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.frame_label = note('Live channel · connect and select an inspection mode')
        layout.addWidget(self.frame_label)
        self.graphs = pg.GraphicsLayoutWidget()
        self.graphs.setBackground('#102333')
        self.plots, self.curves = [], []
        colors = ['#55e2bd', '#62baff', '#f9c56b', '#c29aff', '#fc99b1', '#84d4df']
        for i in range(6):
            plot = self.graphs.addPlot(row=i // 3, col=i % 3, title=self.config['sparams'][i])
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel('left', 'Magnitude', units='dB')
            plot.setLabel('bottom', 'Path length', units='m')
            plot.getAxis('bottom').enableAutoSIPrefix(False)
            curve = plot.plot(pen=pg.mkPen(colors[i], width=1.5))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method='peak')
            self.plots.append(plot)
            self.curves.append(curve)
        layout.addWidget(self.graphs, 1)
        self.views.addTab(page, 'Live channel')

    def _results_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(button('Open saved row…', self.open_result))
        self.result_point, self.result_sweep = integer(0), integer(0)
        self.result_trace = QComboBox()
        self.result_trace.addItems(self.config['sparams'])
        for label, w in [('Point', self.result_point), ('Repeat', self.result_sweep), ('Trace', self.result_trace)]:
            controls.addWidget(QLabel(label))
            controls.addWidget(w)
        self.result_point.valueChanged.connect(self.draw_result)
        self.result_sweep.valueChanged.connect(self.draw_result)
        self.result_trace.currentIndexChanged.connect(self.draw_result)
        layout.addLayout(controls)
        self.result_label = note('Open a completed row or partial checkpoint. Missing endpoint-only traces are never plotted as measurements.')
        layout.addWidget(self.result_label)
        self.result_plot = pg.PlotWidget(background='#102333')
        self.result_plot.setLabel('left', '|S|', units='dB')
        self.result_plot.setLabel('bottom', 'Frequency', units='GHz')
        self.result_plot.showGrid(x=True, y=True, alpha=0.15)
        self.result_curve = self.result_plot.plot(pen=pg.mkPen('#55e2bd', width=1.5))
        layout.addWidget(self.result_plot, 2)
        layout.addWidget(note('Measurement evolution · mean linear |S| converted to dB at each measured position (selected trace and repeat)'))
        self.evolution_plot = pg.PlotWidget(background='#102333')
        self.evolution_plot.setMinimumHeight(150)
        self.evolution_plot.getAxis('left').setWidth(95)
        self.evolution_plot.setLabel('left', 'Mean |S|', units='dB')
        self.evolution_plot.setLabel('bottom', 'Horizontal position', units='mm')
        self.evolution_curve = self.evolution_plot.plot(pen=pg.mkPen('#62baff', width=2), symbol='o')
        layout.addWidget(self.evolution_plot, 1)
        self.views.addTab(page, 'Saved results & evolution')

    def _acquisition_panel(self, layout):
        box = QGroupBox('Record calibrated segmented measurements')
        form = QVBoxLayout(box)
        row = QHBoxLayout()
        self.start_row_btn = button('Open row', self.start_row)
        self.resume_btn = button('Resume checkpoint…', self.resume_row)
        self.close_row_btn = button('Close row', lambda: self.send('close_row'))
        self.point = integer(0, 0, 9)
        self.capture_btn = button('3  Acquire this position', self.acquire, 'primary')
        for w in (self.start_row_btn, self.resume_btn, self.close_row_btn, QLabel('i ='), self.point, self.capture_btn):
            row.addWidget(w)
        form.addLayout(row)
        self.position_ack = QCheckBox('Antenna is at the selected grid position and the scene is ready')
        form.addWidget(self.position_ack)
        self.position_ack.toggled.connect(self.refresh_controls)
        self.point.valueChanged.connect(lambda: self.position_ack.setChecked(False))
        self.points_table = QTableWidget(1, 10)
        self.points_table.setMaximumHeight(65)
        self.points_table.verticalHeader().hide()
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.points_table.cellClicked.connect(lambda row, col: self.point.setValue(col))
        form.addWidget(self.points_table)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        form.addWidget(self.progress)
        self.saved_label = note('Each completed sweep is checkpointed. Stop takes effect after the current instrument call finishes.')
        self.saved_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addWidget(self.saved_label)
        layout.addWidget(box)

    def read_axes(self):
        axes = []
        for i in range(6):
            mode = 'dist' if self.axes.cellWidget(i, 0).currentText() == 'm' else 'time'
            vals = [float(self.axes.item(i, j).text()) for j in range(1, 5)]
            factor = 1 if mode == 'dist' else 1e-9
            axes.append(dict(mode=mode, xmin=vals[0] * factor, xmax=vals[1] * factor, ymin=vals[2], ymax=vals[3]))
        return axes

    def read_form(self):
        c = deepcopy(self.config)
        c.update({key: widget.value() for key, widget in self.fields.items()})
        c['live_start'] *= 1e9
        c['live_stop'] *= 1e9
        c.update(folder=self.folder.text().strip(), powers=[w.value() for w in self.powers],
                 sparams=[x.strip().upper() for x in self.sparams.text().split(',')], window=self.window.currentText(), axes=self.read_axes())
        c['segments'] = []
        for i in range(self.segments.rowCount()):
            values = [self.segments.item(i, j).text() if self.segments.item(i, j) else '' for j in range(3)]
            c['segments'].append([float(values[0]) * 1e9, float(values[1]) * 1e9, int(values[2])])
        for key, fields in self.antenna_fields.items():
            m = c['metadata'][key]
            for field, widget in fields.items():
                if isinstance(widget, list):
                    m[field] = [w.value() for w in widget]
                elif isinstance(widget, QDoubleSpinBox):
                    m[field] = widget.value()
                elif field.endswith('pol_vector'):
                    value = json.loads(widget.text())
                    if value is not None and (not isinstance(value, list) or len(value) != 3 or not all(isinstance(x, (float, int)) for x in value)):
                        raise ValueError('Polarization must be null or a three-number JSON array.')
                    m[field] = value
                else:
                    m[field] = widget.text()
        c['metadata']['notes'] = self.notes.toPlainText()
        return validate(c)

    def load_form(self, c):
        self.config = deepcopy(c)
        for key, w in self.fields.items():
            w.setValue(c[key] / 1e9 if key in ('live_start', 'live_stop') else c[key])
        self.folder.setText(c['folder'])
        for w, v in zip(self.powers, c['powers']):
            w.setValue(v)
        self.sparams.setText(', '.join(c['sparams']))
        self.window.setCurrentText(c['window'])
        self.segments.setRowCount(len(c['segments']))
        for i, (start, stop, points) in enumerate(c['segments']):
            for j, value in enumerate((f'{start/1e9:.8f}', f'{stop/1e9:.8f}', str(points))):
                self.segments.setItem(i, j, QTableWidgetItem(value))
        self.axes.setVerticalHeaderLabels((c['sparams'][:6] + ['Unused'] * 6)[:6])
        for i, a in enumerate(c['axes']):
            w = self.axes.cellWidget(i, 0)
            w.blockSignals(True)
            w.setCurrentText('m' if a['mode'] == 'dist' else 'ns')
            w.blockSignals(False)
            factor = 1 if a['mode'] == 'dist' else 1e9
            for j, v in enumerate((a['xmin'] * factor, a['xmax'] * factor, a['ymin'], a['ymax']), 1):
                self.axes.setItem(i, j, QTableWidgetItem(f'{v:.8g}'))
        for key, fields in self.antenna_fields.items():
            for field, widget in fields.items():
                v = c['metadata'][key][field]
                if isinstance(widget, list):
                    for w, value in zip(widget, v):
                        w.setValue(value)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(v)
                else:
                    widget.setText(json.dumps(v) if field.endswith('pol_vector') else str(v))
        self.notes.setPlainText(c['metadata'].get('notes', ''))
        self.point.setMaximum(c['nh'] - 1)
        self.update_points([])

    def convert_axis_units(self, row, unit):
        factor = 1e9 / 3e8 if unit == 'ns' else 3e8 / 1e9
        for col in (1, 2):
            item = self.axes.item(row, col)
            if item:
                try:
                    item.setText(f'{float(item.text()) * factor:.8g}')
                except ValueError:
                    pass

    def send(self, command, **kwargs):
        if self.busy or self.closing:
            return
        self.error_label.hide()
        self.busy = True
        self.refresh_controls()
        self.statusBar().showMessage(command.replace('_', ' ').capitalize() + '…')
        self.worker.submit(command, **kwargs)

    def connect_device(self):
        self.send('connect', simulation=self.device_type.currentIndex() == 0, address=self.address.text())

    def apply_mode(self, mode):
        try:
            c = self.read_form()
            self.config = deepcopy(c)
            self.send('configure', config=c, mode=mode)
        except Exception as exc:
            self.show_error(str(exc))

    def apply_axes(self):
        try:
            self.send('axes', axes=self.read_axes())
        except Exception as exc:
            self.show_error(str(exc))

    def toggle_live(self):
        self.send('set_live', enabled=not self.state['live'])

    def stop(self):
        self.worker.stop_acquisition()
        self.statusBar().showMessage('Stop requested; waiting for the current instrument call. Completed sweeps are retained.')
        if not self.busy:
            self.send('set_live', enabled=False)

    def autoscale(self):
        for p in self.plots:
            p.enableAutoRange()
        self.send('autoscale')

    def begin_calibration(self):
        self.cal_ack.setChecked(False)
        self.send('begin_calibration', port=self.cal_port.value(), connector=self.connector.currentText(),
                  gender=self.gender.currentText(), kit=self.kit.currentText())

    def confirm_calibration(self):
        if not self.cal_ack.isChecked():
            self.show_error('Verify the active calibration on the VNA and check the confirmation box.')
            return
        self.send('confirm_calibration', note=self.cal_note.text())

    def start_row(self):
        try:
            self.send('start_row', config=self.read_form())
        except Exception as exc:
            self.show_error(str(exc))

    def acquire(self):
        if not self.position_ack.isChecked():
            return
        try:
            from .config import row_signature
            if row_signature(self.read_form()) != row_signature(self.state['config']):
                raise ValueError('Edited campaign fields are not applied. Restore the row settings before acquiring.')
            self.position_ack.setChecked(False)
            self.send('acquire_point', point=self.point.value())
        except Exception as exc:
            self.show_error(str(exc))

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Campaign output folder', self.folder.text())
        if path:
            self.folder.setText(path)

    def save_settings(self):
        try:
            c = self.read_form()
            path, _ = QFileDialog.getSaveFileName(self, 'Save campaign settings', 'data/campaign.json', 'JSON (*.json)')
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(c, indent=2, allow_nan=False), encoding='utf-8')
        except Exception as exc:
            self.show_error(str(exc))

    def load_settings(self):
        if self.busy or self.state['row_active']:
            self.show_error('Close the active row and wait for the current command before loading settings.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load campaign settings', 'data', 'JSON (*.json)')
        if path:
            try:
                self.load_form(validate(json.loads(Path(path).read_text(encoding='utf-8'))))
                self.statusBar().showMessage('Settings loaded into the form. Apply a mode to configure the instrument.')
            except Exception as exc:
                self.show_error(str(exc))

    def resume_row(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Resume row checkpoint', self.folder.text(), 'Partial rows (*.partial.npz)')
        if path:
            self.send('resume_row', path=path)

    def open_result(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open saved row', self.folder.text(), 'Rows (*.npz)')
        if path:
            self.load_result(path)

    def load_result(self, path):
        try:
            result = read_result(path)
            self.result = None
            self.result_point.setMaximum(result['data'].shape[0] - 1)
            self.result_sweep.setMaximum(result['data'].shape[1] - 1)
            self.result_trace.clear()
            self.result_trace.addItems(result['config']['sparams'])
            self.result = result
            self.result_path = str(path)
            self.draw_result()
        except Exception as exc:
            self.show_error('Cannot read results: ' + str(exc))

    def draw_result(self):
        if not self.result:
            return
        r = self.result
        i, sweep, trace = self.result_point.value(), self.result_sweep.value(), self.result_trace.currentIndex()
        if trace < 0:
            return
        prefix = 'SIMULATION · ' if r['simulated'] else 'MEASUREMENT · '
        if r['mask'][i, sweep, trace]:
            x = r['freq'] / 1e9
            y = 20 * np.log10(np.maximum(np.abs(r['data'][i, sweep, trace]), 1e-20))
            x, y = self.segment_gaps(x, y, r['config'])
            self.result_curve.setData(x, y, connect='finite')
            self.result_label.setText(prefix + self.result_path)
        else:
            self.result_curve.setData([], [])
            self.result_label.setText(prefix + 'Not measured at this position / repeat. Endpoint-only traces are absent at interior positions.')
        valid = r['mask'][:, sweep, trace]
        means = np.full(len(valid), np.nan)
        means[valid] = 20 * np.log10(np.maximum(np.abs(r['data'][valid, sweep, trace]).mean(axis=1), 1e-20))
        self.evolution_curve.setData(np.arange(len(valid)) * r['config']['dh'], means, connect='finite')

    @staticmethod
    def segment_gaps(x, y, config):
        lengths = [int(s[2]) for s in config['segments']]
        if len(x) == sum(lengths):
            indices = np.cumsum(lengths)[:-1]
            return np.insert(x, indices, np.nan), np.insert(y, indices, np.nan)
        return x, y

    def draw_frame(self, frame):
        self.last_frame = frame
        mode = frame['mode']
        label = 'VNA time-domain display' if mode == 'time' else 'Segmented frequency response'
        self.frame_label.setText(('SIMULATION  •  ' if frame['simulated'] else 'INSTRUMENT  •  ') + label + '  •  ' + frame['time'])
        for i, (plot, curve) in enumerate(zip(self.plots, self.curves)):
            if i >= len(frame['traces']):
                curve.setData([], [])
                plot.setTitle('Not active')
                continue
            t = frame['traces'][i]
            plot.setTitle(t['name'])
            plot.setLabel('bottom', 'Path length' if t['unit'] == 'm' else ('Delay' if t['unit'] == 'ns' else 'Frequency'), units=t['unit'])
            x, y = t['x'], t['y']
            if mode != 'time':
                x, y = self.segment_gaps(x, y, self.state.get('config') or self.config)
            curve.setData(x, y, connect='finite')

    def update_points(self, points):
        count = len(points) or self.config['nh']
        self.points_table.setColumnCount(count)
        self.points_table.setHorizontalHeaderLabels([str(i) for i in range(count)])
        for i in range(count):
            done = bool(points and points[i])
            item = QTableWidgetItem('✓' if done else '·')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setBackground(QColor('#bcecdc' if done else '#f1f5f8'))
            self.points_table.setItem(0, i, item)
        self.point.setMaximum(count - 1)

    @Slot(str, object)
    def on_event(self, kind, payload):
        if kind == 'frame':
            self.draw_frame(payload)
        elif kind == 'log':
            self.log.appendPlainText(payload)
        elif kind == 'error':
            self.show_error(payload)
        elif kind == 'cal_steps':
            self.cal_step.clear()
            self.cal_step.addItems([f'{i}: {text}' for i, text in enumerate(payload, 1)])
        elif kind == 'loaded_config':
            self.load_form(payload)
        elif kind == 'progress':
            self.progress.setRange(0, payload['total'])
            self.progress.setValue(payload['sweep'])
            self.progress.setFormat(f"i = {payload['point']} · sweep %v / %m")
            self.update_points(payload['points'])
        elif kind in ('done', 'status'):
            command = payload.get('command') if kind == 'done' else None
            self.state = payload['status'] if kind == 'done' else payload
            if kind == 'done':
                self.busy = False
            self.identity.setText(self.state['identity'])
            mode = self.state['mode'].upper()
            self.badge.setText(('SIMULATION  /  ' if self.state['connected'] and self.state['simulated'] else '') + mode)
            self.update_points(self.state['points'])
            if self.state['saved']:
                self.saved_label.setText(self.state['saved'])
            if command == 'acquire_point' and self.state['saved']:
                self.load_result(self.state['saved'])
                missing = [i for i, done in enumerate(self.state['points']) if not done]
                if missing:
                    self.point.setValue(missing[0])
            if command in ('configure', 'axes') and self.state['mode'] == 'time':
                c = self.state['config']
                for plot, a in zip(self.plots, c['axes']):
                    f = 1 if a['mode'] == 'dist' else 1e9
                    plot.setXRange(a['xmin'] * f, a['xmax'] * f, padding=0)
                    plot.setYRange(a['ymin'], a['ymax'], padding=0)
            elif command == 'configure':
                for plot in self.plots:
                    plot.enableAutoRange()
            if command in ('configure', 'begin_calibration'):
                self.cal_ack.setChecked(False)
            self.refresh_controls()
            self.statusBar().showMessage('Ready · ' + ('calibration confirmed for applied segmented settings' if self.state['calibrated'] else 'segmented calibration not yet confirmed'))
            if self.demo and not self.error_label.isVisible():
                if command == 'connect' and self.demo_stage == 0:
                    self.demo_stage = 1
                    self.apply_mode('time')
                elif command == 'configure' and self.demo_stage == 1:
                    self.demo_stage = 2
                    self.send('set_live', enabled=True)

    def refresh_controls(self, *args):
        s = self.state
        ready = not self.busy and not self.closing
        connected = s['connected'] and ready
        self.connect_btn.setEnabled(ready and not s['connected'])
        self.device_type.setEnabled(ready and not s['connected'])
        self.address.setEnabled(ready and not s['connected'])
        self.disconnect_btn.setEnabled(connected)
        self.settings.setEnabled(ready)
        for b in (self.time_btn, self.segment_btn, self.cal_prepare):
            b.setEnabled(connected)
        self.live_btn.setEnabled(connected and s['mode'] in ('time', 'segmented', 'calibration'))
        self.live_btn.setText('Pause live view' if s['live'] else 'Start live view')
        self.auto_btn.setEnabled(connected and s['mode'] in ('time', 'segmented', 'calibration'))
        self.axis_btn.setEnabled(connected and s['mode'] == 'time')
        self.stop_btn.setEnabled(s['connected'])
        cal = connected and s['mode'] == 'calibration'
        self.cal_begin.setEnabled(cal and not s['row_active'] and self.cal_step.count() == 0)
        self.cal_measure.setEnabled(cal and self.cal_step.count() > 0)
        self.cal_apply.setEnabled(cal and self.cal_step.count() > 0)
        self.cal_abort.setEnabled(cal and self.cal_step.count() > 0)
        self.cal_confirm.setEnabled(cal and self.cal_step.count() == 0)
        segmented = connected and s['mode'] == 'segmented' and s['calibrated']
        self.start_row_btn.setEnabled(segmented and not s['row_active'])
        self.resume_btn.setEnabled(ready and not s['row_active'])
        self.close_row_btn.setEnabled(ready and s['row_active'])
        self.capture_btn.setEnabled(segmented and s['row_active'] and self.position_ack.isChecked())
        self.point.setEnabled(ready)
        self.position_ack.setEnabled(ready and s['row_active'])
        for b in self.b2b_buttons:
            b.setEnabled(segmented)

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()
        self.log.appendPlainText('ERROR  ' + message)

    def closeEvent(self, event):
        if self.worker.isRunning():
            event.ignore()
            self.closing = True
            self.worker.request_shutdown()
            self.refresh_controls()
            self.statusBar().showMessage('Closing after the current instrument call; acquired sweeps are already saved.')
        else:
            event.accept()

    @Slot()
    def on_worker_finished(self):
        if self.closing:
            self.close()
