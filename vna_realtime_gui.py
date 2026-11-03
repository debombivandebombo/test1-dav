import sys
import os
import time
import queue
from datetime import datetime
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QGroupBox,
        QStatusBar, QMessageBox, QComboBox, QCheckBox, QLineEdit
    )
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QGroupBox,
        QStatusBar, QMessageBox, QComboBox, QCheckBox, QLineEdit
    )
    from PySide6.QtCore import QThread, Signal as pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mcg_lab_vna_control import connect_vna
from ris_beamforming_controller import RISConfig, RISServerController, RISCoordinateController


# ================= PARÁMETROS DEL VNA =================
FOLDER_NAME = "data/VA_static_ul_dl"
SPARAM_LIST = ['S21', 'S31', 'S41', 'S12', 'S13', 'S14']  # 3 Downlink + 3 Uplink
TRACE_LIST = [1, 2, 3, 4, 5, 6]

F_START = 15.0e9          # 15 GHz
F_STOP = 30.0e9           # 30 GHz
POINTS = 3001             # Puntos de frecuencia
IFBW = 10e3               # 10 kHz
POW_DBM_LIST = [-15, 0, 0, 0]  # Potencia en puertos [1, 2, 3, 4] en dBm
C_LIGHT = 299792458.0     # Velocidad de la luz (m/s)

# ================= PARÁMETROS RIS =====================
RIS_IP = "192.168.100.3"
RIS_PORT = 17000
# ======================================================


class VNAWorker(QThread):
    """Hilo en segundo plano para medir continuamente sin bloquear la interfaz gráfica."""
    data_ready = pyqtSignal(np.ndarray, np.ndarray, str)  # (freqs_ghz, sparams_complex, timestamp)
    status_msg = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, demo_mode: bool = False):
        super().__init__()
        self.running = False
        self.demo_mode = demo_mode
        self.vna = None
        self.freqs = None
        self.cmd_queue: queue.Queue = queue.Queue()

    def enqueue_command(self, cmd_name: str, **kwargs):
        """Encola un comando para ejecutarse de forma segura en el hilo del VNA."""
        self.cmd_queue.put((cmd_name, kwargs))

    def _process_pending_commands(self):
        while not self.cmd_queue.empty():
            try:
                cmd_name, kwargs = self.cmd_queue.get_nowait()
            except queue.Empty:
                break

            if self.demo_mode or self.vna is None:
                continue

            try:
                if cmd_name == "autoscale":
                    self.vna.auto_scale_all()
                    self.status_msg.emit("Autoscale aplicado en la pantalla del VNA.")
                elif cmd_name == "set_vna_timedomain":
                    enabled = kwargs.get("enabled", False)
                    window = kwargs.get("window", "hann")
                    if enabled:
                        for tr in TRACE_LIST:
                            try:
                                self.vna.set_timedomain(trace=tr, window=window)
                            except TypeError:
                                self.vna.set_timedomain(trace=tr)
                        self.vna.auto_scale_all()
                        self.status_msg.emit("Time Domain activado en la pantalla del VNA.")
                    else:
                        self.vna.pseudo_preset()
                        self.vna.configure_multiple_sparams(
                            sparam_list=SPARAM_LIST,
                            f_start=F_START,
                            f_stop=F_STOP,
                            points=POINTS,
                            ifbw=IFBW,
                            pow_dbm_list=POW_DBM_LIST
                        )
                        self.vna.calibration_on()
                        self.vna.set_sweeps(sweeps=1)
                        self.vna.unset_average()
                        self.vna.auto_scale_all()
                        self.status_msg.emit("Dominio de frecuencia restaurado en el VNA.")
                elif cmd_name == "set_vna_xscale":
                    xmin = kwargs.get("xmin", 0.0)
                    xmax = kwargs.get("xmax", 10.0)
                    mode = kwargs.get("mode", "dist")
                    for tr in TRACE_LIST:
                        self.vna.set_Xscale(xmin=xmin, xmax=xmax, mode=mode, trace=tr)
            except Exception as e:
                self.status_msg.emit(f"Aviso comando VNA ({cmd_name}): {e}")

    def run(self):
        if self.demo_mode:
            self._run_demo()
            return

        try:
            self.status_msg.emit("Conectando al VNA Keysight...")
            self.vna = connect_vna(brand='keysight')
            self.vna.pseudo_preset()

            self.status_msg.emit("Configurando parámetros S (Downlink y Uplink, 15-30 GHz)...")
            self.vna.configure_multiple_sparams(
                sparam_list=SPARAM_LIST,
                f_start=F_START,
                f_stop=F_STOP,
                points=POINTS,
                ifbw=IFBW,
                pow_dbm_list=POW_DBM_LIST
            )
            self.vna.calibration_on()
            self.vna.set_sweeps(sweeps=1)
            self.vna.unset_average()

            self.freqs = self.vna.get_freq_vector()
            freqs_ghz = self.freqs / 1e9
            self.vna.auto_scale_all()

            self.running = True
            self.status_msg.emit("Monitorizando en tiempo real (Sin grabar)...")

            while self.running:
                self._process_pending_commands()
                # Mide las 6 trazas simultáneamente: shape (6, POINTS)
                sparams = self.vna.measure_multiple_traces_sparams(trace_list=TRACE_LIST)
                ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                self.data_ready.emit(freqs_ghz, sparams, ts)
                time.sleep(0.02)

            self.vna.run_cont()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_demo(self):
        """Modo demostración sintético para pruebas sin el VNA físico conectado."""
        self.freqs = np.linspace(F_START, F_STOP, POINTS)
        freqs_ghz = self.freqs / 1e9
        rng = np.random.default_rng(42)
        self.running = True
        self.status_msg.emit("Modo DEMO activo: Generando medidas sintéticas en tiempo real...")
        tick = 0
        while self.running:
            self._process_pending_commands()
            tick += 1
            sparams = np.zeros((6, POINTS), dtype=np.complex64)
            for i in range(6):
                # Componente multicamino sintética (ej. trayecto RIS a ~2.5 m y eco a ~4.2 m)
                tau1 = (2.5 + 0.15 * i) / C_LIGHT
                tau2 = (4.2 + 0.20 * i) / C_LIGHT
                amp1 = 10 ** ((-42.0 - 2.0 * i + 1.5 * np.sin(tick * 0.1 + i)) / 20.0)
                amp2 = 10 ** ((-58.0 - 1.5 * i) / 20.0)
                noise = (rng.normal(0, 1e-4, POINTS) + 1j * rng.normal(0, 1e-4, POINTS))
                sparams[i, :] = (
                    amp1 * np.exp(-1j * 2 * np.pi * self.freqs * tau1)
                    + amp2 * np.exp(-1j * 2 * np.pi * self.freqs * tau2)
                    + noise
                )
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            self.data_ready.emit(freqs_ghz, sparams, ts)
            time.sleep(0.08)

    def stop(self):
        self.running = False
        self.wait()


class VNARealtimeWindow(QMainWindow):
    def __init__(self, demo_mode: bool = False):
        super().__init__()
        self.demo_mode = demo_mode
        self.setWindowTitle("Monitor en Tiempo Real VNA (15 - 30 GHz) | Downlink, Uplink & Control RIS")
        self.resize(1380, 900)

        os.makedirs(FOLDER_NAME, exist_ok=True)

        # Estado de grabación
        self.is_recording = False
        self.record_start_time = 0.0
        self.record_duration = 5.0
        self.recorded_sparams = []
        self.recorded_timestamps = []
        self.current_freqs_hz = None
        self.last_freqs_ghz = None
        self.last_sparams = None
        self.last_x_vector = None
        self.last_mag_db = None

        # Estado de dominio y escalas X por modo
        # Modos: "freq_ghz", "td_ns", "td_m"
        self.current_domain = "freq_ghz"
        self.domain_x_limits = {
            "freq_ghz": [F_START / 1e9, F_STOP / 1e9],
            "td_ns": [0.0, 40.0],
            "td_m": [0.0, 12.0],
        }
        self._updating_x_controls = False

        # Controlador del servidor de la RIS (RISServerController + RISConfig) y calculador de coordenadas
        self.ris_server = RISServerController(ip=RIS_IP, port=RIS_PORT)
        self.ris_connected = False
        self.ris_controller = RISCoordinateController(
            server_ip=RIS_IP,
            server_port=RIS_PORT,
            freq_hz=28.0e9,
            tx_pos=(0.0, 1.1, 0.0),
            boresight="y",
            polarization_mode="vh",
        )
        self.last_ris_info = None

        self._build_ui()

        # Iniciar el hilo del VNA
        self.worker = VNAWorker(demo_mode=self.demo_mode)
        self.worker.data_ready.connect(self.update_measurement)
        self.worker.status_msg.connect(self.statusBar().showMessage)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # =====================================================================
        # FILA 1: GRABACIÓN, DOMINIO (FRECUENCIA / TIME DOMAIN) Y ESCALAS X / Y
        # =====================================================================
        top_controls_layout = QHBoxLayout()

        # 1. Grupo de Grabación Bajo Demanda
        record_group = QGroupBox("Control de Grabación Bajo Demanda")
        record_layout = QHBoxLayout(record_group)

        lbl_duration = QLabel("Tiempo (s):")
        lbl_duration.setStyleSheet("font-weight: bold;")

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.5, 3600.0)
        self.spin_duration.setValue(10.0)
        self.spin_duration.setSingleStep(1.0)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setMinimumWidth(80)

        self.btn_record = QPushButton("🔴 Iniciar Grabación y Guardar")
        self.btn_record.setMinimumHeight(32)
        self.btn_record.setStyleSheet(
            "font-weight: bold; background-color: #d9534f; color: white; border-radius: 5px; padding: 4px 10px;"
        )
        self.btn_record.clicked.connect(self.start_recording)

        self.lbl_rec_status = QLabel("Estado: Monitorizando")
        self.lbl_rec_status.setStyleSheet("color: #555555; margin-left: 6px;")

        record_layout.addWidget(lbl_duration)
        record_layout.addWidget(self.spin_duration)
        record_layout.addWidget(self.btn_record)
        record_layout.addWidget(self.lbl_rec_status)

        # 2. Grupo de Dominio (Frecuencia / Time Domain) y Autoscale
        domain_group = QGroupBox("Dominio de Visualización y Autoscale")
        domain_layout = QHBoxLayout(domain_group)

        self.combo_domain = QComboBox()
        self.combo_domain.addItem("Frecuencia (GHz)", "freq_ghz")
        self.combo_domain.addItem("Time Domain - Tiempo (ns)", "td_ns")
        self.combo_domain.addItem("Time Domain - Distancia (m)", "td_m")
        self.combo_domain.currentIndexChanged.connect(self.on_domain_changed)

        self.combo_window = QComboBox()
        self.combo_window.addItem("Ventana: Hann", "hann")
        self.combo_window.addItem("Ventana: Rectangular", "rect")
        self.combo_window.addItem("Ventana: Hamming", "hamming")
        self.combo_window.addItem("Ventana: Blackman", "blackman")
        self.combo_window.setToolTip("Ventana aplicada antes de la IFFT en modo Time Domain")
        self.combo_window.currentIndexChanged.connect(self.refresh_current_plot)

        self.chk_sync_vna_td = QCheckBox("TD en VNA")
        self.chk_sync_vna_td.setToolTip(
            "Si se activa, conmuta también la pantalla física del VNA a modo Time Domain."
        )
        self.chk_sync_vna_td.toggled.connect(self.on_sync_vna_td_toggled)

        self.btn_autoscale = QPushButton("🔍 Autoscale Y")
        self.btn_autoscale.setMinimumHeight(30)
        self.btn_autoscale.setStyleSheet(
            "font-weight: bold; background-color: #0275d8; color: white; border-radius: 4px; padding: 3px 10px;"
        )
        self.btn_autoscale.setToolTip("Ajusta automáticamente los ejes Y de Downlink y Uplink a las trazas actuales")
        self.btn_autoscale.clicked.connect(self.autoscale_plots)

        self.chk_auto_y = QCheckBox("Auto-Y continuo")
        self.chk_auto_y.setToolTip("Ajusta automáticamente la escala Y en cada barrido")

        self.btn_vna_autoscale = QPushButton("📺 Autoscale VNA")
        self.btn_vna_autoscale.setMinimumHeight(30)
        self.btn_vna_autoscale.setToolTip("Envía auto_scale_all() a la pantalla del propio VNA")
        self.btn_vna_autoscale.clicked.connect(lambda: self.worker.enqueue_command("autoscale"))

        domain_layout.addWidget(QLabel("Modo:"))
        domain_layout.addWidget(self.combo_domain)
        domain_layout.addWidget(self.combo_window)
        domain_layout.addWidget(self.chk_sync_vna_td)
        domain_layout.addWidget(self.btn_autoscale)
        domain_layout.addWidget(self.chk_auto_y)
        domain_layout.addWidget(self.btn_vna_autoscale)

        top_controls_layout.addWidget(record_group, stretch=2)
        top_controls_layout.addWidget(domain_group, stretch=3)
        main_layout.addLayout(top_controls_layout)

        # =====================================================================
        # FILA 2: CONTROLES DE ESCALA EJE X, EJE Y DOWNLINK Y EJE Y UPLINK
        # =====================================================================
        axes_controls_layout = QHBoxLayout()

        # 1. Grupo de Escala Eje X
        x_scale_group = QGroupBox("Escala Eje X (Frecuencia / Tiempo / Distancia)")
        x_scale_layout = QHBoxLayout(x_scale_group)

        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setDecimals(2)
        self.spin_xmin.setRange(0.0, 1000.0)
        self.spin_xmin.setValue(F_START / 1e9)
        self.spin_xmin.setSingleStep(0.5)
        self.spin_xmin.setSuffix(" GHz")
        self.spin_xmin.valueChanged.connect(self.update_x_limits)

        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setDecimals(2)
        self.spin_xmax.setRange(0.01, 1000.0)
        self.spin_xmax.setValue(F_STOP / 1e9)
        self.spin_xmax.setSingleStep(0.5)
        self.spin_xmax.setSuffix(" GHz")
        self.spin_xmax.valueChanged.connect(self.update_x_limits)

        self.btn_reset_x = QPushButton("↺ Rango X Completo")
        self.btn_reset_x.setToolTip("Restaura el eje X al rango total del modo actual")
        self.btn_reset_x.clicked.connect(self.reset_x_limits_full)

        x_scale_layout.addWidget(QLabel("Mín:"))
        x_scale_layout.addWidget(self.spin_xmin)
        x_scale_layout.addWidget(QLabel("Máx:"))
        x_scale_layout.addWidget(self.spin_xmax)
        x_scale_layout.addWidget(self.btn_reset_x)

        # 2. Grupo de Escala Eje Y - DOWNLINK
        dl_scale_group = QGroupBox("Eje Y Downlink (dB)")
        dl_scale_layout = QHBoxLayout(dl_scale_group)

        self.spin_dl_ymin = QDoubleSpinBox()
        self.spin_dl_ymin.setRange(-220.0, 50.0)
        self.spin_dl_ymin.setValue(-90.0)
        self.spin_dl_ymin.setSingleStep(5.0)
        self.spin_dl_ymin.setSuffix(" dB")
        self.spin_dl_ymin.valueChanged.connect(self.update_y_limits)

        self.spin_dl_ymax = QDoubleSpinBox()
        self.spin_dl_ymax.setRange(-220.0, 50.0)
        self.spin_dl_ymax.setValue(0.0)
        self.spin_dl_ymax.setSingleStep(5.0)
        self.spin_dl_ymax.setSuffix(" dB")
        self.spin_dl_ymax.valueChanged.connect(self.update_y_limits)

        dl_scale_layout.addWidget(QLabel("Mín:"))
        dl_scale_layout.addWidget(self.spin_dl_ymin)
        dl_scale_layout.addWidget(QLabel("Máx:"))
        dl_scale_layout.addWidget(self.spin_dl_ymax)

        # 3. Grupo de Escala Eje Y - UPLINK
        ul_scale_group = QGroupBox("Eje Y Uplink (dB)")
        ul_scale_layout = QHBoxLayout(ul_scale_group)

        self.spin_ul_ymin = QDoubleSpinBox()
        self.spin_ul_ymin.setRange(-220.0, 50.0)
        self.spin_ul_ymin.setValue(-90.0)
        self.spin_ul_ymin.setSingleStep(5.0)
        self.spin_ul_ymin.setSuffix(" dB")
        self.spin_ul_ymin.valueChanged.connect(self.update_y_limits)

        self.spin_ul_ymax = QDoubleSpinBox()
        self.spin_ul_ymax.setRange(-220.0, 50.0)
        self.spin_ul_ymax.setValue(0.0)
        self.spin_ul_ymax.setSingleStep(5.0)
        self.spin_ul_ymax.setSuffix(" dB")
        self.spin_ul_ymax.valueChanged.connect(self.update_y_limits)

        ul_scale_layout.addWidget(QLabel("Mín:"))
        ul_scale_layout.addWidget(self.spin_ul_ymin)
        ul_scale_layout.addWidget(QLabel("Máx:"))
        ul_scale_layout.addWidget(self.spin_ul_ymax)

        axes_controls_layout.addWidget(x_scale_group, stretch=2)
        axes_controls_layout.addWidget(dl_scale_group, stretch=1)
        axes_controls_layout.addWidget(ul_scale_group, stretch=1)
        main_layout.addLayout(axes_controls_layout)

        # =====================================================================
        # PANEL 3: CONTROL DE LA RIS POR COORDENADAS 3D (UDP -> ris_hardware_server.py)
        # =====================================================================
        ris_group = QGroupBox("Control RIS por Coordenadas 3D (192.168.100.3:17000)")
        ris_layout = QHBoxLayout(ris_group)

        # Coordenadas RX [x, y, z]
        self.spin_rx_x = QDoubleSpinBox()
        self.spin_rx_x.setDecimals(3)
        self.spin_rx_x.setRange(-30.0, 30.0)
        self.spin_rx_x.setSingleStep(0.05)
        self.spin_rx_x.setValue(-0.500)
        self.spin_rx_x.setSuffix(" m")
        self.spin_rx_x.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.spin_rx_y = QDoubleSpinBox()
        self.spin_rx_y.setDecimals(3)
        self.spin_rx_y.setRange(-30.0, 30.0)
        self.spin_rx_y.setSingleStep(0.05)
        self.spin_rx_y.setValue(1.400)
        self.spin_rx_y.setSuffix(" m")
        self.spin_rx_y.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.spin_rx_z = QDoubleSpinBox()
        self.spin_rx_z.setDecimals(3)
        self.spin_rx_z.setRange(-30.0, 30.0)
        self.spin_rx_z.setSingleStep(0.05)
        self.spin_rx_z.setValue(0.000)
        self.spin_rx_z.setSuffix(" m")
        self.spin_rx_z.valueChanged.connect(self._on_ris_coord_spin_changed)

        # Coordenadas TX [x, y, z] y Frecuencia RIS
        self.spin_tx_x = QDoubleSpinBox()
        self.spin_tx_x.setDecimals(2)
        self.spin_tx_x.setRange(-30.0, 30.0)
        self.spin_tx_x.setSingleStep(0.1)
        self.spin_tx_x.setValue(0.00)
        self.spin_tx_x.setSuffix(" m")
        self.spin_tx_x.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.spin_tx_y = QDoubleSpinBox()
        self.spin_tx_y.setDecimals(2)
        self.spin_tx_y.setRange(-30.0, 30.0)
        self.spin_tx_y.setSingleStep(0.1)
        self.spin_tx_y.setValue(1.10)
        self.spin_tx_y.setSuffix(" m")
        self.spin_tx_y.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.spin_tx_z = QDoubleSpinBox()
        self.spin_tx_z.setDecimals(2)
        self.spin_tx_z.setRange(-30.0, 30.0)
        self.spin_tx_z.setSingleStep(0.1)
        self.spin_tx_z.setValue(0.00)
        self.spin_tx_z.setSuffix(" m")
        self.spin_tx_z.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.spin_ris_freq = QDoubleSpinBox()
        self.spin_ris_freq.setDecimals(2)
        self.spin_ris_freq.setRange(1.0, 100.0)
        self.spin_ris_freq.setValue(28.00)
        self.spin_ris_freq.setSingleStep(0.5)
        self.spin_ris_freq.setSuffix(" GHz")
        self.spin_ris_freq.valueChanged.connect(self._on_ris_coord_spin_changed)

        self.combo_ris_opt = QComboBox()
        self.combo_ris_opt.addItem("Opt: 1D Search", "1d_search")
        self.combo_ris_opt.addItem("Opt: Iterative", "iterative")
        self.combo_ris_opt.addItem("Opt: Direct Disc", "direct_disc")
        self.combo_ris_opt.addItem("Opt: Chen Alg1", "chen_alg1")
        self.combo_ris_opt.setToolTip("Algoritmo de optimización discreta de surpRISe")
        self.combo_ris_opt.currentIndexChanged.connect(self._on_ris_coord_spin_changed)

        self.btn_apply_ris = QPushButton("📡 Apuntar RIS")
        self.btn_apply_ris.setMinimumHeight(30)
        self.btn_apply_ris.setStyleSheet(
            "font-weight: bold; background-color: #5cb85c; color: white; border-radius: 4px; padding: 3px 8px;"
        )
        self.btn_apply_ris.clicked.connect(self.apply_ris_from_ui)

        self.btn_ris_zero = QPushButton("Todo 0")
        self.btn_ris_zero.setToolTip("Pone state_matrix_v y state_matrix_h de la RIS en estado 0")
        self.btn_ris_zero.clicked.connect(lambda: self.apply_uniform_ris(0))

        self.btn_ris_one = QPushButton("Todo 1")
        self.btn_ris_one.setToolTip("Pone state_matrix_v y state_matrix_h de la RIS en estado 1")
        self.btn_ris_one.clicked.connect(lambda: self.apply_uniform_ris(1))

        self.btn_ris_on = QPushButton("RIS ON")
        self.btn_ris_on.setToolTip("Envía el comando ON (connect) a ris_hardware_server.py")
        self.btn_ris_on.clicked.connect(self.turn_on_ris)

        self.btn_ris_off = QPushButton("RIS OFF")
        self.btn_ris_off.setToolTip("Envía el comando OFF (disconnect) a ris_hardware_server.py")
        self.btn_ris_off.clicked.connect(self.turn_off_ris)

        self.lbl_ris_status = QLabel("RIS: Listo")
        self.lbl_ris_status.setStyleSheet("color: #333333; font-weight: bold; margin-left: 4px;")

        lbl_rx = QLabel("RX [X, Y, Z]:")
        lbl_rx.setStyleSheet("font-weight: bold;")
        ris_layout.addWidget(lbl_rx)
        ris_layout.addWidget(self.spin_rx_x)
        ris_layout.addWidget(self.spin_rx_y)
        ris_layout.addWidget(self.spin_rx_z)

        lbl_tx = QLabel("TX [X, Y, Z]:")
        lbl_tx.setStyleSheet("font-weight: bold; margin-left: 4px;")
        ris_layout.addWidget(lbl_tx)
        ris_layout.addWidget(self.spin_tx_x)
        ris_layout.addWidget(self.spin_tx_y)
        ris_layout.addWidget(self.spin_tx_z)

        ris_layout.addWidget(QLabel("Freq:"))
        ris_layout.addWidget(self.spin_ris_freq)
        ris_layout.addWidget(self.combo_ris_opt)
        ris_layout.addWidget(self.btn_apply_ris)
        ris_layout.addWidget(self.btn_ris_zero)
        ris_layout.addWidget(self.btn_ris_one)
        ris_layout.addWidget(self.btn_ris_on)
        ris_layout.addWidget(self.btn_ris_off)
        ris_layout.addWidget(self.lbl_ris_status)

        main_layout.addWidget(ris_group)

        # =====================================================================
        # LIENZO DE GRÁFICAS (MATPLOTLIB)
        # =====================================================================
        self.fig = Figure(figsize=(10, 6.8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=1)

        # Subplot 1: Downlink (S21, S31, S41)
        self.ax_dl = self.fig.add_subplot(211)
        self.ax_dl.set_title("Downlink: Transmisión Puerto 1 ➔ Puertos 2, 3, 4 (S21, S31, S41)")
        self.ax_dl.set_ylabel("Magnitud (dB)")
        self.ax_dl.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
        self.ax_dl.set_ylim(self.spin_dl_ymin.value(), self.spin_dl_ymax.value())
        self.ax_dl.grid(True, linestyle='--', alpha=0.6)

        # Subplot 2: Uplink (S12, S13, S14)
        self.ax_ul = self.fig.add_subplot(212)
        self.ax_ul.set_title("Uplink: Recepción en Puerto 1 ⬅ Puertos 2, 3, 4 (S12, S13, S14)")
        self.ax_ul.set_xlabel("Frecuencia (GHz)")
        self.ax_ul.set_ylabel("Magnitud (dB)")
        self.ax_ul.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
        self.ax_ul.set_ylim(self.spin_ul_ymin.value(), self.spin_ul_ymax.value())
        self.ax_ul.grid(True, linestyle='--', alpha=0.6)

        # Inicializar las 6 líneas
        colors_dl = ['#1f77b4', '#ff7f0e', '#2ca02c']
        colors_ul = ['#d62728', '#9467bd', '#8c564b']
        self.lines = []

        for idx, sp in enumerate(SPARAM_LIST[:3]):
            line, = self.ax_dl.plot([], [], label=sp, color=colors_dl[idx], lw=1.2)
            self.lines.append(line)
        self.ax_dl.legend(loc="upper right")

        for idx, sp in enumerate(SPARAM_LIST[3:]):
            line, = self.ax_ul.plot([], [], label=sp, color=colors_ul[idx], lw=1.2)
            self.lines.append(line)
        self.ax_ul.legend(loc="upper right")

        self.setStatusBar(QStatusBar())

    # =========================================================================
    # LÓGICA DE DOMINIO (FRECUENCIA / TIME DOMAIN), ESCALAS X/Y Y AUTOSCALE
    # =========================================================================
    def _compute_window(self, n_points: int) -> np.ndarray:
        win_type = self.combo_window.currentData()
        if win_type == "hann":
            w = np.hanning(n_points)
        elif win_type == "hamming":
            w = np.hamming(n_points)
        elif win_type == "blackman":
            w = np.blackman(n_points)
        else:
            w = np.ones(n_points, dtype=float)
        # Normalizamos por la ganancia coherente para mantener la amplitud real de los picos en dB
        mean_val = float(np.mean(w))
        return w / mean_val if mean_val > 0 else w

    def _transform_sparams_for_display(self, freqs_ghz: np.ndarray, sparams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (x_vector, mag_db) según el modo activo:
        - 'freq_ghz': Frecuencia en GHz vs Magnitud S(f) en dB.
        - 'td_ns': Tiempo en ns vs Respuesta al impulso |h(t)| en dB (vía IFFT).
        - 'td_m': Distancia en metros (c * t) vs Respuesta al impulso |h(d)| en dB (vía IFFT).
        """
        if self.current_domain == "freq_ghz":
            mag_db = 20.0 * np.log10(np.abs(sparams) + 1e-12)
            return freqs_ghz, mag_db

        n_pts = len(freqs_ghz)
        if n_pts > 1:
            df_hz = (freqs_ghz[1] - freqs_ghz[0]) * 1e9
        else:
            df_hz = (F_STOP - F_START) / max(POINTS - 1, 1)

        t_max_sec = 1.0 / max(df_hz, 1.0)
        t_sec = np.linspace(0.0, t_max_sec * (1.0 - 1.0 / n_pts), n_pts)

        window = self._compute_window(n_pts)
        impulse_complex = np.fft.ifft(sparams * window[None, :], axis=1)
        mag_db = 20.0 * np.log10(np.abs(impulse_complex) + 1e-12)

        if self.current_domain == "td_ns":
            return t_sec * 1e9, mag_db
        else:
            return t_sec * C_LIGHT, mag_db

    def _get_full_x_range(self, domain_key: str) -> tuple[float, float]:
        if domain_key == "freq_ghz":
            if self.last_freqs_ghz is not None and len(self.last_freqs_ghz) > 1:
                return float(self.last_freqs_ghz[0]), float(self.last_freqs_ghz[-1])
            return F_START / 1e9, F_STOP / 1e9

        n_pts = len(self.last_freqs_ghz) if self.last_freqs_ghz is not None else POINTS
        df_hz = (
            (self.last_freqs_ghz[1] - self.last_freqs_ghz[0]) * 1e9
            if (self.last_freqs_ghz is not None and len(self.last_freqs_ghz) > 1)
            else (F_STOP - F_START) / max(POINTS - 1, 1)
        )
        t_max_sec = (n_pts - 1) / (n_pts * max(df_hz, 1.0))
        if domain_key == "td_ns":
            return 0.0, float(t_max_sec * 1e9)
        return 0.0, float(t_max_sec * C_LIGHT)

    def on_domain_changed(self):
        """Cambia entre Frecuencia (GHz), Time Domain (ns) y Time Domain (m)."""
        # Guardar los límites X actuales antes de cambiar
        self.domain_x_limits[self.current_domain] = [self.spin_xmin.value(), self.spin_xmax.value()]

        new_domain = self.combo_domain.currentData()
        self.current_domain = new_domain

        if new_domain == "freq_ghz":
            unit = " GHz"
            xlabel = "Frecuencia (GHz)"
            step = 0.5
        elif new_domain == "td_ns":
            unit = " ns"
            xlabel = "Tiempo de Propagación (ns) [Time Domain - IFFT]"
            step = 1.0
        else:
            unit = " m"
            xlabel = "Distancia de Propagación (m) [Time Domain - IFFT]"
            step = 0.5

        full_min, full_max = self._get_full_x_range(new_domain)
        saved_min, saved_max = self.domain_x_limits.get(new_domain, [full_min, full_max])
        saved_min = max(0.0, min(saved_min, full_max))
        saved_max = max(saved_min + 0.01, min(saved_max, full_max))

        self._updating_x_controls = True
        for sp in (self.spin_xmin, self.spin_xmax):
            sp.setRange(0.0, max(full_max * 1.2, 100.0))
            sp.setSingleStep(step)
            sp.setSuffix(unit)
        self.spin_xmin.setValue(saved_min)
        self.spin_xmax.setValue(saved_max)
        self._updating_x_controls = False

        self.ax_ul.set_xlabel(xlabel)
        self.update_x_limits()
        self.refresh_current_plot()

        if self.chk_sync_vna_td.isChecked():
            self.on_sync_vna_td_toggled(True)

    def on_sync_vna_td_toggled(self, checked: bool):
        win_map = {"hann": "hann", "hamming": "hamming", "blackman": "blackman", "rect": "rect"}
        win = win_map.get(self.combo_window.currentData(), "hann")
        enable_td_on_vna = checked and (self.current_domain in ("td_ns", "td_m"))
        self.worker.enqueue_command("set_vna_timedomain", enabled=enable_td_on_vna, window=win)

    def reset_x_limits_full(self):
        """Restaura el eje X al rango máximo disponible en el modo actual."""
        full_min, full_max = self._get_full_x_range(self.current_domain)
        self._updating_x_controls = True
        self.spin_xmin.setValue(full_min)
        self.spin_xmax.setValue(full_max)
        self._updating_x_controls = False
        self.update_x_limits()

    def update_x_limits(self):
        """Actualiza los límites del eje X de Downlink y Uplink al modificar las casillas."""
        if self._updating_x_controls:
            return
        xmin = self.spin_xmin.value()
        xmax = self.spin_xmax.value()
        if xmin < xmax:
            self.domain_x_limits[self.current_domain] = [xmin, xmax]
            self.ax_dl.set_xlim(xmin, xmax)
            self.ax_ul.set_xlim(xmin, xmax)
            self.canvas.draw_idle()

            if self.chk_sync_vna_td.isChecked() and self.current_domain in ("td_ns", "td_m"):
                if self.current_domain == "td_m":
                    self.worker.enqueue_command("set_vna_xscale", xmin=xmin, xmax=xmax, mode="dist")
                else:
                    self.worker.enqueue_command("set_vna_xscale", xmin=xmin * 1e-9, xmax=xmax * 1e-9, mode="time")

    def update_y_limits(self):
        """Actualiza los límites del eje Y de Downlink y Uplink al modificar las casillas."""
        dl_min = self.spin_dl_ymin.value()
        dl_max = self.spin_dl_ymax.value()
        if dl_min < dl_max:
            self.ax_dl.set_ylim(dl_min, dl_max)

        ul_min = self.spin_ul_ymin.value()
        ul_max = self.spin_ul_ymax.value()
        if ul_min < ul_max:
            self.ax_ul.set_ylim(ul_min, ul_max)

        self.canvas.draw_idle()

    def autoscale_plots(self):
        """Ajusta automáticamente la escala Y de Downlink y Uplink según los datos visibles en [xmin, xmax]."""
        if self.last_x_vector is None or self.last_mag_db is None:
            return

        xmin, xmax = self.spin_xmin.value(), self.spin_xmax.value()
        mask = (self.last_x_vector >= xmin) & (self.last_x_vector <= xmax)
        if not np.any(mask):
            mask = np.ones_like(self.last_x_vector, dtype=bool)

        dl_data = self.last_mag_db[:3, mask]
        ul_data = self.last_mag_db[3:6, mask]

        margin_db = 4.0
        dl_min = float(np.floor(np.min(dl_data) - margin_db))
        dl_max = float(np.ceil(np.max(dl_data) + margin_db))
        if dl_max <= dl_min:
            dl_max = dl_min + 10.0

        ul_min = float(np.floor(np.min(ul_data) - margin_db))
        ul_max = float(np.ceil(np.max(ul_data) + margin_db))
        if ul_max <= ul_min:
            ul_max = ul_min + 10.0

        self.spin_dl_ymin.blockSignals(True)
        self.spin_dl_ymax.blockSignals(True)
        self.spin_ul_ymin.blockSignals(True)
        self.spin_ul_ymax.blockSignals(True)

        self.spin_dl_ymin.setValue(dl_min)
        self.spin_dl_ymax.setValue(dl_max)
        self.spin_ul_ymin.setValue(ul_min)
        self.spin_ul_ymax.setValue(ul_max)

        self.spin_dl_ymin.blockSignals(False)
        self.spin_dl_ymax.blockSignals(False)
        self.spin_ul_ymin.blockSignals(False)
        self.spin_ul_ymax.blockSignals(False)

        self.update_y_limits()

    def refresh_current_plot(self):
        """Re-dibuja el último barrido recibido con el dominio o ventana seleccionados."""
        if self.last_freqs_ghz is None or self.last_sparams is None:
            return
        x_vec, mag_db = self._transform_sparams_for_display(self.last_freqs_ghz, self.last_sparams)
        self.last_x_vector = x_vec
        self.last_mag_db = mag_db

        for i in range(6):
            self.lines[i].set_data(x_vec, mag_db[i, :])

        if self.chk_auto_y.isChecked():
            self.autoscale_plots()
        else:
            self.canvas.draw_idle()

    # =========================================================================
    # LÓGICA DE CONTROL DE LA RIS POR COORDENADAS
    # =========================================================================
    def _on_ris_coord_spin_changed(self):
        """Aplica automáticamente el cambio de configuración en la RIS al modificar cualquier coordenada."""
        self.apply_ris_from_ui()

    def apply_ris_from_ui(self):
        """Toma las coordenadas RX y TX de la interfaz, genera el RISConfig y lo aplica con RISServerController."""
        rx_pos = [self.spin_rx_x.value(), self.spin_rx_y.value(), self.spin_rx_z.value()]
        tx_pos = [self.spin_tx_x.value(), self.spin_tx_y.value(), self.spin_tx_z.value()]
        freq_hz = self.spin_ris_freq.value() * 1e9
        boresight = "y"  # Eje perpendicular fijado a 'y'
        optimizer = self.combo_ris_opt.currentData() if hasattr(self, "combo_ris_opt") else "1d_search"
        pol_mode = "vh"  # Ambas polarizaciones con la misma configuración
        ip = RIS_IP
        port = RIS_PORT

        try:
            # 1. Calculamos las matrices 40x40 de estados según las coordenadas RX/TX
            info = self.ris_controller.compute_states_for_coordinates(
                rx_coords=rx_pos,
                tx_pos=tx_pos,
                freq_hz=freq_hz,
                boresight=boresight,
                optimizer=optimizer,
                polarization_mode=pol_mode,
            )

            # 2. Empaquetamos en RISConfig(state_matrix_v, state_matrix_h) y enviamos con RISServerController
            config = RISConfig(
                state_matrix_v=info["estados_40x40"],
                state_matrix_h=info["estados_h_40x40"],
            )
            self.ris_server.addr = (ip, port)
            if not self.ris_connected:
                self.ris_server.connect()
                self.ris_connected = True
            self.ris_server.apply_config(config)

            self.last_ris_info = info
            msg = (
                f"✅ RX=({rx_pos[0]:+.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:+.2f}) m | "
                f"Prx est: {info['p_rx_dbm']:.1f} dBm | {info['latencia_ms']:.1f} ms"
            )
            self.lbl_ris_status.setText(msg)
            self.lbl_ris_status.setStyleSheet("color: #2ca02c; font-weight: bold; margin-left: 4px;")
            self.statusBar().showMessage(
                f"RISConfig aplicado mediante RISServerController en {ip}:{port} -> {msg}"
            )
        except Exception as e:
            self.lbl_ris_status.setText(f"❌ Error RIS: {e}")
            self.lbl_ris_status.setStyleSheet("color: #d9534f; font-weight: bold; margin-left: 4px;")
            self.statusBar().showMessage(f"Error al enviar configuración a la RIS: {e}")

    def apply_uniform_ris(self, state_bit: int):
        ip = RIS_IP
        port = RIS_PORT
        try:
            bit = 1 if int(state_bit) != 0 else 0
            mat_40x40 = np.full((40, 40), bit, dtype=int)
            config = RISConfig(
                state_matrix_v=mat_40x40,
                state_matrix_h=mat_40x40.copy(),
            )
            self.ris_server.addr = (ip, port)
            if not self.ris_connected:
                self.ris_server.connect()
                self.ris_connected = True
            self.ris_server.apply_config(config)

            self.last_ris_info = {
                "mode": f"uniform_{bit}",
                "estados_40x40": mat_40x40,
                "ris_config": config,
            }
            self.lbl_ris_status.setText(f"✅ RIS: Todo {bit} enviado")
            self.lbl_ris_status.setStyleSheet("color: #0275d8; font-weight: bold; margin-left: 4px;")
            self.statusBar().showMessage(f"RISConfig uniforme ({bit}) aplicado con RISServerController en {ip}:{port}.")
        except Exception as e:
            self.lbl_ris_status.setText(f"❌ Error RIS: {e}")
            self.lbl_ris_status.setStyleSheet("color: #d9534f; font-weight: bold; margin-left: 4px;")

    def turn_on_ris(self):
        ip = RIS_IP
        port = RIS_PORT
        try:
            self.ris_server.addr = (ip, port)
            self.ris_server.connect()
            self.ris_connected = True
            self.lbl_ris_status.setText("🟢 RIS: Encendida (ON)")
            self.lbl_ris_status.setStyleSheet("color: #2ca02c; font-weight: bold; margin-left: 4px;")
            self.statusBar().showMessage(f"controller.connect() (ON) enviado al servidor de la RIS ({ip}:{port}).")
        except Exception as e:
            self.lbl_ris_status.setText(f"❌ Error RIS: {e}")

    def turn_off_ris(self):
        ip = RIS_IP
        port = RIS_PORT
        try:
            self.ris_server.addr = (ip, port)
            self.ris_server.disconnect()
            self.ris_connected = False
            self.lbl_ris_status.setText("⏹ RIS: Apagada (OFF)")
            self.lbl_ris_status.setStyleSheet("color: #555555; font-weight: bold; margin-left: 4px;")
            self.statusBar().showMessage(f"controller.disconnect() (OFF) enviado al servidor de la RIS ({ip}:{port}).")
        except Exception as e:
            self.lbl_ris_status.setText(f"❌ Error RIS: {e}")

    # =========================================================================
    # ADQUISICIÓN Y GRABACIÓN DE MEDIDAS DEL VNA
    # =========================================================================
    def start_recording(self):
        """Activa el almacenamiento de medidas durante el tiempo elegido."""
        if self.is_recording:
            return

        self.record_duration = self.spin_duration.value()
        self.recorded_sparams = []
        self.recorded_timestamps = []
        self.record_start_time = time.time()
        self.is_recording = True

        self.btn_record.setEnabled(False)
        self.spin_duration.setEnabled(False)
        self.btn_record.setText("⏳ Grabando...")

    def update_measurement(self, freqs_ghz: np.ndarray, sparams: np.ndarray, ts: str):
        """Actualiza las gráficas en tiempo real y guarda en el buffer si está grabando."""
        if self.current_freqs_hz is None:
            self.current_freqs_hz = freqs_ghz * 1e9

        self.last_freqs_ghz = freqs_ghz
        self.last_sparams = sparams

        self.refresh_current_plot()

        if self.is_recording:
            elapsed = time.time() - self.record_start_time
            remaining = max(0.0, self.record_duration - elapsed)

            self.recorded_sparams.append(sparams.astype(np.complex64))
            self.recorded_timestamps.append(ts)

            self.lbl_rec_status.setText(
                f"🔴 GRABANDO ({remaining:.1f}s) | Barridos: {len(self.recorded_sparams)}"
            )
            self.lbl_rec_status.setStyleSheet("font-weight: bold; color: #d9534f; margin-left: 6px;")

            if elapsed >= self.record_duration:
                self.finish_and_save_recording()

    def finish_and_save_recording(self):
        """Detiene la grabación y vuelca las medidas a un archivo .npz."""
        self.is_recording = False

        sparams_array = np.array(self.recorded_sparams, dtype=np.complex64)
        file_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{FOLDER_NAME}/ul_dl_capture_{file_ts}.npz"

        conf_params = {
            'ifbw': IFBW,
            'pow_dbm_list': POW_DBM_LIST,
            'sparam_list': SPARAM_LIST,
            'f_start': F_START,
            'f_stop': F_STOP,
            'points': POINTS,
            'record_duration_s': self.record_duration,
            'num_captured_sweeps': len(self.recorded_sparams),
            'ris_rx_coords_m': [self.spin_rx_x.value(), self.spin_rx_y.value(), self.spin_rx_z.value()],
            'ris_tx_coords_m': [self.spin_tx_x.value(), self.spin_tx_y.value(), self.spin_tx_z.value()],
            'ris_freq_ghz': self.spin_ris_freq.value(),
        }
        if self.last_ris_info is not None and "estados_40x40" in self.last_ris_info:
            conf_params['ris_states_40x40'] = self.last_ris_info["estados_40x40"]

        np.savez(
            filename,
            sparams=sparams_array,
            freqs_hz=self.current_freqs_hz,
            timestamps=np.array(self.recorded_timestamps),
            conf_params=conf_params
        )

        self.btn_record.setEnabled(True)
        self.spin_duration.setEnabled(True)
        self.btn_record.setText("🔴 Iniciar Grabación y Guardar")
        self.lbl_rec_status.setText(f"✅ Guardado ({len(self.recorded_sparams)} barridos)")
        self.lbl_rec_status.setStyleSheet("font-weight: bold; color: #2ca02c; margin-left: 6px;")
        self.statusBar().showMessage(f"Archivo guardado exitosamente en {filename}")

    def handle_error(self, err_msg: str):
        QMessageBox.critical(self, "Error en el VNA", f"Se produjo un error de comunicación:\n{err_msg}")
        self.lbl_rec_status.setText("❌ Error de conexión")

    def closeEvent(self, event):
        """Cierra de forma segura el hilo del VNA al salir de la aplicación."""
        self.statusBar().showMessage("Cerrando conexión con el VNA...")
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    app = QApplication(sys.argv)
    window = VNARealtimeWindow(demo_mode=demo)
    window.show()
    sys.exit(app.exec())
