import sys
import os
import time
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QGroupBox, QStatusBar, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mcg_lab_vna_control import connect_vna


# ================= PARÁMETROS DEL VNA =================
FOLDER_NAME = "data/VA_static_ul_dl"
SPARAM_LIST = ['S21', 'S31', 'S41', 'S12', 'S13', 'S14']  # 3 Downlink + 3 Uplink
TRACE_LIST = [1, 2, 3, 4, 5, 6]

F_START = 15.0e9          # 15 GHz
F_STOP = 30.0e9           # 30 GHz
POINTS = 3001             # Puntos de frecuencia
IFBW = 10e3               # 10 kHz
POW_DBM_LIST = [-15, 0, 0, 0]  # Potencia en puertos [1, 2, 3, 4] en dBm
# ======================================================


class VNAWorker(QThread):
    """Hilo en segundo plano para medir continuamente sin bloquear la interfaz gráfica."""
    data_ready = pyqtSignal(np.ndarray, np.ndarray, str)  # (freqs_ghz, sparams_complex, timestamp)
    status_msg = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.vna = None
        self.freqs = None

    def run(self):
        try:
            self.status_msg.emit("Conectando al VNA Keysight...")
            self.vna = connect_vna(brand='keysight')
            self.vna.pseudoo_preset()        

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
                # Mide las 6 trazas simultáneamente: shape (6, POINTS)
                sparams = self.vna.measure_multiple_traces_sparams(trace_list=TRACE_LIST)
                ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                self.data_ready.emit(freqs_ghz, sparams, ts)
                time.sleep(0.02)

            self.vna.run_cont()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


class VNARealtimeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor en Tiempo Real VNA (15 - 30 GHz) | Downlink & Uplink")
        self.resize(1200, 820)

        os.makedirs(FOLDER_NAME, exist_ok=True)

        # Estado de grabación
        self.is_recording = False
        self.record_start_time = 0.0
        self.record_duration = 5.0
        self.recorded_sparams = []
        self.recorded_timestamps = []
        self.current_freqs_hz = None

        self._build_ui()

        # Iniciar el hilo del VNA
        self.worker = VNAWorker()
        self.worker.data_ready.connect(self.update_measurement)
        self.worker.status_msg.connect(self.statusBar().showMessage)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Fila Superior: Controles de Grabación y Escala Y ---
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
        self.spin_duration.setMinimumWidth(85)

        self.btn_record = QPushButton("🔴 Iniciar Grabación y Guardar")
        self.btn_record.setMinimumHeight(34)
        self.btn_record.setStyleSheet(
            "font-weight: bold; background-color: #d9534f; color: white; border-radius: 5px; padding: 4px 12px;"
        )
        self.btn_record.clicked.connect(self.start_recording)

        self.lbl_rec_status = QLabel("Estado: Monitorizando")
        self.lbl_rec_status.setStyleSheet("color: #555555; margin-left: 8px;")

        record_layout.addWidget(lbl_duration)
        record_layout.addWidget(self.spin_duration)
        record_layout.addWidget(self.btn_record)
        record_layout.addWidget(self.lbl_rec_status)

        # 2. Grupo de Escala Eje Y - DOWNLINK
        dl_scale_group = QGroupBox("Eje Y Downlink (dB)")
        dl_scale_layout = QHBoxLayout(dl_scale_group)

        self.spin_dl_ymin = QDoubleSpinBox()
        self.spin_dl_ymin.setRange(-200.0, 50.0)
        self.spin_dl_ymin.setValue(-90.0)
        self.spin_dl_ymin.setSingleStep(5.0)
        self.spin_dl_ymin.setSuffix(" dB")
        self.spin_dl_ymin.valueChanged.connect(self.update_y_limits)

        self.spin_dl_ymax = QDoubleSpinBox()
        self.spin_dl_ymax.setRange(-200.0, 50.0)
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
        self.spin_ul_ymin.setRange(-200.0, 50.0)
        self.spin_ul_ymin.setValue(-90.0)
        self.spin_ul_ymin.setSingleStep(5.0)
        self.spin_ul_ymin.setSuffix(" dB")
        self.spin_ul_ymin.valueChanged.connect(self.update_y_limits)

        self.spin_ul_ymax = QDoubleSpinBox()
        self.spin_ul_ymax.setRange(-200.0, 50.0)
        self.spin_ul_ymax.setValue(0.0)
        self.spin_ul_ymax.setSingleStep(5.0)
        self.spin_ul_ymax.setSuffix(" dB")
        self.spin_ul_ymax.valueChanged.connect(self.update_y_limits)

        ul_scale_layout.addWidget(QLabel("Mín:"))
        ul_scale_layout.addWidget(self.spin_ul_ymin)
        ul_scale_layout.addWidget(QLabel("Máx:"))
        ul_scale_layout.addWidget(self.spin_ul_ymax)

        # Añadir los 3 grupos al panel superior
        top_controls_layout.addWidget(record_group, stretch=2)
        top_controls_layout.addWidget(dl_scale_group, stretch=1)
        top_controls_layout.addWidget(ul_scale_group, stretch=1)

        main_layout.addLayout(top_controls_layout)

        # --- Lienzo de Gráficas (Matplotlib) ---
        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Subplot 1: Downlink (S21, S31, S41)
        self.ax_dl = self.fig.add_subplot(211)
        self.ax_dl.set_title("Downlink: Transmisión Puerto 1 ➔ Puertos 2, 3, 4 (S21, S31, S41)")
        self.ax_dl.set_ylabel("Magnitud (dB)")
        self.ax_dl.set_xlim(F_START / 1e9, F_STOP / 1e9)
        self.ax_dl.set_ylim(self.spin_dl_ymin.value(), self.spin_dl_ymax.value())
        self.ax_dl.grid(True, linestyle='--', alpha=0.6)

        # Subplot 2: Uplink (S12, S13, S14)
        self.ax_ul = self.fig.add_subplot(212)
        self.ax_ul.set_title("Uplink: Recepción en Puerto 1 ⬅ Puertos 2, 3, 4 (S12, S13, S14)")
        self.ax_ul.set_xlabel("Frecuencia (GHz)")
        self.ax_ul.set_ylabel("Magnitud (dB)")
        self.ax_ul.set_xlim(F_START / 1e9, F_STOP / 1e9)
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

        mag_db = 20 * np.log10(np.abs(sparams) + 1e-12)
        for i in range(6):
            self.lines[i].set_data(freqs_ghz, mag_db[i, :])

        self.canvas.draw_idle()

        if self.is_recording:
            elapsed = time.time() - self.record_start_time
            remaining = max(0.0, self.record_duration - elapsed)

            self.recorded_sparams.append(sparams.astype(np.complex64))
            self.recorded_timestamps.append(ts)

            self.lbl_rec_status.setText(
                f"🔴 GRABANDO ({remaining:.1f}s) | Barridos: {len(self.recorded_sparams)}"
            )
            self.lbl_rec_status.setStyleSheet("font-weight: bold; color: #d9534f; margin-left: 8px;")

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
            'num_captured_sweeps': len(self.recorded_sparams)
        }

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
        self.lbl_rec_status.setStyleSheet("font-weight: bold; color: #2ca02c; margin-left: 8px;")
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
    app = QApplication(sys.argv)
    window = VNARealtimeWindow()
    window.show()
    sys.exit(app.exec())