import os
import time
import numpy as np
from mcg_lab_vna_control import connect_vna

# ================= PARÁMETROS DE CALIBRACIÓN =================
FOLDER_NAME = "data/VA_static_ul_dl"

# Misma configuración de frecuencias y trazas que usarás en la medida (15 - 30 GHz)
SPARAM_LIST = ['S21', 'S31', 'S41', 'S12', 'S13', 'S14']
F_START = 15.0e9          # 15 GHz
F_STOP = 30.0e9           # 30 GHz
POINTS = 3001             # Puntos de frecuencia
IFBW = 10e3               # 10 kHz
POW_DBM_LIST = [-15, 0, 0, 0]

# Parámetros del Kit de Calibración
PORTS_TO_CALIBRATE = [1, 2, 3, 4]
CONN_TYPE = '2.92'        # Tipo de conector (2.92 mm)
CONN_GENDER = 'f'         # Género del conector ('f' = female / hembra)
CALKIT = '85561A'         # Kit de calibración ('85561A' o 'ZV-Z129')
# =============================================================


def wait_sweep(vna, multiplier=6):
    """Espera el tiempo necesario para completar el barrido de todas las trazas."""
    sweep_time = vna.query_sweep_time(verbose=False)
    time.sleep(multiplier * sweep_time + 1.0)


def calibrate_sol_port(vna, port: int):
    """Guía paso a paso para la calibración SOL (Open, Short, Load) de un puerto."""
    print(f"\n" + "=" * 60)
    print(f"🔧 INICIANDO CALIBRACIÓN SOL - PUERTO {port}")
    print("=" * 60)

    # Inicia el asistente SOL en el VNA (imprime los pasos requeridos por pantalla)
    vna.init_SOL_calibration(
        port=port,
        conn_type=CONN_TYPE,
        conn_gender=CONN_GENDER,
        calkit=CALKIT
    )

    standards = ["OPEN (Abierto)", "SHORT (Corto)", "LOAD / MATCH (Carga 50 Ω)"]

    for step_idx, std_name in enumerate(standards, start=1):
        input(f"\n👉 [Paso {step_idx}/3] Conecta '{std_name}' en el PUERTO {port} y pulsa ENTER...")
        print(f"   ⏳ Midiendo paso {step_idx} ({std_name}) en el Puerto {port}...")
        vna.measure_SOL_calibrator(step_idx)

    print(f"\n💾 Aplicando calibración SOL en el Puerto {port}...")
    vna.apply_calibration()
    print(f"✅ Puerto {port} calibrado correctamente.")


def calibrate_through_b2b(vna):
    """Guía opcional para medir y guardar los archivos Through (Back-to-Back)."""
    print("\n" + "=" * 60)
    print("🔗 CALIBRACIÓN THROUGH (BACK-TO-BACK) EN TRANSMISIÓN")
    print("=" * 60)

    os.makedirs(FOLDER_NAME, exist_ok=True)

    # Trazas de referencia: (Nombre, Número de traza en VNA, Puertos a unir)
    b2b_traces = [
        ("S21", 1, "Puerto 1 y Puerto 2"),
        ("S31", 2, "Puerto 1 y Puerto 3"),
        ("S41", 3, "Puerto 1 y Puerto 4"),
    ]

    for sp_name, trace_num, ports_desc in b2b_traces:
        ans = input(f"\n¿Quieres medir el Through para {sp_name} ({ports_desc})? [s/N]: ").strip().lower()
        if ans == 's':
            input(f"👉 Conecta directamente los cables del {ports_desc} (Through) y pulsa ENTER...")
            print(f"   ⏳ Midiendo traza {trace_num} ({sp_name})...")
            sparam = vna.measure_trace_sparam(trace=trace_num)
            vna.auto_scale_all()

            filepath = os.path.join(FOLDER_NAME, f"{sp_name}_b2b_cal.npy")
            np.save(filepath, sparam)
            print(f"   ✅ Guardado en: {filepath}")


def main():
    print("============================================================")
    print("       ASISTENTE DE CALIBRACIÓN DEL VNA (15 - 30 GHz)       ")
    print("============================================================")

    print("\n1. Conectando al VNA Keysight...")
    vna = connect_vna(brand='keysight')

    print("2. Haciendo Preset y aplicando configuración base (15 - 30 GHz)...")
    vna.preset()
    vna.configure_multiple_sparams(
        sparam_list=SPARAM_LIST,
        f_start=F_START,
        f_stop=F_STOP,
        points=POINTS,
        ifbw=IFBW,
        pow_dbm_list=POW_DBM_LIST
    )

    wait_sweep(vna, multiplier=len(SPARAM_LIST))
    vna.auto_scale_all()
    vna.run_cont()
    print("✅ Configuración base cargada en el VNA.")

    # Selección de puertos a calibrar
    print(f"\nKit configurado: {CALKIT} | Conector: {CONN_TYPE} mm ({CONN_GENDER})")
    ports_input = input("¿Qué puertos quieres calibrar con SOL? (Pulsa ENTER para todos [1,2,3,4] o escribe ej. '1,2'): ").strip()

    if ports_input:
        selected_ports = [int(p.strip()) for p in ports_input.split(",") if p.strip().isdigit()]
    else:
        selected_ports = PORTS_TO_CALIBRATE

    # Ejecutar calibración SOL puerto por puerto
    for port in selected_ports:
        calibrate_sol_port(vna, port)

    # Activar explícitamente la corrección de calibración
    vna.calibration_on()
    vna.run_cont()

    # Preguntar si también se desea medir el Through (Back-to-Back)
    do_b2b = input("\n¿Deseas realizar también la medida Through (Back-to-Back) de los cables? [s/N]: ").strip().lower()
    if do_b2b == 's':
        calibrate_through_b2b(vna)
        vna.run_cont()

    print("\n🎉 ¡PROCESO DE CALIBRACIÓN COMPLETADO!")
    print("El VNA ha quedado calibrado y con la corrección activa.")
    print("⚠️ IMPORTANTE: Si ahora abres 'vna_realtime_gui.py', recuerda comentar 'self.vna.preset()' en la GUI para no borrar esta calibración.")


if __name__ == "__main__":
    main()