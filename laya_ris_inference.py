"""Módulo y script interactivo de inferencia con Laya para RIS 40x40.

Calcula en milisegundos la matriz binaria 40x40 (3200 estados V/H)
a partir de las coordenadas 3D de 1 transmisor (TX) y 1 a 3 receptores (RX),
utilizando el modelo entrenado en `laya_ris_experiments` (checkpoint_best.pt).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Inyección automática de entornos para compatibilidad con test1-dav y agentic-ris
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
AGENTIC_RIS_SRC = WORKSPACE_ROOT / "agentic-ris" / "src"
AGENTIC_RIS_SITE = WORKSPACE_ROOT / "agentic-ris" / ".venv" / "Lib" / "site-packages"
SURPRISE_SRC = WORKSPACE_ROOT / "surpRISe" / "src"
DEFAULT_EXP_DIR = WORKSPACE_ROOT / "agentic-ris" / "laya_ris_experiments" / "laya_v5"

for p in (AGENTIC_RIS_SRC, SURPRISE_SRC, AGENTIC_RIS_SITE):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import yaml

from laya_ris.observations.sensing import true_rtt_ns
from laya_ris.observations.text import TextFormatter, to_structured
from laya_ris.ris_policy.geometry import RISGeometry
from laya_ris.scenarios.geometry import vec_to_dir
from laya_ris.scenarios.schema import (
    Observation,
    PhysicalScenario,
    RISConfig as LayaRISConfig,
    UE,
    UEObservation,
)
from laya_ris.training.controller import Controller
from laya_ris.training.data import Item
from ris_beamforming_controller import RISConfig, RISServerController


class LayaRISInferencer:
    """Cargador en memoria del modelo Laya entrenado para inferencia rápida de la RIS."""

    def __init__(
        self,
        exp_dir: Union[str, Path] = DEFAULT_EXP_DIR,
        device: str = "auto",
        warmup: bool = True,
    ) -> None:
        self.exp_dir = Path(exp_dir)
        if not self.exp_dir.exists():
            raise FileNotFoundError(f"Directorio de experimento no encontrado: {self.exp_dir}")

        cfg_path = self.exp_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuración no encontrada en: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Ajuste de ruta de checkpoint lector local
        reader_candidates = [
            self.exp_dir / "reader" / "reader.pt",
            self.exp_dir.parent / "reader" / "reader.pt",
            WORKSPACE_ROOT / "agentic-ris" / "laya_ris_experiments" / "reader" / "reader.pt",
            WORKSPACE_ROOT / "agentic-ris" / "artifacts" / "laya_reader" / "reader.pt",
        ]
        local_reader = next((p for p in reader_candidates if p.exists()), None)
        self.cfg["laya_backbone"]["init_from"] = str(local_reader) if local_reader is not None else None

        # Usar snapshot local de HuggingFace si existe para inferencia 100% offline
        snapshot_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--convaiinnovations--laya"
            / "snapshots"
            / "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
        )
        if snapshot_dir.exists():
            self.cfg["laya_backbone"]["model_id"] = str(snapshot_dir)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.cfg["laya_backbone"]["device"] = self.device

        # Construir geometría RIS y controlador Laya
        self.ris_schema_cfg = LayaRISConfig.from_dict(self.cfg["scenarios"]["ris"])
        self.controller = Controller(self.cfg, self.ris_schema_cfg)

        ckpt_path = self.exp_dir / "checkpoint_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.controller.load_checkpoint_state(ckpt["state"])
        if hasattr(self.controller, "prepare_for_fast_inference"):
            self.controller.prepare_for_fast_inference()
        else:
            self.controller.eval()

        self.formatter = TextFormatter({
            "template_version": "v1",
            "randomize_ue_order": False,
            "paraphrase": False,
        })

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Ejecuta una inferencia sintética inicial para compilar kernels de PyTorch/CUDA."""
        dummy_tx = [0.0, 1.1, 0.0]
        dummy_rx = [[-0.5, 1.4, 0.0]]
        self.compute_states_for_coordinates(dummy_rx, dummy_tx, boresight="y")

    @staticmethod
    def _to_laya_frame(coords: np.ndarray, boresight: str = "y") -> np.ndarray:
        """Convierte coordenadas 3D del sistema del laboratorio (con normal `boresight`)

        al sistema canónico de Laya (con normal `+Z = [0, 0, 1]`).
        En boresight 'y': normal +Y -> +Z, eje lateral -X -> +X, eje vertical -Z -> +Y.
        """
        arr = np.asarray(coords, dtype=float)
        b = boresight.lower().strip()
        if b == "y":
            # [x, y, z] -> [-x, z, y]
            if arr.ndim == 1:
                return np.array([-arr[0], arr[2], arr[1]], dtype=float)
            return np.stack([-arr[:, 0], arr[:, 2], arr[:, 1]], axis=-1)
        elif b == "x":
            # [x, y, z] -> [y, z, x]
            if arr.ndim == 1:
                return np.array([arr[1], arr[2], arr[0]], dtype=float)
            return np.stack([arr[:, 1], arr[:, 2], arr[:, 0]], axis=-1)
        else:
            return arr.copy()

    def build_item_from_coordinates(
        self,
        tx_pos: Sequence[float],
        rx_coords: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
        boresight: str = "y",
        priorities: Optional[Sequence[str]] = None,
        freq_hz: Optional[float] = None,
    ) -> Item:
        """Construye un objeto `Item` (PhysicalScenario + Observation + Prompt Text) determinista."""
        tx_lab = np.asarray(tx_pos, dtype=float).reshape(3)
        rx_lab = np.asarray(rx_coords, dtype=float)
        if rx_lab.ndim == 1:
            if rx_lab.size != 3:
                raise ValueError("Las coordenadas del RX deben tener 3 valores [x, y, z].")
            rx_lab = rx_lab.reshape(1, 3)
        elif rx_lab.ndim != 2 or rx_lab.shape[1] != 3:
            raise ValueError("Las coordenadas deben tener forma [3] o [K, 3].")

        num_ues = len(rx_lab)
        if num_ues > 3:
            raise ValueError(f"Laya soporta hasta 3 UEs simultáneos (recibidos {num_ues}).")

        fc = float(freq_hz) if freq_hz is not None else float(self.ris_schema_cfg.fc)

        tx_laya = self._to_laya_frame(tx_lab, boresight=boresight)
        rx_laya = self._to_laya_frame(rx_lab, boresight=boresight)

        # 1. Escenario Físico en marco Laya
        ues_scen = []
        for k in range(num_ues):
            prio = priorities[k] if (priorities and k < len(priorities)) else "normal"
            weight = 2.0 if prio == "high" else 1.0
            ues_scen.append(UE(uid=k, position=rx_laya[k].tolist(), priority=prio, weight=weight))

        scenario = PhysicalScenario(
            scenario_id="interactive_request",
            family_id="manual",
            split="inference",
            ris=self.ris_schema_cfg,
            tx_position=tx_laya.tolist(),
            tx_gain_dbi=15.7,
            rx_gain_dbi=0.0,
            ues=ues_scen,
            protected=[],
        )

        # 2. Observación determinista (sin ruido aleatorio)
        inc_az, inc_el, tx_r = vec_to_dir(tx_laya)
        ue_obs_list = []
        for k, u in enumerate(ues_scen):
            az, el, r = vec_to_dir(rx_laya[k])
            rtt = true_rtt_ns(scenario, rx_laya[k])
            ue_obs_list.append(
                UEObservation(
                    uid=k,
                    priority=u.priority,
                    az_deg=float(az),
                    el_deg=float(el),
                    aoa_sigma_deg=1.0,
                    rtt_ns=float(rtt),
                    rtt_sigma_ns=0.5,
                )
            )

        obs = Observation(
            scenario_id="interactive_request",
            fc_ghz=fc / 1e9,
            nrow=self.ris_schema_cfg.nrow,
            ncol=self.ris_schema_cfg.ncol,
            incident_az_deg=float(inc_az),
            incident_el_deg=float(inc_el),
            incident_sigma_deg=1.0,
            tx_range_m=float(tx_r),
            ues=ue_obs_list,
            protected=[],
            rtt_bias_ns=0.0,
        )

        text, order = self.formatter.render_ordered(obs, paraphrase=False)
        structured = to_structured(obs, order)

        return Item(scenario=scenario, obs=obs, text=text, structured=structured, order=order)

    def compute_states_for_coordinates(
        self,
        rx_coords: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
        tx_pos: Sequence[float] = (0.0, 1.1, 0.0),
        freq_hz: float = 28.5e9,
        boresight: str = "y",
        polarization_mode: str = "vh",
        priorities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Infiere la matriz 40x40 en milisegundos con Laya y empaqueta en RISConfig."""
        t0 = time.perf_counter()

        item = self.build_item_from_coordinates(
            tx_pos=tx_pos,
            rx_coords=rx_coords,
            boresight=boresight,
            priorities=priorities,
            freq_hz=freq_hz,
        )

        with torch.inference_mode():
            mask_flat = self.controller.act([item], rule="mean")[0].cpu().numpy().astype(int)

        estados_2d = mask_flat.reshape(self.ris_schema_cfg.nrow, self.ris_schema_cfg.ncol)

        # Configuración de polarización
        pol = polarization_mode.lower().strip()
        if pol == "v_only":
            mat_v = estados_2d
            mat_h = np.zeros_like(estados_2d, dtype=int)
        elif pol == "h_only":
            mat_v = np.zeros_like(estados_2d, dtype=int)
            mat_h = estados_2d
        else:  # "vh"
            mat_v = estados_2d
            mat_h = estados_2d.copy()

        ris_config = RISConfig(state_matrix_v=mat_v, state_matrix_h=mat_h)
        gw_pattern = RISServerController.build_gw_pattern(ris_config)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "rx_coords": rx_coords,
            "tx_pos": tx_pos,
            "freq_hz": freq_hz,
            "boresight": boresight,
            "optimizer": "laya_v5_residual_phasor",
            "polarization_mode": pol,
            "ris_config": ris_config,
            "estados_40x40": mat_v,
            "estados_h_40x40": mat_h,
            "gw_pattern": gw_pattern,
            "latencia_ms": elapsed_ms,
            "prompt_text": item.text,
        }

    def apply_coordinates(
        self,
        rx_coords: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
        tx_pos: Sequence[float] = (0.0, 1.1, 0.0),
        server_ip: str = "192.168.100.3",
        server_port: int = 17000,
        freq_hz: float = 28.5e9,
        boresight: str = "y",
        polarization_mode: str = "vh",
        send_udp: bool = True,
    ) -> Dict[str, Any]:
        """Calcula la matriz con Laya y la envía por UDP al servidor de la RIS."""
        info = self.compute_states_for_coordinates(
            rx_coords=rx_coords,
            tx_pos=tx_pos,
            freq_hz=freq_hz,
            boresight=boresight,
            polarization_mode=polarization_mode,
        )

        if send_udp:
            server = RISServerController(ip=server_ip, port=server_port)
            server.connect()
            server.apply_config(info["ris_config"])
            info["udp_sent"] = True
            info["server_addr"] = f"{server_ip}:{server_port}"
        else:
            info["udp_sent"] = False

        return info


def _export_matrix(mat: np.ndarray, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, mat)
    elif suffix in (".csv", ".txt"):
        # Formato CSV compatible
        np.savetxt(path, mat, fmt="%d", delimiter=",")
    else:
        # Por defecto CSV
        np.savetxt(str(path) + ".csv", mat, fmt="%d", delimiter=",")
    print(f"[OK] Matriz 40x40 exportada exitosamente a: {path}")


def interactive_session(inferencer: LayaRISInferencer) -> None:
    """Consola interactiva para evaluar múltiples escenarios en tiempo real."""
    print("=" * 70)
    print("   CONSOLA INTERACTIVA DE INFERENCIA LAYA RIS 40x40")
    print("=" * 70)
    print("Introduce coordenadas 3D en metros. Escribe 'q' o 'salir' para terminar.\n")

    default_tx = "0.0, 1.1, 0.0"

    while True:
        try:
            line_tx = input(f"Transmisor TX [x, y, z] (default: {default_tx}): ").strip()
            if line_tx.lower() in ("q", "quit", "exit", "salir"):
                break
            if not line_tx:
                line_tx = default_tx

            tx = [float(v.strip()) for v in line_tx.replace(";", ",").split(",") if v.strip()]
            if len(tx) != 3:
                print("Error: TX debe contener 3 valores x, y, z.")
                continue

            line_rx = input("Receptor(es) RX [x, y, z] (si son varios, sepáralos con ';'): ").strip()
            if line_rx.lower() in ("q", "quit", "exit", "salir"):
                break
            if not line_rx:
                print("Debes introducir al menos un receptor.")
                continue

            rx_list = []
            for item in line_rx.split(";"):
                item = item.strip()
                if not item:
                    continue
                coords = [float(v.strip()) for v in item.split(",") if v.strip()]
                if len(coords) != 3:
                    print(f"Error: Coordenada RX inválida '{item}' (deben ser 3 números).")
                    rx_list = []
                    break
                rx_list.append(coords)

            if not rx_list:
                continue

            # Inferencia
            res = inferencer.compute_states_for_coordinates(rx_coords=rx_list, tx_pos=tx)
            mat = res["estados_40x40"]
            pct_ones = (np.sum(mat) / mat.size) * 100.0

            print("\n" + "-" * 70)
            print(f"[LAYA] INFERENCIA COMPLETADA EN {res['latencia_ms']:.2f} ms")
            print(f"   TX: {tx} m | UEs ({len(rx_list)}): {rx_list} m")
            print(f"   Distribución de estados 40x40: 1s={pct_ones:.1f}% | 0s={100.0 - pct_ones:.1f}%")
            print("-" * 70)

            act = input("¿Deseas exportar o enviar por UDP? [c=csv, n=npy, u=udp, enter=continuar]: ").strip().lower()
            if act in ("c", "csv"):
                fname = input("Nombre de archivo (ej: ris_laya.csv): ").strip() or "ris_laya.csv"
                _export_matrix(mat, fname)
            elif act in ("n", "npy"):
                fname = input("Nombre de archivo (ej: ris_laya.npy): ").strip() or "ris_laya.npy"
                _export_matrix(mat, fname)
            elif act in ("u", "udp"):
                ip = input("IP del servidor RIS (default 192.168.100.3): ").strip() or "192.168.100.3"
                port = int(input("Puerto UDP (default 17000): ").strip() or "17000")
                server = RISServerController(ip=ip, port=port)
                server.connect()
                server.apply_config(res["ris_config"])
                print(f"[OK] Configuración de Laya enviada por UDP a {ip}:{port}")
            print("\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script de inferencia interactivo Laya RIS 40x40.")
    parser.add_argument("--tx", nargs=3, type=float, default=[0.0, 1.1, 0.0], metavar=("X", "Y", "Z"),
                        help="Coordenadas 3D del transmisor TX en metros (ej: 0.0 1.1 0.0)")
    parser.add_argument("--rx", nargs="+", type=float, default=None,
                        help="Coordenadas RX en metros: 3 valores para 1 RX (X Y Z) o múltiplos de 3 para varios.")
    parser.add_argument("--boresight", type=str, default="y", choices=["y", "z", "x"],
                        help="Eje de boresight en laboratorio (default: 'y')")
    parser.add_argument("--freq-ghz", type=float, default=28.5, help="Frecuencia en GHz (default: 28.5)")
    parser.add_argument("--export-csv", type=str, default=None, help="Ruta de archivo para guardar matriz 40x40 en CSV")
    parser.add_argument("--export-npy", type=str, default=None, help="Ruta de archivo para guardar matriz 40x40 en NPY")
    parser.add_argument("--send-udp", action="store_true", help="Enviar matriz resultante por UDP a la RIS")
    parser.add_argument("--ip", type=str, default="192.168.100.3", help="IP del servidor RIS (default: 192.168.100.3)")
    parser.add_argument("--port", type=int, default=17000, help="Puerto UDP del servidor RIS (default: 17000)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--interactive", action="store_true", help="Iniciar consola interactiva")
    args = parser.parse_args()

    print("Cargando Laya y pesos entrenados...")
    inferencer = LayaRISInferencer(device=args.device)
    print(f"[OK] Laya listo en dispositivo: {inferencer.device.upper()}\n")

    if args.interactive or args.rx is None:
        interactive_session(inferencer)
        return

    # Parsear RXs
    rx_raw = args.rx
    if len(rx_raw) % 3 != 0:
        raise ValueError("El número total de valores para --rx debe ser múltiplo de 3 (X Y Z por receptor).")
    rx_coords = np.array(rx_raw, dtype=float).reshape(-1, 3)

    info = inferencer.compute_states_for_coordinates(
        rx_coords=rx_coords,
        tx_pos=args.tx,
        freq_hz=args.freq_ghz * 1e9,
        boresight=args.boresight,
    )

    mat = info["estados_40x40"]
    pct_ones = (np.sum(mat) / mat.size) * 100.0

    print("=" * 70)
    print(f"[LAYA] INFERENCIA COMPLETADA EN {info['latencia_ms']:.2f} ms")
    print(f"   TX: {args.tx} m | RX ({len(rx_coords)}): {rx_coords.tolist()} m")
    print(f"   Matriz 40x40: 1s={pct_ones:.1f}% | 0s={100.0 - pct_ones:.1f}%")
    print("=" * 70)

    if args.export_csv:
        _export_matrix(mat, args.export_csv)
    if args.export_npy:
        _export_matrix(mat, args.export_npy)

    if args.send_udp:
        server = RISServerController(ip=args.ip, port=args.port)
        server.connect()
        server.apply_config(info["ris_config"])
        print(f"[OK] RISConfig de Laya enviado por UDP a {args.ip}:{args.port}")


if __name__ == "__main__":
    main()
