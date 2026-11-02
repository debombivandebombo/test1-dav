"""
Controlador externo para calcular y enviar la configuración de la RIS (40x40 dual-pol, 3200 estados)
a partir de coordenadas 3D [x, y, z] hacia `ris_hardware_server.py` (en `ris-fr2-sandbox`).

Cumple exactamente el formato del README de `ris-fr2-sandbox`:
- Protocolo UDP (`socket.SOCK_DGRAM`) en `127.0.0.1:17000` (configurable).
- Mensaje JSON UTF-8: `{"cmd": "SET_CONFIG", "data": gw_pattern}` (o `"ON"` / `"OFF"`).
- `gw_pattern`: lista de 3200 enteros (0 o 1) intercalando polarización vertical (índices pares)
  y horizontal (índices impares), donde cada matriz 40x40 se invierte verticalmente y se
  transpone antes de aplanarse: `state_matrix[::-1, :].T.flatten().astype(int)`.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Intentamos importar RISConfig y RISServerController desde el repositorio hermano `surpRISe`
_SURPRISE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "surpRISe", "src"))
if os.path.isdir(_SURPRISE_SRC) and _SURPRISE_SRC not in sys.path:
    sys.path.insert(0, _SURPRISE_SRC)

try:
    from surpRISe.components.ris_controller import RISConfig, RISServerController
except ImportError:
    @dataclass
    class RISConfig:
        state_matrix_v: np.ndarray
        state_matrix_h: Optional[np.ndarray] = None

        def __post_init__(self) -> None:
            self.state_matrix_v = np.asarray(self.state_matrix_v, dtype=int)
            if self.state_matrix_h is None:
                self.state_matrix_h = self.state_matrix_v.copy()
            else:
                self.state_matrix_h = np.asarray(self.state_matrix_h, dtype=int)

    class RISServerController:
        def __init__(self, ip: str = "127.0.0.1", port: int = 17000) -> None:
            self.addr = (ip, port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def connect(self) -> None:
            self._send_command("ON")

        def disconnect(self) -> None:
            self._send_command("OFF")

        @staticmethod
        def build_gw_pattern(config: RISConfig) -> list[int]:
            v_flat = config.state_matrix_v[::-1, :].T.flatten().astype(int)
            h_flat = config.state_matrix_h[::-1, :].T.flatten().astype(int)

            gw_pattern = [0] * (v_flat.size + h_flat.size)
            gw_pattern[0::2] = v_flat.tolist()
            gw_pattern[1::2] = h_flat.tolist()
            return gw_pattern

        def apply_config(self, config: RISConfig) -> list[int]:
            gw_pattern = self.build_gw_pattern(config)
            self._send_command("SET_CONFIG", gw_pattern)
            return gw_pattern

        def _send_command(self, cmd: str, data=None) -> None:
            msg = {"cmd": cmd, "data": data}
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)


class RISCoordinateController:
    """
    Calcula el patrón de fases/estados de la RIS (campo cercano, modelo general de Tang
    con optimización SCRPA para 1 usuario o WMMSE-MM para multiusuario) en base a las
    coordenadas 3D introducidas y envía la configuración al servidor UDP de la RIS.
    """

    def __init__(
        self,
        server_ip: str = "127.0.0.1",
        server_port: int = 17000,
        freq_hz: float = 28.0e9,
        ris_nx: int = 40,
        ris_ny: int = 40,
        ris_bits: int = 1,
        tx_pos: Sequence[float] = (0.0, 1.1, 0.0),
        ris_pos: Sequence[float] = (0.0, 0.0, 0.0),
        p_tx_w: float = 1.0,
        sigma2_w: float = 1e-10,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.freq_hz = float(freq_hz)
        self.ris_nx = int(ris_nx)
        self.ris_ny = int(ris_ny)
        self.ris_bits = int(ris_bits)
        self.tx_pos = np.asarray(tx_pos, dtype=float).reshape(3)
        self.ris_pos = np.asarray(ris_pos, dtype=float).reshape(3)
        self.p_tx_w = float(p_tx_w)
        self.sigma2_w = float(sigma2_w)

        self.server_ctrl = RISServerController(ip=self.server_ip, port=self.server_port)
        self._update_geometry()

    def _update_geometry(self) -> None:
        """Pre-calcula la malla 3D de celdas de la RIS y el canal incidente TX -> RIS (G)."""
        self.wavelength = 3e8 / self.freq_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        self.dx = self.wavelength / 2.0
        self.dy = self.wavelength / 2.0
        n_total = self.ris_nx * self.ris_ny

        x_grid = (np.arange(self.ris_nx) - (self.ris_nx - 1) / 2.0) * self.dx + self.ris_pos[0]
        z_grid = (np.arange(self.ris_ny) - (self.ris_ny - 1) / 2.0) * self.dy + self.ris_pos[2]
        xx, zz = np.meshgrid(x_grid, z_grid)
        self.elem_coords = np.column_stack([xx.ravel(), np.full(n_total, self.ris_pos[1]), zz.ravel()])

        vec_tx = self.tx_pos - self.elem_coords
        d_tx = np.linalg.norm(vec_tx, axis=1)
        cos_tx = np.clip(np.abs(vec_tx[:, 1]) / np.maximum(d_tx, 1e-9), 1e-6, 1.0)
        self.G_channel = (
            (np.sqrt(self.dx * self.dy * cos_tx) / (4.0 * np.pi * d_tx))
            * np.exp(-1j * self.k0 * d_tx)
        )[:, None]

    def set_endpoint(self, server_ip: str, server_port: int) -> None:
        self.server_ip = server_ip.strip()
        self.server_port = int(server_port)
        self.server_ctrl.addr = (self.server_ip, self.server_port)

    def set_scene_parameters(
        self,
        tx_pos: Optional[Sequence[float]] = None,
        freq_hz: Optional[float] = None,
    ) -> None:
        needs_rebuild = False
        if tx_pos is not None:
            new_tx = np.asarray(tx_pos, dtype=float).reshape(3)
            if not np.allclose(new_tx, self.tx_pos):
                self.tx_pos = new_tx
                needs_rebuild = True
        if freq_hz is not None and abs(float(freq_hz) - self.freq_hz) > 1.0:
            self.freq_hz = float(freq_hz)
            needs_rebuild = True
        if needs_rebuild:
            self._update_geometry()

    def _quantize_phases(self, phases_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phases_mod = np.mod(phases_rad, 2.0 * np.pi)
        if self.ris_bits <= 0:
            return phases_mod, np.zeros_like(phases_mod, dtype=int)
        levels = 2 ** self.ris_bits
        step_rad = (2.0 * np.pi) / levels
        states_int = np.round(phases_mod / step_rad).astype(int) % levels
        phases_q_rad = states_int * step_rad
        return phases_q_rad, states_int

    def _compute_ris_to_rx_channel(self, rx_positions: np.ndarray) -> np.ndarray:
        """Evalúa el canal esférico de campo cercano (Tang General) desde cada celda a cada RX."""
        if rx_positions.ndim == 1:
            rx_positions = rx_positions[None, :]
        cols = []
        for rx_pos in rx_positions:
            vec_rx = rx_pos - self.elem_coords
            d_rx = np.linalg.norm(vec_rx, axis=1)
            cos_rx = np.clip(np.abs(vec_rx[:, 1]) / np.maximum(d_rx, 1e-9), 1e-6, 1.0)
            h_rk = (
                (np.sqrt(self.dx * self.dy * cos_rx) / (4.0 * np.pi * d_rx))
                * np.exp(-1j * self.k0 * d_rx)
            )
            cols.append(h_rk)
        return np.column_stack(cols)

    def compute_states_for_coordinates(
        self,
        rx_coords: Sequence[float] | np.ndarray,
        tx_pos: Optional[Sequence[float]] = None,
        freq_hz: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        max_iter: int = 50,
    ) -> dict:
        """
        Calcula la matriz 40x40 de estados de la RIS para una coordenada RX [x, y, z]
        (mediante SCRPA) o una lista de coordenadas RX [[x1, y1, z1], ...] (mediante WMMSE-MM).
        """
        t0 = time.perf_counter()
        self.set_scene_parameters(tx_pos=tx_pos, freq_hz=freq_hz)

        rx_arr = np.asarray(rx_coords, dtype=float)
        if rx_arr.ndim == 1:
            if rx_arr.size != 3:
                raise ValueError("Las coordenadas RX deben tener 3 valores: [x, y, z] en metros.")
            rx_arr = rx_arr.reshape(1, 3)
        elif rx_arr.ndim == 2 and rx_arr.shape[1] == 3:
            pass
        else:
            raise ValueError("Formato de coordenadas inválido. Usa [x, y, z] o [[x1, y1, z1], ...].")

        H_r = self._compute_ris_to_rx_channel(rx_arr)
        N, K = H_r.shape
        pesos = np.ones(K, dtype=float) if weights is None else np.asarray(weights, dtype=float)

        # Canal en cascada: A[:, k] = h_{r,k}^* * g
        A = H_r.conj() * self.G_channel[:, 0:1]

        if K == 1:
            fases_opt_cont = np.mod(-np.angle(A[:, 0]), 2.0 * np.pi)
            fases_q_rad, estados_int = self._quantize_phases(fases_opt_cont)
            theta_q = np.exp(1j * fases_q_rad)
            h_eff = np.vdot(A[:, 0], theta_q)
            p_rx_w = self.p_tx_w * (np.abs(h_eff) ** 2)
            p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))
            wsr = float(pesos[0] * np.log2(1.0 + p_rx_w / self.sigma2_w))
        else:
            theta = np.exp(-1j * np.angle(A @ pesos))
            historico_wsr = []
            for _ in range(max_iter):
                h_eff_vec = np.array([np.vdot(A[:, k], theta) for k in range(K)])
                snr = (self.p_tx_w * (np.abs(h_eff_vec) ** 2)) / self.sigma2_w
                wsr = float(np.sum(pesos * np.log2(1.0 + snr)))
                historico_wsr.append(wsr)
                if len(historico_wsr) > 1 and abs(historico_wsr[-1] - historico_wsr[-2]) < 1e-6:
                    break

                rx_sig = np.sqrt(self.p_tx_w) * h_eff_vec
                u = rx_sig / (np.abs(rx_sig) ** 2 + self.sigma2_w)
                mse = np.maximum(1.0 - np.real(u.conj() * rx_sig), 1e-12)
                rho = pesos / mse

                U = np.zeros((N, N), dtype=complex)
                v = np.zeros(N, dtype=complex)
                for k in range(K):
                    a_k = A[:, k].conj()
                    U += rho[k] * (np.abs(u[k]) ** 2) * self.p_tx_w * np.outer(a_k, a_k.conj())
                    v += rho[k] * (u[k].conj() * np.sqrt(self.p_tx_w)) * a_k

                U = 0.5 * (U + U.conj().T)
                lambda_max = float(np.max(np.linalg.eigvalsh(U)))
                for _ in range(25):
                    z = (lambda_max * theta - U @ theta) + v
                    theta = np.exp(1j * np.angle(z))

            fases_opt_cont = np.mod(np.angle(theta), 2.0 * np.pi)
            fases_q_rad, estados_int = self._quantize_phases(fases_opt_cont)
            theta_q = np.exp(1j * fases_q_rad)
            h_eff_vec = np.array([np.vdot(A[:, k], theta_q) for k in range(K)])
            p_rx_w = float(np.mean(self.p_tx_w * (np.abs(h_eff_vec) ** 2)))
            p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))

        estados_2d = estados_int.reshape((self.ris_nx, self.ris_ny))
        config = RISConfig(state_matrix_v=estados_2d)
        gw_pattern = self.server_ctrl.build_gw_pattern(config)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "rx_coords": rx_arr,
            "tx_pos": self.tx_pos.copy(),
            "freq_hz": self.freq_hz,
            "estados_40x40": estados_2d,
            "fases_deg": np.rad2deg(fases_q_rad).reshape((self.ris_nx, self.ris_ny)),
            "gw_pattern": gw_pattern,
            "p_rx_dbm": p_rx_dbm,
            "wsr_bps_hz": wsr,
            "latencia_ms": elapsed_ms,
        }

    def apply_coordinates(
        self,
        rx_coords: Sequence[float] | np.ndarray,
        tx_pos: Optional[Sequence[float]] = None,
        freq_hz: Optional[float] = None,
        send_udp: bool = True,
        save_csv_path: Optional[str] = None,
    ) -> dict:
        """
        Calcula la configuración óptima de la RIS para `rx_coords` y la envía por UDP
        al servidor `ris_hardware_server.py` usando `{"cmd": "SET_CONFIG", "data": gw_pattern}`.
        """
        t0 = time.perf_counter()
        res = self.compute_states_for_coordinates(rx_coords=rx_coords, tx_pos=tx_pos, freq_hz=freq_hz)
        config = RISConfig(state_matrix_v=res["estados_40x40"])

        if save_csv_path:
            np.savetxt(save_csv_path, config.state_matrix_v, fmt="%d", delimiter=",")

        if send_udp:
            self.server_ctrl.addr = (self.server_ip, self.server_port)
            self.server_ctrl.apply_config(config)

        res["latencia_ms"] = (time.perf_counter() - t0) * 1000.0
        res["udp_sent"] = send_udp
        res["server_addr"] = f"{self.server_ip}:{self.server_port}"
        return res

    def apply_uniform_state(self, state_val: int = 0, send_udp: bool = True) -> dict:
        """Aplica un estado uniforme (todo 0 o todo 1) a las 1600 celdas (3200 bits)."""
        bit = 1 if int(state_val) != 0 else 0
        estados_2d = np.full((self.ris_nx, self.ris_ny), bit, dtype=int)
        config = RISConfig(state_matrix_v=estados_2d)
        gw_pattern = self.server_ctrl.build_gw_pattern(config)
        if send_udp:
            self.server_ctrl.addr = (self.server_ip, self.server_port)
            self.server_ctrl.apply_config(config)
        return {
            "estados_40x40": estados_2d,
            "gw_pattern": gw_pattern,
            "udp_sent": send_udp,
            "server_addr": f"{self.server_ip}:{self.server_port}",
        }

    def turn_on(self) -> None:
        self.server_ctrl.addr = (self.server_ip, self.server_port)
        self.server_ctrl.connect()

    def turn_off(self) -> None:
        self.server_ctrl.addr = (self.server_ip, self.server_port)
        self.server_ctrl.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula y envía configuración RIS a partir de coordenadas 3D.")
    parser.add_argument("--rx", nargs=3, type=float, default=[-0.5, 1.4, 0.0], metavar=("X", "Y", "Z"),
                        help="Coordenadas 3D del receptor RX en metros (ej: -0.5 1.4 0.0)")
    parser.add_argument("--tx", nargs=3, type=float, default=[0.0, 1.1, 0.0], metavar=("X", "Y", "Z"),
                        help="Coordenadas 3D del transmisor TX en metros (ej: 0.0 1.1 0.0)")
    parser.add_argument("--freq-ghz", type=float, default=28.0, help="Frecuencia de operación en GHz")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP de ris_hardware_server.py")
    parser.add_argument("--port", type=int, default=17000, help="Puerto UDP de ris_hardware_server.py")
    parser.add_argument("--csv", type=str, default=None, help="Ruta opcional para guardar la matriz 40x40 en CSV")
    args = parser.parse_args()

    ctrl = RISCoordinateController(
        server_ip=args.ip,
        server_port=args.port,
        freq_hz=args.freq_ghz * 1e9,
        tx_pos=args.tx,
    )
    info = ctrl.apply_coordinates(args.rx, save_csv_path=args.csv)
    print(
        f"✓ RIS configurada para RX={args.rx} m | "
        f"Estados enviados={len(info['gw_pattern'])} a {info['server_addr']} | "
        f"Prx est={info['p_rx_dbm']:.2f} dBm | Tiempo={info['latencia_ms']:.2f} ms"
    )
