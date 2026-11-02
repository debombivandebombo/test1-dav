"""
Controlador externo para generar configuraciones `RISConfig` (40x40 dual-pol, 3200 estados)
a partir de coordenadas 3D [x, y, z] utilizando las clases nativas de `surpRISe` (`RIS`, `Antenna`,
`tang_general_channel`) y enviarlas a `ris_hardware_server.py` (en `ris-fr2-sandbox`).

Cumple exactamente el formato y flujo descritos en `ris-fr2-sandbox/README.md` y `ris_hardware_server.py`:
1. Protocolo UDP (`socket.SOCK_DGRAM`) en `127.0.0.1:17000` (configurable).
2. Comandos JSON UTF-8:
   - `{"cmd": "ON", "data": null}` (`controller.connect()`)
   - `{"cmd": "OFF", "data": null}` (`controller.disconnect()`)
   - `{"cmd": "SET_CONFIG", "data": gw_pattern}` (`controller.apply_config(config)`)
3. `RISConfig(state_matrix_v=..., state_matrix_h=...)`:
   - `v_flat = config.state_matrix_v[::-1, :].T.flatten().astype(int)`
   - `h_flat = config.state_matrix_h[::-1, :].T.flatten().astype(int)`
   - `gw_pattern[0::2] = v_flat.tolist()`
   - `gw_pattern[1::2] = h_flat.tolist()`
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

# Añadimos `surpRISe/src` al path para usar las clases nativas del proyecto
_SURPRISE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "surpRISe", "src"))
if os.path.isdir(_SURPRISE_SRC) and _SURPRISE_SRC not in sys.path:
    sys.path.insert(0, _SURPRISE_SRC)

try:
    from surpRISe.components.antenna import Antenna
    from surpRISe.components.ris import RIS
    from surpRISe.channels.tx2ris2rx.tang_general_channel import tang_general_channel
    _HAS_SURPRISE = True
except ImportError:
    _HAS_SURPRISE = False


# =============================================================================
# DEFINICIÓN EXACTA DEL README DE `ris-fr2-sandbox`
# =============================================================================
@dataclass
class RISConfig:
    state_matrix_v: np.ndarray  # Vertical polarization state matrix (i.e. 40x40)
    state_matrix_h: np.ndarray  # Horizontal polarization state matrix (i.e. 40x40)

    def __init__(
        self,
        state_matrix_v: np.ndarray,
        state_matrix_h: Optional[np.ndarray] = None,
    ) -> None:
        self.state_matrix_v = np.asarray(state_matrix_v, dtype=int)
        if state_matrix_h is None:
            self.state_matrix_h = self.state_matrix_v.copy()
        else:
            self.state_matrix_h = np.asarray(state_matrix_h, dtype=int)


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
        v_flat = config.state_matrix_v[::-1, :].T.flatten().astype(int)
        h_flat = config.state_matrix_h[::-1, :].T.flatten().astype(int)

        gw_pattern = [0] * (v_flat.size + h_flat.size)
        gw_pattern[0::2] = v_flat.tolist()
        gw_pattern[1::2] = h_flat.tolist()

        # Apply the config to the RIS
        self._send_command("SET_CONFIG", gw_pattern)
        return gw_pattern

    def _send_command(self, cmd: str, data=None) -> None:
        msg = {"cmd": cmd, "data": data}
        self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)


# =============================================================================
# CONTROLADOR DE COORDENADAS 3D -> RISConfig -> RISServerController
# =============================================================================
class RISCoordinateController:
    """
    Calcula la matriz 40x40 de estados de la RIS (`state_matrix_v` y `state_matrix_h`)
    usando el modelo físico de `surpRISe` (`RIS`, `Antenna`, `tang_general_channel`)
    para 1 usuario o WMMSE-MM para multiusuario, empaqueta en `RISConfig` y lo envía
    a `ris_hardware_server.py` mediante `RISServerController`.
    """

    C_LIGHT = 299792458.0

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
        boresight: str = "auto",
        optimizer: str = "1d_search",
        polarization_mode: str = "vh",
        tx_gain_dbi: float = 15.0,
        rx_gain_dbi: float = 15.0,
        p_tx_w: float = 1.0,
        sigma2_w: float = 1e-10,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.freq_hz = float(freq_hz)
        self.ris_nx = int(ris_nx)  # Ncol (40)
        self.ris_ny = int(ris_ny)  # Nrow (40)
        self.ris_bits = int(ris_bits)
        self.tx_pos = np.asarray(tx_pos, dtype=float).reshape(3)
        self.ris_pos = np.asarray(ris_pos, dtype=float).reshape(3)
        self.boresight = boresight.lower().strip()
        self.optimizer = optimizer.lower().strip()
        self.polarization_mode = polarization_mode.lower().strip()
        self.tx_gain_dbi = float(tx_gain_dbi)
        self.rx_gain_dbi = float(rx_gain_dbi)
        self.p_tx_w = float(p_tx_w)
        self.sigma2_w = float(sigma2_w)

        self.server_ctrl = RISServerController(ip=self.server_ip, port=self.server_port)
        self._is_connected = False
        self._update_geometry()

    def _resolve_boresight(self) -> str:
        """
        Determina el eje normal ('x', 'y' o 'z') de la RIS.
        Si `self.boresight == 'auto'`, elige el eje con mayor componente en `tx_pos - ris_pos`.
        """
        if self.boresight in ("x", "y", "z"):
            return self.boresight
        diff = np.abs(self.tx_pos - self.ris_pos)
        idx = int(np.argmax(diff))
        return ("x", "y", "z")[idx]

    @staticmethod
    def _build_element_positions(
        center: np.ndarray,
        boresight: str,
        ncol: int,
        nrow: int,
        delta_h: float,
        delta_v: float,
    ) -> np.ndarray:
        """
        Genera exactamente las mismas posiciones 3D (Nrow, Ncol, 3) que `RIS._generate_positions()` en `surpRISe`:
        - Fila 0 arriba (-r_off), Fila Nrow-1 abajo.
        - Columnas de izquierda a derecha según el plano perpendicular al boresight.
        """
        col_indices = np.arange(ncol) - (ncol - 1) / 2.0
        row_indices = np.arange(nrow) - (nrow - 1) / 2.0
        col_offsets = col_indices * delta_h
        row_offsets = row_indices * delta_v
        c_off, r_off = np.meshgrid(col_offsets, row_offsets)

        positions = np.zeros((nrow, ncol, 3), dtype=float)
        if boresight == "x":
            positions[:, :, 0] = center[0]
            positions[:, :, 1] = center[1] + c_off
            positions[:, :, 2] = center[2] - r_off
        elif boresight == "y":
            positions[:, :, 0] = center[0] - c_off
            positions[:, :, 1] = center[1]
            positions[:, :, 2] = center[2] - r_off
        else:  # "z"
            positions[:, :, 0] = center[0] + c_off
            positions[:, :, 1] = center[1] - r_off
            positions[:, :, 2] = center[2]
        return positions

    def _update_geometry(self) -> None:
        """Construye la instancia `RIS` de `surpRISe` y precalcula el canal TX -> RIS."""
        self.wavelength = self.C_LIGHT / self.freq_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        self.delta_h = self.wavelength / 2.0
        self.delta_v = self.wavelength / 2.0
        self.dx = 0.47 * self.wavelength
        self.dy = 0.47 * self.wavelength
        self.active_boresight = self._resolve_boresight()

        levels = max(2 ** self.ris_bits, 2)
        step_rad = (2.0 * np.pi) / levels
        phase_shifts_dict = {str(i): i * step_rad for i in range(levels)}
        att_dict = {str(i): 1.0 for i in range(levels)}

        if _HAS_SURPRISE:
            self.ris_obj = RIS(
                center=self.ris_pos.tolist(),
                boresight=self.active_boresight,
                Ncol=self.ris_nx,
                Nrow=self.ris_ny,
                delta_h=self.delta_h,
                delta_v=self.delta_v,
                dx=self.dx,
                dy=self.dy,
                fc=self.freq_hz,
                radiation_factor=1.0,
                phase_shifts_dict=phase_shifts_dict,
                att_dict=att_dict,
            )
            self.tx_antenna = Antenna(position=self.tx_pos.tolist(), gain_dbi=self.tx_gain_dbi)
            self.channel_model = tang_general_channel()
            self.ris_positions_3d = self.ris_obj.positions.copy()  # (Nrow, Ncol, 3)
            self.normal_vec = self.ris_obj.normal.copy()
        else:
            self.ris_obj = None
            self.tx_antenna = None
            self.channel_model = None
            self.ris_positions_3d = self._build_element_positions(
                center=self.ris_pos,
                boresight=self.active_boresight,
                ncol=self.ris_nx,
                nrow=self.ris_ny,
                delta_h=self.delta_h,
                delta_v=self.delta_v,
            )
            normal_map = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
            }
            self.normal_vec = normal_map[self.active_boresight]

        # Aplanado por filas (Nrow * Ncol, 3) coherente con `ris.positions`
        self.elem_coords = self.ris_positions_3d.reshape(-1, 3)

        vec_tx = self.tx_pos[None, :] - self.elem_coords
        d_tx = np.linalg.norm(vec_tx, axis=1)
        cos_tx = np.clip(np.abs(vec_tx @ self.normal_vec) / np.maximum(d_tx, 1e-9), 1e-6, 1.0)
        sqrt_g_tx = np.sqrt(10.0 ** (self.tx_gain_dbi / 10.0))
        self.G_channel = (
            (sqrt_g_tx * np.sqrt(self.dx * self.dy * cos_tx) / (4.0 * np.pi * d_tx))
            * np.exp(-1j * self.k0 * d_tx)
        )[:, None]

    def set_endpoint(self, server_ip: str, server_port: int) -> None:
        new_addr = (server_ip.strip(), int(server_port))
        if new_addr != self.server_ctrl.addr:
            self._is_connected = False
        self.server_ip, self.server_port = new_addr
        self.server_ctrl.addr = new_addr

    def set_scene_parameters(
        self,
        tx_pos: Optional[Sequence[float]] = None,
        freq_hz: Optional[float] = None,
        boresight: Optional[str] = None,
        optimizer: Optional[str] = None,
        polarization_mode: Optional[str] = None,
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
        if boresight is not None and boresight.lower().strip() != self.boresight:
            self.boresight = boresight.lower().strip()
            needs_rebuild = True
        if optimizer is not None:
            self.optimizer = optimizer.lower().strip()
        if polarization_mode is not None:
            self.polarization_mode = polarization_mode.lower().strip()
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
        sqrt_g_rx = np.sqrt(10.0 ** (self.rx_gain_dbi / 10.0))
        cols = []
        for rx_pos in rx_positions:
            vec_rx = rx_pos[None, :] - self.elem_coords
            d_rx = np.linalg.norm(vec_rx, axis=1)
            cos_rx = np.clip(np.abs(vec_rx @ self.normal_vec) / np.maximum(d_rx, 1e-9), 1e-6, 1.0)
            h_rk = (
                (sqrt_g_rx * np.sqrt(self.dx * self.dy * cos_rx) / (4.0 * np.pi * d_rx))
                * np.exp(-1j * self.k0 * d_rx)
            )
            cols.append(h_rk)
        return np.column_stack(cols)

    def _build_ris_config(
        self,
        estados_2d: np.ndarray,
        state_matrix_h: Optional[np.ndarray] = None,
    ) -> RISConfig:
        """
        Construye el objeto `RISConfig(state_matrix_v=..., state_matrix_h=...)` según el modo
        de polarización configurado ('vh', 'v_only', 'h_only') o la matriz H explícita.
        """
        mat_base = np.asarray(estados_2d, dtype=int)
        if state_matrix_h is not None:
            return RISConfig(state_matrix_v=mat_base, state_matrix_h=np.asarray(state_matrix_h, dtype=int))

        if self.polarization_mode == "v_only":
            mat_v = mat_base
            mat_h = np.zeros_like(mat_base, dtype=int)
        elif self.polarization_mode == "h_only":
            mat_v = np.zeros_like(mat_base, dtype=int)
            mat_h = mat_base
        else:  # "vh" (ambas polarizaciones activas con la misma configuración)
            mat_v = mat_base
            mat_h = mat_base.copy()

        return RISConfig(state_matrix_v=mat_v, state_matrix_h=mat_h)

    def compute_states_for_coordinates(
        self,
        rx_coords: Sequence[float] | np.ndarray,
        tx_pos: Optional[Sequence[float]] = None,
        freq_hz: Optional[float] = None,
        boresight: Optional[str] = None,
        optimizer: Optional[str] = None,
        polarization_mode: Optional[str] = None,
        weights: Optional[Sequence[float]] = None,
        max_iter: int = 50,
    ) -> dict:
        """
        Calcula la matriz 40x40 de estados de la RIS (`Nrow x Ncol`) y el `RISConfig`
        correspondiente para una coordenada RX `[x, y, z]` (usando `tang_general_channel.optimize`
        de `surpRISe`) o para múltiples usuarios `[[x1, y1, z1], ...]` (usando WMMSE-MM).
        """
        t0 = time.perf_counter()
        self.set_scene_parameters(
            tx_pos=tx_pos,
            freq_hz=freq_hz,
            boresight=boresight,
            optimizer=optimizer,
            polarization_mode=polarization_mode,
        )

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

        # Canal físico en cascada por celda: c_{n,k} = h_{r,n,k} * g_n ~ exp(-j*k0*(d_tx + d_rx))
        # Definimos A[:, k] = (h_{r,k} * g)^* ~ exp(+j*k0*(d_tx + d_rx)) para que
        # np.vdot(A[:, k], theta) = sum_n h_{r,n,k} * g_n * theta_n sea exactamente el canal físico.
        A = (H_r * self.G_channel[:, 0:1]).conj()

        if K == 1 and _HAS_SURPRISE and self.ris_bits == 1:
            rx_antenna = Antenna(position=rx_arr[0].tolist(), gain_dbi=self.rx_gain_dbi)
            opt_name = self.optimizer if self.optimizer in (
                "direct_disc", "iterative", "1d_search", "grid_search", "chen_alg1"
            ) else "1d_search"
            self.channel_model.optimize(
                optimizer=opt_name,
                mode="disc",
                wave_model="spherical",
                tx_antenna=self.tx_antenna,
                rx_antenna=rx_antenna,
                ris=self.ris_obj,
                verbose=False,
                fc=self.freq_hz,
            )
            estados_2d = self.ris_obj.get_all_state_indices().astype(int)  # (Nrow, Ncol) = (40, 40)
            fases_q_2d = self.ris_obj.elem_phase_shift.copy()
            theta_q = np.exp(1j * fases_q_2d.reshape(-1))

            pg = float(
                self.channel_model.path_gain_multrx(
                    tx_antenna=self.tx_antenna,
                    rx_antenna=rx_antenna,
                    rx_positions=rx_arr,
                    ris=self.ris_obj,
                )[0]
            )
            p_rx_w = self.p_tx_w * pg
            p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))
            wsr = float(pesos[0] * np.log2(1.0 + p_rx_w / self.sigma2_w))
        elif K == 1:
            # Alineación de fase esférica exacta: phi_n = +k0 * (d_tx + d_rx) mod 2*pi
            fases_opt_cont = np.mod(np.angle(A[:, 0]), 2.0 * np.pi)
            fases_q_rad, estados_int = self._quantize_phases(fases_opt_cont)
            estados_2d = estados_int.reshape((self.ris_ny, self.ris_nx))
            fases_q_2d = fases_q_rad.reshape((self.ris_ny, self.ris_nx))
            theta_q = np.exp(1j * fases_q_rad)
            h_eff = np.vdot(A[:, 0], theta_q)
            p_rx_w = self.p_tx_w * (np.abs(h_eff) ** 2)
            p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))
            wsr = float(pesos[0] * np.log2(1.0 + p_rx_w / self.sigma2_w))
        else:
            # Multiusuario WMMSE-MM con el canal físico A = (H_r * G)^*
            theta = np.exp(1j * np.angle(A @ pesos))
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
                    a_k = A[:, k]
                    U += rho[k] * (np.abs(u[k]) ** 2) * self.p_tx_w * np.outer(a_k, a_k.conj())
                    v += rho[k] * (u[k].conj() * np.sqrt(self.p_tx_w)) * a_k

                U = 0.5 * (U + U.conj().T)
                lambda_max = float(np.max(np.linalg.eigvalsh(U)))
                for _ in range(25):
                    z = (lambda_max * theta - U @ theta) + v
                    theta = np.exp(1j * np.angle(z))

            fases_opt_cont = np.mod(np.angle(theta), 2.0 * np.pi)
            if _HAS_SURPRISE and self.ris_obj is not None:
                self.ris_obj.set_all_shift(fases_opt_cont.reshape((self.ris_ny, self.ris_nx)), mode="disc")
                estados_2d = self.ris_obj.get_all_state_indices().astype(int)
                fases_q_2d = self.ris_obj.elem_phase_shift.copy()
                theta_q = np.exp(1j * fases_q_2d.reshape(-1))
            else:
                fases_q_rad, estados_int = self._quantize_phases(fases_opt_cont)
                estados_2d = estados_int.reshape((self.ris_ny, self.ris_nx))
                fases_q_2d = fases_q_rad.reshape((self.ris_ny, self.ris_nx))
                theta_q = np.exp(1j * fases_q_rad)

            h_eff_vec = np.array([np.vdot(A[:, k], theta_q) for k in range(K)])
            p_rx_w = float(np.mean(self.p_tx_w * (np.abs(h_eff_vec) ** 2)))
            p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))

        config = self._build_ris_config(estados_2d)
        gw_pattern = self.server_ctrl.build_gw_pattern(config)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "rx_coords": rx_arr,
            "tx_pos": self.tx_pos.copy(),
            "freq_hz": self.freq_hz,
            "boresight": self.active_boresight,
            "optimizer": self.optimizer if K == 1 else "wmmse_mm",
            "polarization_mode": self.polarization_mode,
            "ris_config": config,
            "estados_40x40": config.state_matrix_v,
            "estados_h_40x40": config.state_matrix_h,
            "fases_deg": np.rad2deg(fases_q_2d),
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
        boresight: Optional[str] = None,
        optimizer: Optional[str] = None,
        polarization_mode: Optional[str] = None,
        send_udp: bool = True,
        ensure_on: bool = True,
    ) -> dict:
        """
        Calcula el `RISConfig` óptimo para `rx_coords` y lo envía por UDP a `ris_hardware_server.py`
        siguiendo el flujo del README de `ris-fr2-sandbox`:
        1. `controller.connect()` -> `{"cmd": "ON", "data": null}` (si `ensure_on=True`)
        2. `controller.apply_config(config)` -> `{"cmd": "SET_CONFIG", "data": gw_pattern}`
        """
        t0 = time.perf_counter()
        res = self.compute_states_for_coordinates(
            rx_coords=rx_coords,
            tx_pos=tx_pos,
            freq_hz=freq_hz,
            boresight=boresight,
            optimizer=optimizer,
            polarization_mode=polarization_mode,
        )
        config: RISConfig = res["ris_config"]

        if send_udp:
            self.server_ctrl.addr = (self.server_ip, self.server_port)
            if ensure_on and not self._is_connected:
                self.server_ctrl.connect()
                self._is_connected = True
            self.server_ctrl.apply_config(config)

        res["latencia_ms"] = (time.perf_counter() - t0) * 1000.0
        res["udp_sent"] = send_udp
        res["server_addr"] = f"{self.server_ip}:{self.server_port}"
        return res

    def apply_uniform_state(
        self,
        state_val: int = 0,
        send_udp: bool = True,
        ensure_on: bool = True,
    ) -> dict:
        """Aplica un estado uniforme (todo 0 o todo 1) a `state_matrix_v` y `state_matrix_h`."""
        bit = 1 if int(state_val) != 0 else 0
        estados_2d = np.full((self.ris_ny, self.ris_nx), bit, dtype=int)
        config = RISConfig(state_matrix_v=estados_2d, state_matrix_h=estados_2d.copy())
        gw_pattern = self.server_ctrl.build_gw_pattern(config)
        if send_udp:
            self.server_ctrl.addr = (self.server_ip, self.server_port)
            if ensure_on and not self._is_connected:
                self.server_ctrl.connect()
                self._is_connected = True
            self.server_ctrl.apply_config(config)
        return {
            "ris_config": config,
            "estados_40x40": config.state_matrix_v,
            "estados_h_40x40": config.state_matrix_h,
            "gw_pattern": gw_pattern,
            "udp_sent": send_udp,
            "server_addr": f"{self.server_ip}:{self.server_port}",
        }

    def turn_on(self) -> None:
        self.server_ctrl.addr = (self.server_ip, self.server_port)
        self.server_ctrl.connect()
        self._is_connected = True

    def turn_off(self) -> None:
        self.server_ctrl.addr = (self.server_ip, self.server_port)
        self.server_ctrl.disconnect()
        self._is_connected = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula y envía RISConfig a partir de coordenadas 3D.")
    parser.add_argument("--rx", nargs=3, type=float, default=[-0.5, 1.4, 0.0], metavar=("X", "Y", "Z"),
                        help="Coordenadas 3D del receptor RX en metros (ej: -0.5 1.4 0.0)")
    parser.add_argument("--tx", nargs=3, type=float, default=[0.0, 1.1, 0.0], metavar=("X", "Y", "Z"),
                        help="Coordenadas 3D del transmisor TX en metros (ej: 0.0 1.1 0.0)")
    parser.add_argument("--boresight", type=str, default="auto", choices=["auto", "x", "y", "z"],
                        help="Eje normal de la RIS ('auto', 'x', 'y', 'z')")
    parser.add_argument("--optimizer", type=str, default="1d_search",
                        choices=["1d_search", "iterative", "direct_disc", "grid_search", "chen_alg1"],
                        help="Optimizador de fase de surpRISe")
    parser.add_argument("--pol", type=str, default="vh", choices=["vh", "v_only", "h_only"],
                        help="Modo de polarización en RISConfig (vh, v_only, h_only)")
    parser.add_argument("--freq-ghz", type=float, default=28.0, help="Frecuencia de operación en GHz")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP de ris_hardware_server.py")
    parser.add_argument("--port", type=int, default=17000, help="Puerto UDP de ris_hardware_server.py")
    args = parser.parse_args()

    ctrl = RISCoordinateController(
        server_ip=args.ip,
        server_port=args.port,
        freq_hz=args.freq_ghz * 1e9,
        tx_pos=args.tx,
        boresight=args.boresight,
        optimizer=args.optimizer,
        polarization_mode=args.pol,
    )
    info = ctrl.apply_coordinates(args.rx)
    print(
        f"✓ RIS configurada para RX={args.rx} m (boresight='{info['boresight']}', opt='{info['optimizer']}') | "
        f"Estados enviados={len(info['gw_pattern'])} a {info['server_addr']} | "
        f"Prx est={info['p_rx_dbm']:.2f} dBm | Tiempo={info['latencia_ms']:.2f} ms"
    )
