"""Atomic row checkpoints and notebook-compatible completed row files."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import numpy as np
from .config import legacy_conf, validate


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


class RowStore:
    def __init__(self, folder, config, frequencies, identity, simulated, calibration):
        self.config = validate(config)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stem = f"setup_{config['setup']}_nv_{config['row']}"
        self.path = self.folder / (self.stem + '.partial.npz')
        self.final = self.folder / (self.stem + '.npz')
        if self.path.exists() or self.final.exists():
            raise FileExistsError('This row already exists. Resume its checkpoint or choose a new setup/row.')
        self.freq = np.asarray(frequencies, dtype=np.float64)
        n = config['sweeps'] if config['average'] == 1 else 1
        self.data = np.zeros((config['nh'], n, len(config['sparams']), len(self.freq)), dtype=np.complex64)
        self.mask = np.zeros(self.data.shape[:3], dtype=bool)
        self.timestamps = np.full((n, config['nh']), '', dtype='<U40')
        self.identity, self.simulated = identity, simulated
        self.calibration = deepcopy(calibration)
        self.save()

    @property
    def complete_points(self):
        return self.mask[:, :, :3].all(axis=(1, 2))

    @property
    def complete(self):
        return bool(self.complete_points.all() and self.mask[0].all() and self.mask[-1].all())

    def put(self, point, sweep, values):
        count = self.data.shape[2] if point in (0, self.data.shape[0] - 1) else 3
        values = np.asarray(values)
        if values.shape != (count, len(self.freq)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid acquisition shape or nonfinite samples: expected {(count, len(self.freq))}.')
        old_values = self.data[point, sweep].copy()
        old_mask = self.mask[point, sweep].copy()
        old_timestamp = self.timestamps[sweep, point]
        self.data[point, sweep, :count] = values
        self.mask[point, sweep, :count] = True
        self.timestamps[sweep, point] = timestamp()
        try:
            self.save()
        except Exception:
            # A failed disk write must not make a retry skip an unsaved sweep.
            self.data[point, sweep] = old_values
            self.mask[point, sweep] = old_mask
            self.timestamps[sweep, point] = old_timestamp
            raise

    def save(self):
        c = self.config
        meta = deepcopy(c['metadata'])
        meta['bs_params'].update(num_rows=c['nv'], num_cols=c['nh'], vertical_spacing=c['dv'], horizontal_spacing=c['dh'])
        target = self.final if self.complete else self.path
        temp = self.folder / ('.' + self.stem + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('xb') as stream:
                np.savez(stream, row_sparams=self.data, measured_mask=self.mask, freq_vector=self.freq,
                         timestamp_list=self.timestamps, conf_params=legacy_conf(c),
                         bs_params=meta['bs_params'], ue0_params=meta['ue0_params'],
                         ue1_params=meta['ue1_params'], ue2_params=meta['ue2_params'],
                         config_json=json.dumps(c, allow_nan=False), schema_version=1,
                         instrument_identity=self.identity, simulated=self.simulated,
                         calibration_json=json.dumps(self.calibration), va_vert_pos=c['va_vert_pos'])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        if self.complete:
            self.path.unlink(missing_ok=True)
        return target

    @classmethod
    def resume(cls, path):
        path = Path(path)
        if not path.name.endswith('.partial.npz'):
            raise ValueError('Choose an incomplete .partial.npz checkpoint. Completed rows are read-only.')
        with np.load(path, allow_pickle=False) as z:
            if int(z['schema_version']) != 1:
                raise ValueError('Unsupported checkpoint version.')
            row = cls.__new__(cls)
            row.config = validate(json.loads(str(z['config_json'])))
            row.folder, row.path = path.parent, path
            row.stem = f"setup_{row.config['setup']}_nv_{row.config['row']}"
            row.final = path.parent / (row.stem + '.npz')
            if path.name != row.stem + '.partial.npz' or row.final.exists():
                raise ValueError('Checkpoint name is inconsistent or a completed row already exists.')
            row.freq = z['freq_vector'].copy()
            row.data, row.mask = z['row_sparams'].copy(), z['measured_mask'].copy()
            row.timestamps = z['timestamp_list'].copy()
            row.identity, row.simulated = str(z['instrument_identity']), bool(z['simulated'])
            row.calibration = json.loads(str(z['calibration_json']))
        c = row.config
        n = c['sweeps'] if c['average'] == 1 else 1
        expected = (c['nh'], n, len(c['sparams']), sum(int(s[2]) for s in c['segments']))
        if (row.data.shape != expected or row.mask.shape != expected[:3]
                or row.freq.shape != (expected[-1],) or row.timestamps.shape != (n, c['nh'])
                or row.mask.dtype != bool or not np.isfinite(row.data).all()):
            raise ValueError('Checkpoint dimensions or data are invalid.')
        return row


def read_result(path):
    """Read only numeric fields; never unpickle notebook metadata."""
    with np.load(path, allow_pickle=False) as z:
        if 'freq_vector' not in z or 'measured_mask' not in z:
            raise ValueError('Select a GUI result containing freq_vector and measured_mask.')
        return dict(data=z['row_sparams'].copy(), mask=z['measured_mask'].copy(),
                    freq=z['freq_vector'].copy(), config=json.loads(str(z['config_json'])),
                    simulated=bool(z['simulated']))
