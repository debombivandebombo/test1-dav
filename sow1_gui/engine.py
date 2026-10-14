"""Measurement state machine; also usable in tests without a GUI."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import threading
import numpy as np
from .config import validate, calibration_signature, row_signature
from .instrument import KeysightInstrument, SimulatedInstrument
from .storage import RowStore, timestamp


class Engine:
    def __init__(self, emit=lambda kind, value: None):
        self.emit = emit
        self.device = None
        self.mode = 'disconnected'
        self.config = None
        self.calibration = None
        self.row = None
        self.session = None
        self.session_base = None
        self.live = False
        self.cancel = threading.Event()
        self.cal_steps = []
        self.cal_done = set()

    def status(self):
        return dict(connected=self.device is not None, mode=self.mode,
                    identity=self.device.identity if self.device else 'No instrument connected',
                    simulated=self.device.simulated if self.device else True, live=self.live,
                    calibrated=bool(self.config and self.calibration and self.calibration['signature'] == calibration_signature(self.config)),
                    row_active=self.row is not None,
                    points=self.row.complete_points.tolist() if self.row else [],
                    saved=str(self.row.final if self.row.complete else self.row.path) if self.row else '',
                    config=deepcopy(self.config))

    def log(self, message):
        self.emit('log', f'{timestamp()}  {message}')
        if self.session:
            with (self.session / 'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(time=timestamp(), message=message)) + '\n')

    def connect(self, simulation=True, address=''):
        if self.device:
            raise ValueError('Disconnect the current instrument first.')
        if self.row and self.row.simulated != simulation:
            raise ValueError('Close this row before connecting a different simulation/hardware mode.')
        self.device = SimulatedInstrument(address) if simulation else KeysightInstrument(address)
        self.mode = 'unconfigured'
        self.calibration = None
        self.log('Connected: ' + self.device.identity)

    def disconnect(self):
        self.live = False
        if self.device:
            try:
                self.device.close()
            finally:
                self.device = None
                self.mode = 'disconnected'
                self.calibration = None
        self.cal_steps = []
        self.emit('cal_steps', [])

    def configure(self, config, mode):
        if not self.device:
            raise ValueError('Connect an instrument first.')
        c = validate(config)
        if mode not in ('time', 'segmented', 'calibration'):
            raise ValueError('Unknown mode.')
        if self.cal_steps:
            raise ValueError('Apply or abort the current SOL calibration first.')
        if self.row and row_signature(c) != row_signature(self.row.config):
            raise ValueError('This row has frozen acquisition settings. Close it before changing them.')
        if mode == 'segmented' and (not self.calibration or self.calibration['signature'] != calibration_signature(c)):
            raise ValueError('Prepare segmented calibration and confirm the active calibration for these settings first.')
        self.live = False
        self.mode = 'unconfigured'
        self.device.configure(c, mode)
        if mode == 'segmented' and not self.device.correction():
            raise RuntimeError('Correction is OFF after restoring segmented mode.')
        self.config, self.mode = c, mode
        self.log(f'Mode: {mode}. Configuration applied.')

    def confirm_calibration(self, note):
        if self.mode != 'calibration' or self.cal_steps:
            raise ValueError('Prepare segmented calibration; apply or abort any SOL sequence first.')
        if not note.strip():
            raise ValueError('Record the calibration identifier or verification note.')
        self.device.confirm_calibration()
        self.calibration = dict(signature=calibration_signature(self.config), confirmed_at=timestamp(),
                                note=note.strip(), identity=self.device.identity,
                                basis='operator verified coverage; correction enabled')
        self.log('Calibration confirmed: ' + note.strip())

    def axes(self, axes):
        if self.mode != 'time':
            raise ValueError('Switch to time-domain inspection first.')
        c = deepcopy(self.config)
        c['axes'] = axes
        validate(c)
        self.device.axes(axes)
        self.config = c

    def autoscale(self):
        if not self.device or self.mode in ('unconfigured', 'disconnected'):
            raise ValueError('Configure the instrument first.')
        self.device.autoscale()

    def set_live(self, enabled):
        if enabled and (self.mode not in ('time', 'segmented', 'calibration') or self.cal_steps):
            raise ValueError('Finish calibration and configure a display mode first.')
        self.live = enabled
        if not enabled and self.device:
            self.device.hold()

    def poll(self):
        self.emit('frame', dict(mode=self.mode, traces=self.device.live(), time=timestamp(), simulated=self.device.simulated))

    def begin_calibration(self, port, connector, gender, kit):
        if self.mode != 'calibration' or self.row or self.cal_steps:
            raise ValueError('Close the active row and prepare calibration; finish any existing SOL sequence.')
        self.live = False
        self.calibration = None
        self.cal_steps = self.device.calibration_begin(port, connector, gender, kit)
        self.cal_done = set()
        self.emit('cal_steps', self.cal_steps)
        self.log(f'SOL started on port {port}.')

    def measure_standard(self, step):
        if not 1 <= step <= len(self.cal_steps):
            raise ValueError('Select a calibration step returned by the VNA.')
        self.device.calibration_step(step)
        self.cal_done.add(step)
        self.log(f'Calibration standard {step} acquired.')

    def apply_calibration(self, name):
        if not self.cal_steps or len(self.cal_done) != len(self.cal_steps):
            raise ValueError('Acquire every requested standard before applying SOL.')
        if not re.fullmatch(r'[\w .-]{1,64}', name):
            raise ValueError('Use 1–64 letters, numbers, spaces, dots, hyphens or underscores for the calset name.')
        self.device.calibration_apply(name)
        self.cal_steps = []
        self.emit('cal_steps', [])
        self.log('SOL saved to instrument calset: ' + name)

    def abort_calibration(self):
        if not self.device or not self.cal_steps:
            raise ValueError('No SOL sequence is active.')
        self.device.calibration_abort()
        self.cal_steps = []
        self.emit('cal_steps', [])
        self.log('SOL sequence aborted.')

    def _require_segmented(self):
        if self.mode != 'segmented' or not self.calibration or self.calibration['signature'] != calibration_signature(self.config):
            raise ValueError('Restore the confirmed calibrated segmented configuration first.')
        if not self.device.correction():
            raise RuntimeError('VNA correction is OFF.')

    def _session(self):
        base = Path(self.config['folder']).resolve()
        # Never mix real and synthetic runs or reuse a session across reconnects
        # to a different kind of instrument.
        key = (str(base), self.device.simulated)
        if self.session is None or key != self.session_base:
            suffix = '_SIMULATION' if self.device.simulated else '_MEASUREMENT'
            self.session = base / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + suffix)
            self.session.mkdir(parents=True, exist_ok=False)
            self.session_base = key
        return self.session

    def start_row(self, config):
        self._require_segmented()
        c = validate(config)
        if self.row:
            raise ValueError('Close the current row before starting another.')
        if row_signature(c) != row_signature(self.config):
            raise ValueError('Apply the segmented settings before starting this row.')
        self.row = RowStore(self._session(), c, self.device.frequencies(), self.device.identity,
                            self.device.simulated, self.calibration)
        self.log(f"Row opened: setup {c['setup']}, row {c['row']}; manually position the antenna before each point.")

    def close_row(self):
        if self.row:
            path = self.row.save()
            self.log('Row closed; data retained at ' + str(path))
            self.row = None

    def resume_row(self, path):
        if self.row:
            raise ValueError('Close the current row first.')
        row = RowStore.resume(path)
        if self.device and row.simulated != self.device.simulated:
            raise ValueError('Cannot resume a simulated row on hardware or a hardware row in simulation.')
        self.row = row
        self.config = deepcopy(row.config)
        self.calibration = None  # a resumed file is not evidence of active instrument calibration
        self.live = False
        self.mode = 'unconfigured' if self.device else 'disconnected'
        self.session, self.session_base = row.folder, (str(Path(row.config['folder']).resolve()), row.simulated)
        self.emit('loaded_config', deepcopy(row.config))
        self.log('Checkpoint loaded. Reconfirm instrument calibration before continuing.')

    def acquire_point(self, point):
        self._require_segmented()
        if not self.row or not 0 <= point < self.row.config['nh']:
            raise ValueError('Open a row and choose a valid horizontal index.')
        if self.row.simulated != self.device.simulated:
            raise ValueError('The checkpoint and connected instrument have different simulation modes.')
        if self.row.identity != self.device.identity:
            raise ValueError('The connected instrument differs from the one used to open this row.')
        if row_signature(self.config) != row_signature(self.row.config):
            raise ValueError('Active configuration differs from the row snapshot.')
        if self.row.complete_points[point]:
            raise ValueError('This point is already complete; choose another point or a new setup.')
        if not np.array_equal(self.device.frequencies(), self.row.freq):
            raise ValueError('Frequency vector differs from this row.')
        self.live = False
        c = self.row.config
        endpoint = point in (0, c['nh'] - 1)
        count = len(c['sparams']) if endpoint else 3
        self.device.set_acquisition(c)
        try:
            if endpoint:
                self.device.extra_traces(c['sparams'][3:])
            for sweep in range(self.row.data.shape[1]):
                if self.cancel.is_set():
                    break
                if self.row.mask[point, sweep, :count].all():
                    continue
                values = self.device.acquire(count)
                # Retain the calibration confirmation used for the latest acquisition.
                self.row.calibration = deepcopy(self.calibration)
                self.row.put(point, sweep, values)
                self.emit('progress', dict(point=point, sweep=sweep + 1, total=self.row.data.shape[1],
                                           points=self.row.complete_points.tolist()))
                self.emit('frame', dict(mode='segmented', time=timestamp(), simulated=self.device.simulated,
                    traces=[dict(name=c['sparams'][i], x=self.row.freq / 1e9,
                                 y=20 * np.log10(np.maximum(np.abs(values[i]), 1e-20)), unit='GHz')
                            for i in range(min(count, 6))]))
        finally:
            try:
                if endpoint:
                    self.device.remove_extras(count)
            except Exception:
                self.mode = 'unconfigured'
                raise
            finally:
                self.device.hold()
        self.log(f'Point {point}: ' + ('complete' if self.row.complete_points[point] else 'paused; partial sweeps saved'))
        if self.row.complete:
            self.log('ROW COMPLETE: ' + str(self.row.final))

    def b2b(self, trace):
        self._require_segmented()
        if trace not in (1, 2, 3):
            raise ValueError('B2B trace must be S21, S31 or S41.')
        self.live = False
        self.device.set_acquisition(self.config)
        values = self.device.acquire(3)[trace - 1]
        target = self._session() / f"{self.config['sparams'][trace-1]}_b2b_cal_{datetime.now().strftime('%H%M%S_%f')}.npy"
        with target.open('xb') as f:
            np.save(f, values)
        np.savez(target.with_suffix('.npz'), sparam=values, freq_vector=self.device.frequencies(),
                 config_json=json.dumps(self.config), calibration_json=json.dumps(self.calibration),
                 simulated=self.device.simulated, instrument_identity=self.device.identity)
        self.device.hold()
        self.log('Saved through reference: ' + str(target))
