"""Keysight adapter and explicit simulation. Only the worker owns an adapter."""
from copy import deepcopy
import time
import numpy as np


class KeysightInstrument:
    simulated = False

    def __init__(self, address=''):
        # connect_vna currently ignores its address argument. Instantiate its driver
        # directly so an explicit VISA resource is actually respected.
        from mcg_lab_vna_control.drivers.keysight_streamline import STREAMLINEVNA
        self.vna = STREAMLINEVNA(resource_str=address.strip() or None)
        try:
            self.vna.open()
            self.vna.write('FORMat:DATA REAL,32')
            self.vna.write('FORMat:BORDer NORMal')  # driver reads big-endian floats
            self.identity = self.vna.query('*IDN?').strip()
        except Exception:
            self.vna.close()
            raise
        self.mode = 'unconfigured'
        self.config = None
        self.names = []

    def check_errors(self):
        errors = []
        for _ in range(20):
            reply = self.vna.query('SYSTem:ERRor?').strip()
            if int(reply.split(',')[0]) == 0:
                break
            errors.append(reply)
        if errors:
            raise RuntimeError('VNA error queue: ' + '; '.join(errors))

    def configure(self, c, mode):
        v = self.vna
        self.mode = 'unconfigured'
        v.mode_hold()
        v.pseudo_preset()  # deliberately preserve stored calibration
        self.names = c['sparams'][:6] if mode in ('time', 'calibration') else c['sparams'][:3]
        if mode == 'time':
            v.configure_multiple_sparams(sparam_list=self.names, f_start=c['live_start'],
                f_stop=c['live_stop'], points=c['live_points'], ifbw=c['live_ifbw'], pow_dbm_list=c['powers'])
            for i in range(len(self.names)):
                v.set_timedomain(trace=i + 1, window=c['window'])
        else:
            v.configure_multiple_segmented_sparams(sparam_list=self.names,
                f_start_list=[s[0] for s in c['segments']], f_stop_list=[s[1] for s in c['segments']],
                points_list=[int(s[2]) for s in c['segments']], ifbw=c['ifbw'], pow_dbm_list=c['powers'])
            if mode == 'segmented':
                v.calibration_on()
        for i in range(len(self.names)):
            v.write(f'CALCulate1:MEASure{i+1}:FORMat MLOGarithmic')
        v.wait_operation_complete()
        self.check_errors()
        self.config = deepcopy(c)
        self.mode = mode
        if mode == 'time':
            self.axes(c['axes'])
        v.run_cont()

    def axes(self, axes):
        if self.mode != 'time':
            raise ValueError('VNA X-axis changes require time-domain mode.')
        for i, a in enumerate(axes[:len(self.names)], 1):
            self.vna.set_Xscale(xmin=a['xmin'], xmax=a['xmax'], mode=a['mode'], trace=i)
            self.vna.set_Yscale(ymin=a['ymin'], ymax=a['ymax'], trace=i)
        self.vna.wait_operation_complete()
        self.check_errors()
        self.config['axes'] = deepcopy(axes)

    def autoscale(self):
        self.vna.auto_scale_all()
        self.check_errors()

    def frequencies(self):
        if self.mode == 'time':
            raise ValueError('Frequency stimulus readback requires a frequency-domain configuration.')
        # The driver's get_freq_vector reconstructs the requested grid. Read the
        # instrument stimulus in 64-bit precision as well, so saved axes describe
        # the actual sweep and front-panel grid changes are detected.
        self.vna.write('FORMat:DATA REAL,64')
        try:
            values = np.asarray(self.vna.inst.query_binary_values('CALCulate1:MEASure1:X:VALues?',
                                datatype='d', is_big_endian=True, container=np.array), dtype=np.float64)
        finally:
            self.vna.write('FORMat:DATA REAL,32')
        expected = self.vna.get_freq_vector()
        if values.shape != expected.shape or not np.allclose(values, expected, rtol=0, atol=1):
            raise RuntimeError('Instrument frequency grid differs from the applied configuration. Restore segmented mode.')
        self.check_errors()
        return values

    def correction(self):
        return self.vna.query('SENSe1:CORRection:STATe?').strip().upper() in ('1', '+1', 'ON')

    def confirm_calibration(self):
        self.vna.calibration_on()
        self.vna.wait_operation_complete()
        self.check_errors()
        if not self.correction():
            raise RuntimeError('The VNA did not enable correction.')

    def live(self):
        v = self.vna
        v.set_sweeps(1, verbose=False)
        v.unset_average()
        v.run_sweep()
        v.wait_operation_complete()
        traces = []
        frequencies = self.frequencies() / 1e9 if self.mode != 'time' else None
        for i, name in enumerate(self.names, 1):
            # FDATA is the instrument's displayed transform, unlike SDATA used
            # for the calibrated complex frequency-domain campaign data.
            y = np.asarray(v.inst.query_binary_values(f'CALCulate1:MEASure{i}:DATA:FDATa?',
                           datatype='f', is_big_endian=True, container=np.array))
            if self.mode == 'time':
                start = float(v.query(f'CALCulate1:MEASure{i}:TRANsform:TIME:STARt?'))
                stop = float(v.query(f'CALCulate1:MEASure{i}:TRANsform:TIME:STOP?'))
                a = self.config['axes'][i - 1]
                factor, unit = (3e8, 'm') if a['mode'] == 'dist' else (1e9, 'ns')
                x = np.linspace(start, stop, len(y)) * factor
            else:
                x, unit = frequencies, 'GHz'
            if len(x) != len(y) or len(y) < 2 or not np.isfinite(y).all():
                raise RuntimeError(f'Invalid display data for {name}.')
            traces.append(dict(name=name, x=x, y=y, unit=unit))
        self.check_errors()
        v.run_cont()
        return traces

    def set_acquisition(self, c):
        if c['average'] == 1:
            self.vna.set_sweeps(1, verbose=False)
            self.vna.unset_average()
        else:
            self.vna.set_sweeps(c['sweeps'], verbose=False)
            self.vna.set_average(c['average'])

    def extra_traces(self, names):
        self.vna.add_multiple_sparams(sparam_list=names, display_sparam_list=[False] * len(names))
        self.vna.wait_operation_complete()
        self.check_errors()

    def remove_extras(self, count):
        self.vna.delete_multiple_sparams(trace_list=list(range(4, count + 1)))
        self.check_errors()

    def acquire(self, count):
        if not self.correction():
            raise RuntimeError('Correction is OFF. Restore calibrated segmented mode.')
        data = self.vna.measure_multiple_traces_sparams(trace_list=list(range(1, count + 1)))
        self.check_errors()
        return data

    def calibration_begin(self, port, connector, gender, kit):
        genders = list('nnnn')
        genders[port - 1] = gender
        # The driver's init_SOL_calibration swallows exceptions; the underlying
        # public initializer propagates them and lets the UI show actual failures.
        self.vna.init_general_calibration(conn_type=connector, conn_gender=''.join(genders), calkit=kit)
        self.vna.wait_operation_complete()
        self.check_errors()
        count = int(self.vna.query('SENSe1:CORRection:COLLect:GUIDed:STEPs?'))
        if not 1 <= count <= 100:
            raise RuntimeError('Instrument returned an invalid calibration step count.')
        return [self.vna.query(f'SENSe1:CORRection:COLLect:GUIDed:DESCription? {i}').strip()
                for i in range(1, count + 1)]

    def calibration_step(self, step):
        self.vna.measure_SOL_calibrator(step)
        self.vna.wait_operation_complete()
        self.check_errors()

    def calibration_apply(self, name):
        self.vna.apply_calibration(calset_name=name)
        self.vna.wait_operation_complete()
        self.check_errors()

    def calibration_abort(self):
        self.vna.abort_calibration()
        self.vna.wait_operation_complete()
        self.check_errors()

    def hold(self):
        self.vna.mode_hold()

    def close(self):
        try:
            self.hold()
        finally:
            self.vna.close()


class SimulatedInstrument:
    """Deterministic synthetic channel; never imports or opens VISA."""
    simulated = True
    identity = 'SIMULATION / synthetic SOW1 channel / no hardware'

    def __init__(self, address=''):
        self.mode = 'unconfigured'
        self.config = None
        self.names = []
        self.corrected = False
        self.tick = 0
        self.rng = np.random.default_rng(2026)

    def configure(self, c, mode):
        self.config, self.mode = deepcopy(c), mode
        self.names = c['sparams'][:6] if mode in ('time', 'calibration') else c['sparams'][:3]
        self.corrected = mode == 'segmented'

    def axes(self, axes):
        self.config['axes'] = deepcopy(axes)

    def autoscale(self):
        pass

    def frequencies(self):
        c = self.config
        if self.mode == 'time':
            return np.linspace(c['live_start'], c['live_stop'], c['live_points'])
        return np.concatenate([np.linspace(s, e, int(n)) for s, e, n in c['segments']])

    def correction(self):
        return self.corrected

    def confirm_calibration(self):
        self.corrected = True

    def live(self):
        self.tick += 1
        traces = []
        for i, name in enumerate(self.names):
            if self.mode == 'time':
                a = self.config['axes'][i]
                x = np.linspace(a['xmin'], a['xmax'], self.config['live_points'])
                span = a['xmax'] - a['xmin']
                peak = a['xmin'] + span * (0.45 + 0.03 * np.sin(self.tick / 10 + i))
                y = -108 + self.rng.normal(0, 1.4, len(x))
                y += 55 * np.exp(-((x - peak) / (span * 0.016)) ** 2)
                y += 25 * np.exp(-((x - peak - span * 0.19) / (span * 0.025)) ** 2)
                if a['mode'] == 'time':
                    x = x * 1e9
                unit = 'm' if a['mode'] == 'dist' else 'ns'
            else:
                x = self.frequencies() / 1e9
                y = -65 - 5 * i + 5 * np.sin(x * 8 + self.tick / 12)
                unit = 'GHz'
            traces.append(dict(name=name, x=x, y=y, unit=unit))
        return traces

    def set_acquisition(self, c):
        pass

    def extra_traces(self, names):
        pass

    def remove_extras(self, count):
        pass

    def acquire(self, count):
        time.sleep(0.04)
        f = self.frequencies()
        self.tick += 1
        return np.asarray([10 ** ((-60 - i * 3) / 20) * np.exp(-2j * np.pi * f * (650 + i * 7) * 1e-9)
                           + (self.rng.normal(size=len(f)) + 1j * self.rng.normal(size=len(f))) * 1e-6
                           for i in range(count)], dtype=np.complex64)

    def calibration_begin(self, port, connector, gender, kit):
        return [f'SIMULATION: connect {standard} to port {port}' for standard in ('OPEN', 'SHORT', 'LOAD')]

    def calibration_step(self, step):
        pass

    def calibration_apply(self, name):
        self.corrected = True

    def calibration_abort(self):
        pass

    def hold(self):
        pass

    def close(self):
        pass
