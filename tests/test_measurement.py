from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
from sow1_gui.config import defaults, validate
from sow1_gui.engine import Engine
from sow1_gui.instrument import KeysightInstrument
from sow1_gui.storage import RowStore, read_result


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c = defaults()
        self.c.update(folder=self.tmp.name, nh=3, sweeps=2, live_points=101,
                      segments=[[1e9, 1.1e9, 5], [2e9, 2.1e9, 7]])
        self.e = Engine()
        self.e.connect(simulation=True)
        self.addCleanup(self.e.disconnect)

    def ready(self):
        self.e.configure(self.c, 'calibration')
        self.e.confirm_calibration('test fixture')
        self.e.configure(self.c, 'segmented')

    def test_roundtrip_modes_endpoint_selection_and_compatible_row(self):
        self.ready()
        self.e.configure(self.c, 'time')
        self.assertFalse(self.e.device.correction())
        self.e.configure(self.c, 'segmented')
        self.e.start_row(self.c)
        acquired = []
        original = self.e.device.acquire
        self.e.device.acquire = lambda n: (acquired.append(n), original(n))[1]
        for i in range(3):
            self.e.acquire_point(i)
        self.assertEqual(acquired, [12, 12, 3, 3, 12, 12])
        row = self.e.row
        self.assertTrue(row.complete)
        self.assertTrue(row.final.exists())
        self.assertFalse(row.path.exists())
        r = read_result(row.final)
        self.assertEqual(r['data'].shape, (3, 2, 12, 12))
        self.assertFalse(r['mask'][1, :, 3:].any())
        self.assertTrue(np.all(r['data'][1, :, 3:] == 0))
        np.testing.assert_array_equal(r['freq'], self.e.device.frequencies())
        with np.load(row.final, allow_pickle=True) as z:
            self.assertEqual(z['conf_params'].item()['sparam_list'], self.c['sparams'])
            self.assertEqual(z['bs_params'].item()['num_cols'], 3)
            self.assertEqual(z['timestamp_list'].shape, (2, 3))
        with self.assertRaises(ValueError):
            self.e.acquire_point(0)

    def test_cancel_resume_skips_completed_sweeps_and_requires_confirmation(self):
        self.ready()
        self.e.start_row(self.c)
        calls = []
        original = self.e.device.acquire
        def acquire(n):
            calls.append(n)
            self.e.cancel.set()
            return original(n)
        self.e.device.acquire = acquire
        self.e.acquire_point(0)
        path = self.e.row.path
        self.assertTrue(path.exists())
        self.assertTrue(self.e.row.mask[0, 0].all())
        self.assertFalse(self.e.row.mask[0, 1].any())
        self.e.close_row()
        self.e.resume_row(path)
        with self.assertRaises(ValueError):
            self.e.configure(self.c, 'segmented')
        self.ready()
        self.e.cancel.clear()
        self.e.device.acquire = original
        self.e.acquire_point(0)
        self.assertTrue(self.e.row.complete_points[0])
        self.assertEqual(calls, [12])

    def test_changed_calibration_grid_and_row_settings_are_rejected(self):
        self.ready()
        changed = deepcopy(self.c)
        changed['powers'][0] -= 1
        with self.assertRaises(ValueError):
            self.e.configure(changed, 'segmented')
        self.e.start_row(self.c)
        changed = deepcopy(self.c)
        changed['metadata']['notes'] = 'different scene'
        with self.assertRaises(ValueError):
            self.e.configure(changed, 'time')

    def test_acquisition_failure_cleans_up_extra_traces(self):
        self.ready()
        self.e.start_row(self.c)
        self.e.device.acquire = MagicMock(side_effect=RuntimeError('connection lost'))
        self.e.device.remove_extras = MagicMock()
        with self.assertRaises(RuntimeError):
            self.e.acquire_point(0)
        self.e.device.remove_extras.assert_called_once_with(12)
        resumed = RowStore.resume(self.e.row.path)
        self.assertFalse(resumed.mask.any())

    def test_bad_sample_shape_does_not_mark_point_complete(self):
        self.ready()
        self.e.start_row(self.c)
        self.e.device.acquire = lambda count: np.zeros((count, 1))
        with self.assertRaises(ValueError):
            self.e.acquire_point(1)
        self.assertFalse(self.e.row.mask.any())

    def test_disk_failure_rolls_back_unsaved_sweep(self):
        self.ready()
        self.e.start_row(self.c)
        self.e.row.save = MagicMock(side_effect=OSError('disk full'))
        with self.assertRaises(OSError):
            self.e.acquire_point(1)
        self.assertFalse(self.e.row.mask.any())
        self.assertFalse(RowStore.resume(self.e.row.path).mask.any())

    def test_average_uses_one_saved_result_per_point(self):
        self.c.update(sweeps=4, average=4)
        self.ready()
        self.e.start_row(self.c)
        self.e.acquire_point(1)
        self.assertEqual(self.e.row.data.shape[1], 1)
        self.assertTrue(self.e.row.complete_points[1])

    def test_sol_requires_all_instrument_steps(self):
        self.e.configure(self.c, 'calibration')
        self.e.begin_calibration(1, '2.92', 'f', '85561A Databased')
        self.e.measure_standard(1)
        with self.assertRaises(ValueError):
            self.e.apply_calibration('test')
        with self.assertRaises(ValueError):
            self.e.configure(self.c, 'time')
        self.e.measure_standard(2)
        self.e.measure_standard(3)
        self.e.apply_calibration('test')
        self.assertIsNone(self.e.calibration)
        self.e.confirm_calibration('verified all required ports on instrument')

    def test_b2b_unique_files_and_frequency_sidecar(self):
        self.ready()
        self.e.b2b(1)
        self.e.b2b(1)
        files = list(self.e.session.glob('S21_b2b*.npy'))
        self.assertEqual(len(files), 2)
        with np.load(files[0].with_suffix('.npz'), allow_pickle=False) as z:
            self.assertTrue(bool(z['simulated']))
            np.testing.assert_array_equal(z['freq_vector'], self.e.device.frequencies())

    def test_validation_rejects_unsafe_or_ambiguous_shapes(self):
        for key, value in [('nh', 0), ('row', 1), ('powers', [0]), ('average', 3), ('ifbw', float('nan'))]:
            c = deepcopy(self.c)
            c[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                validate(c)
        c = deepcopy(self.c)
        c['segments'][1][0] = c['segments'][0][1] - 1
        with self.assertRaises(ValueError):
            validate(c)


class AdapterTest(unittest.TestCase):
    def test_frequency_readback_uses_64_bits_and_rejects_grid_change(self):
        d = KeysightInstrument.__new__(KeysightInstrument)
        d.vna = MagicMock()
        d.mode = 'segmented'
        expected = np.array([1e9, 1e9 + 1])
        d.vna.get_freq_vector.return_value = expected
        d.vna.inst.query_binary_values.return_value = expected
        d.vna.query.return_value = '0,"No error"'
        np.testing.assert_array_equal(d.frequencies(), expected)
        self.assertEqual(d.vna.inst.query_binary_values.call_args.kwargs['datatype'], 'd')
        self.assertEqual(d.vna.write.call_args.args[0], 'FORMat:DATA REAL,32')
        d.vna.inst.query_binary_values.return_value = expected + 100
        with self.assertRaises(RuntimeError):
            d.frequencies()

    def test_live_reads_formatted_transform_with_instrument_time_limits(self):
        d = KeysightInstrument.__new__(KeysightInstrument)
        d.vna = MagicMock()
        d.mode, d.names, d.config = 'time', ['S21'], defaults()
        def query(command):
            if 'STARt?' in command:
                return '1e-6'
            if 'STOP?' in command:
                return '2e-6'
            if 'ERRor?' in command:
                return '0,"No error"'
            raise AssertionError(command)
        d.vna.query.side_effect = query
        d.vna.inst.query_binary_values.return_value = np.array([-80., -50., -90.])
        traces = d.live()
        np.testing.assert_allclose(traces[0]['x'], [300, 450, 600])
        np.testing.assert_array_equal(traces[0]['y'], [-80, -50, -90])
        self.assertIn('FDATa?', d.vna.inst.query_binary_values.call_args.args[0])
        d.vna.run_cont.assert_called_once()

    def test_axes_use_library_calls_and_time_in_seconds(self):
        d = KeysightInstrument.__new__(KeysightInstrument)
        d.vna = MagicMock()
        d.vna.query.return_value = '0,"No error"'
        d.mode, d.names, d.config = 'time', ['S21'], defaults()
        axes = [dict(mode='time', xmin=1e-6, xmax=2e-6, ymin=-120, ymax=-20)]
        d.axes(axes)
        d.vna.set_Xscale.assert_called_once_with(xmin=1e-6, xmax=2e-6, mode='time', trace=1)
        d.vna.set_Yscale.assert_called_once_with(ymin=-120, ymax=-20, trace=1)


if __name__ == '__main__':
    unittest.main()
