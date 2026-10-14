"""Campaign defaults and validation, independent of Qt and hardware."""
from copy import deepcopy
import json
import math

SPARAMS = ['S21', 'S31', 'S41', 'S22', 'S33', 'S44', 'S32', 'S42', 'S23', 'S43', 'S24', 'S34']


def defaults():
    ue = dict(ant_coord=[4.97, 2.33], ant_height=1.04, ant_orientation=[69, 0],
              v_pol_vector=[0, 0, 1], h_pol_vector=None, ant_type='power-log',
              ant_name='Aaronia-PowerLOG-70180', num_rows=1, num_cols=1,
              vertical_spacing=None, horizontal_spacing=None)
    return dict(
        folder='data/VA_nexus_outdoor_trials_2', setup=6, row=0, va_vert_pos=0,
        nh=10, nv=1, dh=10.0, dv=10.0, ifbw=10000.0,
        powers=[-15.0, 0.0, 0.0, 0.0], sweeps=10, average=1,
        sparams=SPARAMS.copy(),
        segments=[[3.79996e9, 3.90004e9, 279], [6.79984e9, 7.20016e9, 557],
                  [7.29984e9, 7.70016e9, 557], [10.04984e9, 10.45016e9, 557],
                  [12.79984e9, 13.20016e9, 557], [14.79984e9, 15.20016e9, 557]],
        live_start=10e9, live_stop=15e9, live_points=10001, live_ifbw=10000.0,
        window='hann', interval=0.5,
        axes=[dict(mode='dist', xmin=160.0 if i < 3 else 0.0,
                   xmax=230.0 if i < 3 else 20.0, ymin=-120.0, ymax=-20.0) for i in range(6)],
        metadata=dict(
            bs_params=dict(ant_coord=[0.80, 0.02], ant_height=1.69,
                           ant_orientation=[111, 0], v_pol_vector=[0, 0, 1],
                           h_pol_vector=None, ant_type='omnidirectional', ant_name='QOM-ST-2-18-S-SG-R'),
            ue0_params={**deepcopy(ue), 'ant_coord': [3.71, 4.00], 'ant_height': 1.16},
            ue1_params=deepcopy(ue), ue2_params=deepcopy(ue), notes=''))


def validate(config):
    c = deepcopy(config)
    # Also reject NaN/Infinity in arbitrary metadata before writing JSON.
    json.dumps(c, allow_nan=False)
    for key in ('nh', 'nv', 'sweeps', 'average', 'live_points'):
        if isinstance(c[key], bool) or int(c[key]) != c[key] or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if c['live_points'] < 2:
        raise ValueError('Live sweep needs at least two points.')
    for key in ('setup', 'row', 'va_vert_pos'):
        if int(c[key]) != c[key] or c[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    if c['row'] >= c['nv']:
        raise ValueError('Row index must be smaller than Nv (indices start at zero).')
    if not str(c['folder']).strip():
        raise ValueError('Choose an output folder.')
    for key in ('ifbw', 'live_ifbw', 'dh', 'dv', 'interval'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if not 0 < c['live_start'] < c['live_stop']:
        raise ValueError('Live frequency start must be positive and below stop.')
    if len(c['powers']) != 4 or not all(math.isfinite(x) for x in c['powers']):
        raise ValueError('Specify four finite port powers.')
    if c['sparams'][:3] != SPARAMS[:3] or not 3 <= len(c['sparams']) <= 16:
        raise ValueError('The first three traces must be S21, S31, S41; use 3–16 traces.')
    allowed = {f'S{i}{j}' for i in range(1, 5) for j in range(1, 5)}
    if len(set(c['sparams'])) != len(c['sparams']) or not set(c['sparams']) <= allowed:
        raise ValueError('S-parameters must be unique and use ports 1–4.')
    if not c['segments']:
        raise ValueError('At least one segment is required.')
    last_stop = 0
    for start, stop, points in c['segments']:
        if not last_stop < start < stop or points < 2 or int(points) != points:
            raise ValueError('Segments must be ascending, nonoverlapping, with start < stop and ≥2 points.')
        last_stop = stop
    if c['average'] > 1 and c['sweeps'] < c['average']:
        raise ValueError('Grouped sweeps must be at least the averaging count to fill the average.')
    if c['window'] not in ('hann', 'hamming', 'rectangle', 'kaiser', 'bohman'):
        raise ValueError('Unsupported transform window.')
    if len(c['axes']) != 6:
        raise ValueError('Six live trace axis settings are required.')
    for a in c['axes']:
        if a['mode'] not in ('dist', 'time') or not a['xmin'] < a['xmax'] or not a['ymin'] < a['ymax']:
            raise ValueError('Axis limits must increase; mode must be dist or time.')
        if a['ymax'] - a['ymin'] < 10:
            raise ValueError('Use a Y span of at least 10 dB for the VNA scale divisions.')
    for key in ('bs_params', 'ue0_params', 'ue1_params', 'ue2_params'):
        if not isinstance(c['metadata'].get(key), dict):
            raise ValueError(f'Metadata must contain an object named {key}.')
    samples = c['sweeps'] if c['average'] == 1 else 1
    size = c['nh'] * samples * len(c['sparams']) * sum(int(s[2]) for s in c['segments']) * 8
    if size > 1_000_000_000:
        raise ValueError('A row would exceed 1 GB. Reduce points, positions, or repeats.')
    return c


def calibration_signature(c):
    return json.dumps({k: c[k] for k in ('segments', 'ifbw', 'powers', 'sparams')}, sort_keys=True)


def row_signature(c):
    keys = ('setup', 'row', 'va_vert_pos', 'nh', 'nv', 'dh', 'dv', 'sweeps', 'average', 'metadata', 'folder')
    return calibration_signature(c) + json.dumps({k: c[k] for k in keys}, sort_keys=True)


def legacy_conf(c):
    return dict(ifbw=c['ifbw'], pow_dbm_list=c['powers'], n_sweeps=c['sweeps'],
                n_average=c['average'], sparam_list=c['sparams'],
                f_start_list=[s[0] for s in c['segments']], f_stop_list=[s[1] for s in c['segments']],
                points_list=[int(s[2]) for s in c['segments']])
