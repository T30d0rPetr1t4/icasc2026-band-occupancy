"""Decoder for R&S ARGUS/ESMD frequency-sweep .exp exports.

Record layout (26 bytes, little endian):
    FILETIME (uint64) | frequency Hz (float64) | level dBuV/m (float64) | pad (uint16)
The header length varies; the data block start is located by searching for
the first run of consecutive valid FILETIME values at the record stride.
"""
import glob, os, re, struct, datetime
import numpy as np

REC = 26
DT = np.dtype([('ft', '<u8'), ('freq', '<f8'), ('level', '<f8'), ('pad', '<u2')])
EPOCH = datetime.datetime(1601, 1, 1)
FT_LO = int((datetime.datetime(2000, 1, 1) - EPOCH).total_seconds() * 1e7)
FT_HI = int((datetime.datetime(2100, 1, 1) - EPOCH).total_seconds() * 1e7)


def find_data_start(data, search_limit=300000, confirm=6):
    limit = min(search_limit, len(data) - 8)
    for i in range(0, limit, 2):
        v = struct.unpack_from('<Q', data, i)[0]
        if FT_LO <= v <= FT_HI:
            ok = True
            for k in range(1, confirm):
                off = i + k * REC
                if off + 8 > len(data):
                    ok = False; break
                v2 = struct.unpack_from('<Q', data, off)[0]
                if not (FT_LO <= v2 <= FT_HI):
                    ok = False; break
            if ok:
                return i
    return None


def decode_exp(path):
    data = open(path, 'rb').read()
    start = find_data_start(data)
    if start is None:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.uint64)
    n = (len(data) - start) // REC
    arr = np.frombuffer(data[start:start + n * REC], dtype=DT, count=n)
    f = arr['freq'] / 1e6
    L = arr['level']
    ok = (f > 1) & (f < 6000) & (L > -200) & (L < 200)
    return f[ok], L[ok], arr['ft'][ok]


def ft_to_datetime64(ft):
    us = (ft // 10).astype('int64')
    return np.datetime64('1601-01-01T00:00:00', 'us') + us.astype('timedelta64[us]')


def parse_xml(path):
    txt = open(path, encoding='utf-8-sig', errors='ignore').read()
    def g(p, cast=str):
        m = re.search(p, txt)
        return cast(m.group(1).strip()) if m else None
    return dict(
        station=g(r'<MD_S_NAME>([^<]*)'),
        fmin=g(r'Minimum Frequency:\s*([\d.]+)', float),
        fmax=g(r'Maximum Frequency:\s*([\d.]+)', float),
        step_khz=g(r'Step Width:\s*([\d.]+)', float),
        detector=g(r'Detector:\s*([^\r\n<]+)'),
        ifbw=g(r'IF Bandwidth:\s*([^\r\n<]+)'),
        antenna=g(r'<MD_M_ANT>([^<]*)'),
        receiver=g(r'<MD_M_REC>([^<]*)'),
        start=g(r'<MD_M_START>([^<]*)'),
        stop=g(r'<MD_M_STOP>([^<]*)'),
    )


def load_mission(folder, max_files=None):
    """Load all .exp files of one band folder -> dict with concatenated arrays."""
    files = sorted(glob.glob(os.path.join(folder, '*.exp')))
    if max_files:
        files = files[:max_files]
    F, L, T = [], [], []
    for p in files:
        f, l, t = decode_exp(p)
        F.append(f); L.append(l); T.append(t)
    f = np.concatenate(F); L = np.concatenate(L); t = np.concatenate(T)
    xmls = glob.glob(os.path.join(folder, '*.xml'))
    meta = parse_xml(xmls[0]) if xmls else {}
    return dict(freq=f, level=L, ft=t, files=files, meta=meta)


def sweep_matrix(f, L, t, decimals=5):
    """Arrange records into a (n_sweeps x n_channels) matrix using sweep
    boundaries detected as frequency wrap-around. Returns (uf, M, t_sweep)."""
    fr = np.round(f, decimals)
    uf = np.unique(fr)
    idx = np.searchsorted(uf, fr)
    # sweep boundaries: frequency decreases
    brk = np.flatnonzero(np.diff(fr) < 0) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(fr)]])
    M = np.full((len(starts), len(uf)), np.nan)
    ts = np.empty(len(starts), dtype='datetime64[us]')
    dt = ft_to_datetime64(t)
    for k, (s, e) in enumerate(zip(starts, ends)):
        M[k, idx[s:e]] = L[s:e]
        ts[k] = dt[s]
    return uf, M, ts
