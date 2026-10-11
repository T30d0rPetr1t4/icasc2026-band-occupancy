"""Statistical pipeline: histogram, log-chi-square (log-gamma) noise model,
Pfa thresholds, quantization-aware binning, model comparison, detrending and
spurious-channel classification.
"""
import numpy as np
from scipy import special, optimize, stats, ndimage

LN10_10 = np.log(10.0) / 10.0
QUANT = 0.1  # dB level quantum of the ESMD export


# ----------------------------------------------------------------- binning
def fd_bin_width(x):
    """Freedman–Diaconis width."""
    q75, q25 = np.percentile(x, [75, 25])
    return 2.0 * (q75 - q25) * x.size ** (-1.0 / 3.0)


def quantized_bin_width(x, q=QUANT, rule='fd'):
    """Bin width: an integer multiple of the level quantum q, never below q,
    rounded from the Freedman–Diaconis (or Scott) width."""
    if rule == 'scott':
        w = 3.49 * np.std(x) * x.size ** (-1.0 / 3.0)
    else:
        w = fd_bin_width(x)
    return q * max(1, int(round(w / q)))


def histogram_q(x, w, q=QUANT, lo=None, hi=None):
    """Histogram with edges aligned on the quantum grid (edges at k*w - q/2)."""
    lo = np.floor((x.min() if lo is None else lo) / w) * w - q / 2
    hi = np.ceil((x.max() if hi is None else hi) / w) * w + w
    edges = np.arange(lo, hi + w / 2, w)
    cnt, _ = np.histogram(x, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    return ctr, cnt, edges


def hist_mode(x, w=None):
    if w is None:
        w = quantized_bin_width(x)
    ctr, cnt, _ = histogram_q(x, w)
    return ctr[np.argmax(cnt)], w


# --------------------------------------------------- log-gamma noise model
# x = 10 log10 P, P ~ Gamma(k, theta).  Parametrised by (mu, k) with
# mu = 10 log10(k theta) = mode of x.
def lg_logpdf(x, mu, k):
    z = k * 10 ** ((x - mu) / 10.0)          # P/theta
    return np.log(LN10_10) + k * np.log(z) - z - special.gammaln(k)


def lg_cdf(x, mu, k):
    z = k * 10 ** ((x - mu) / 10.0)
    return special.gammainc(k, z)


def lg_sf(x, mu, k):
    z = k * 10 ** ((x - mu) / 10.0)
    return special.gammaincc(k, z)


def lg_threshold(mu, k, pfa):
    """Analytic threshold: P(x > T) = pfa."""
    z = special.gammainccinv(k, pfa)
    return mu + 10 * np.log10(z / k)


def lg_sigma(k):
    """Std of x in dB: 10/ln10 * sqrt(trigamma(k))."""
    return (10 / np.log(10)) * np.sqrt(special.polygamma(1, k))


def _binned_trunc_nll(params, ctr, cnt, w, xt, logpdf, cdf):
    """Poisson NLL of binned data truncated at xt (bins with centre <= xt)."""
    m = ctr <= xt
    # bin probabilities via cdf differences
    lo = ctr[m] - w / 2; hi = ctr[m] + w / 2
    p = cdf(hi, *params) - cdf(lo, *params)
    p = np.clip(p, 1e-300, None)
    p /= cdf(xt + w / 2, *params)
    n = cnt[m].sum()
    return -(cnt[m] * np.log(p)).sum() + n * np.log(1.0)  # multinomial


def fit_loggamma(x, trunc_above_mode=3.0, w=None):
    """Truncated ML fit of the log-gamma model on the histogram of x,
    using only bins up to (mode + trunc_above_mode) dB so that the signal
    tail does not bias the noise fit. Returns dict."""
    if w is None:
        w = quantized_bin_width(x)
    ctr, cnt, edges = histogram_q(x, w)
    mu0 = ctr[np.argmax(cnt)]
    xt = mu0 + trunc_above_mode
    lo_half = x[x <= mu0]
    s0 = max(0.3, np.sqrt(np.mean((lo_half - mu0) ** 2)))
    k0 = max(1.5, (4.343 / s0) ** 2)

    def nll(p):
        mu, lk = p
        return _binned_trunc_nll((mu, np.exp(lk)), ctr, cnt, w, xt, lg_logpdf, lg_cdf)
    res = optimize.minimize(nll, [mu0, np.log(k0)], method='Nelder-Mead',
                            options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    mu, k = res.x[0], float(np.exp(res.x[1]))
    return dict(mu=mu, k=k, w=w, mode_hist=mu0, xt=xt, ctr=ctr, cnt=cnt,
                sigma=lg_sigma(k), nll=res.fun, n_used=int(cnt[ctr <= xt].sum()))


# --- competing models (same truncated binned ML) -------------------------
def gauss_cdf(x, mu, s):
    return stats.norm.cdf(x, mu, s)


def gumbel_cdf(x, mu, b):
    # Gumbel (min) — the k=1 log-exponential limit has this shape
    return 1 - np.exp(-np.exp((x - mu) / b))


def fit_model(x, cdf, p0, trunc_above_mode=3.0, w=None):
    if w is None:
        w = quantized_bin_width(x)
    ctr, cnt, edges = histogram_q(x, w)
    mu0 = ctr[np.argmax(cnt)]
    xt = mu0 + trunc_above_mode

    def nll(p):
        if p[1] <= 0:
            return 1e300
        return _binned_trunc_nll(tuple(p), ctr, cnt, w, xt, None, cdf)
    res = optimize.minimize(nll, p0, method='Nelder-Mead',
                            options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    return dict(params=res.x, nll=res.fun, w=w, ctr=ctr, cnt=cnt, xt=xt)


def gof_stats(ctr, cnt, w, xt, cdf, params, n_par):
    """Chi-square/dof and KS distance on the truncated (noise) region."""
    m = ctr <= xt
    lo = ctr[m] - w / 2; hi = ctr[m] + w / 2
    p = cdf(hi, *params) - cdf(lo, *params)
    p = p / cdf(xt + w / 2, *params)
    n = cnt[m].sum()
    exp = n * p
    ok = exp > 5
    chi2 = ((cnt[m][ok] - exp[ok]) ** 2 / exp[ok]).sum()
    dof = ok.sum() - n_par - 1
    F_emp = np.cumsum(cnt[m]) / n
    F_mod = np.cumsum(p)
    ks = np.max(np.abs(F_emp - F_mod))
    return chi2 / max(dof, 1), ks, int(n)


# ---------------------------------------------------------- detrending
def channel_modes(M, w):
    """Histogram-max level of each column of the sweep matrix M."""
    modes = np.full(M.shape[1], np.nan)
    for j in range(M.shape[1]):
        c = M[:, j]; c = c[np.isfinite(c)]
        if c.size < 10:
            continue
        ctr, cnt, _ = histogram_q(c, w)
        modes[j] = ctr[np.argmax(cnt)]
    return modes


def noise_ridge(uf, modes, win=9, excl_db=3.0):
    """Interpolate the noise-band tilt through the column modes, excluding
    columns whose mode stands out from a running-median baseline (signal-
    dominated columns). Returns ridge (same length as uf) and mask of
    columns used."""
    base = ndimage.median_filter(modes, size=win, mode='nearest')
    good = np.isfinite(modes) & (modes - base <= excl_db)
    ridge = np.interp(uf, uf[good], modes[good])
    # light smoothing (3-point) to remove residual quantization steps
    ridge = ndimage.uniform_filter1d(ridge, 3, mode='nearest')
    return ridge, good


# ---------------------------------------------------------- classification
def fm_transmitters(uf, M, margin_db=20.0):
    """Identify transmitter centre frequencies in an FM-band sweep record:
    channels whose median level exceeds the band floor (20th percentile of
    channel medians) by margin_db, grouped into contiguous runs; centre =
    argmax of median within run."""
    med = np.nanmedian(M, 0)
    floor = np.percentile(med[np.isfinite(med)], 20)
    hot = med > floor + margin_db
    tx = []
    j = 0
    while j < len(uf):
        if hot[j]:
            k = j
            while k + 1 < len(uf) and hot[k + 1]:
                k += 1
            seg = slice(j, k + 1)
            jj = j + int(np.argmax(med[seg]))
            tx.append((uf[jj], med[jj]))
            j = k + 1
        else:
            j += 1
    return tx, floor


def arithmetic_matches(f, fm_list, tol):
    """Return list of (label) explanations of f as 2*fi or fi+fj."""
    out = []
    for fi in fm_list:
        if abs(f - 2 * fi) <= tol:
            out.append(f'2x{fi:.1f}')
    n = len(fm_list)
    for a in range(n):
        for b in range(a + 1, n):
            if abs(f - (fm_list[a] + fm_list[b])) <= tol:
                out.append(f'{fm_list[a]:.1f}+{fm_list[b]:.1f}')
    return out


def classify_channels(uf, D, ts, T, pfa, fm_list, tol, margin_max=15.0,
                      occ_cont=0.9, min_days_frac=0.5):
    """Rule set on the detrended sweep matrix D (levels relative to local
    floor) with threshold offset T. Returns list of dicts per channel."""
    n_sw = D.shape[0]
    days = ts.astype('datetime64[D]')
    uday = np.unique(days)
    A = D > T                                   # activity matrix
    occ = A.mean(0)
    # binomial upper bound of false-alarm occupancy
    occ_fa = pfa + 4 * np.sqrt(pfa * (1 - pfa) / n_sw)
    rows = []
    for j, f in enumerate(uf):
        a = A[:, j]
        act = occ[j] > occ_fa
        # days with activity above the false-alarm expectation
        nd = 0
        for d in uday:
            m = days == d
            if m.sum() and a[m].mean() > pfa + 4 * np.sqrt(pfa * (1 - pfa) / m.sum()):
                nd += 1
        exc = D[a, j]
        margin = float(np.median(exc)) if exc.size else 0.0
        p95 = float(np.percentile(exc, 95)) if exc.size else 0.0
        matches = arithmetic_matches(f, fm_list, tol)
        if not act:
            cls = 'noise'
        elif occ[j] >= occ_cont:
            cls = 'continuous'
        elif margin <= margin_max and nd >= min_days_frac * len(uday):
            cls = 'spurious'
        elif margin <= margin_max:
            cls = 'weak-sporadic'
        else:
            cls = 'signal'
        rows.append(dict(f=f, occ=occ[j], active=act, days=nd, ndays=len(uday),
                         margin=margin, p95=p95, cls=cls, matches=matches))
    return rows, A, occ_fa


def coactivation(A, idx):
    """Mean pairwise phi coefficient among columns idx of activity matrix A."""
    X = A[:, idx].astype(float)
    if X.shape[1] < 2:
        return np.nan, None
    C = np.corrcoef(X.T)
    iu = np.triu_indices(len(idx), 1)
    return float(np.nanmean(C[iu])), C
