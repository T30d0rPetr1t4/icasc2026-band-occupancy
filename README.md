# Band occupancy and spurious-emission classification from radio-monitoring archives

Code, sample data and results accompanying the paper

> T. Petrița and R. Dobre-Baron, *A Statistical Computing Pipeline for Band Occupancy and
> Spurious-Emission Classification from Radio Monitoring Archives*, Proc. Int. Conf. Advanced
> Scientific Computing (ICASC 2026), Cluj-Napoca, Romania, Oct. 2026.

The pipeline decodes the proprietary `.exp` sweep exports of the Rohde & Schwarz ARGUS/ESMD
monitoring system, builds the level histogram and the frequency–level occurrence heatmap, fits a
log-χ² noise model with analytic false-alarm thresholds, detrends the heatmap against the noise
ridge (frequency) and the band floor (time), and classifies channel activity with a rule set
(occupancy, margin, recurrence, FM frequency arithmetic, co-activation). Every figure and table of
the paper is reproduced by one notebook from one month of `.exp` files.

## Repository layout

```
notebooks/paper_figures_from_exp.ipynb   the complete pipeline, executed on the sample data (Figs. 2-5, Tables I-III, plus supplementary figures kept for the journal extension)
src/expio.py                             .exp decoder (26-byte records, automatic header detection), optional XML metadata, sweep matrix
src/pipeline.py                          histogram, noise model, thresholds, GOF, detrending, FM extraction, classification
data/2026.08/<mission>/<band>/           sample month (August 2026), see data/README.md
results/                                 figures (PDF + PNG), tables (CSV) and results.json produced by the notebook
requirements.txt                         NumPy, SciPy, pandas, Matplotlib, Jupyter
```

## Quick start

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook paper_figures_from_exp.ipynb      # Run All: ~15 s for the sample month
```

Cell 1 holds every parameter. To process a different archive point `WORK_ROOT` to a folder with the
same layout (`<mission>/<band>/*.exp`; the ARGUS XML descriptors are optional) and list the stations in `STATIONS`
(band-III mission, FM mission, label).

To use the decoder outside the notebook:

```python
import sys; sys.path.append('src')
from expio import load_mission, sweep_matrix
b = load_mission('data/2026.08/rRD_02/0101')          # all .exp files of one band folder
uf, M, ts = sweep_matrix(b['freq'], b['level'], b['ft'])   # channels (MHz), sweeps x channels (dBµV/m), timestamps
```

## `.exp` sweep export format (reverse-engineered)

Little-endian records of 26 bytes after a variable-length header:

| offset | type    | content                          |
|-------:|---------|----------------------------------|
| 0      | uint64  | Windows FILETIME (100-ns ticks since 1601-01-01, UTC) |
| 8      | float64 | frequency, Hz                    |
| 16     | float64 | level, dBµV/m (0.1 dB quantum)   |
| 24     | uint16  | padding                          |

The data block is located by searching for the first run of six consecutive plausible FILETIME
values at the 26-byte stride. A sweep boundary is a frequency wrap-around.

## Goodness-of-fit statistics (`pipeline.gof_stats`)

Noise models are compared on the *truncated noise region* of the level histogram (bins up to the
histogram mode + 3 dB, quantization-aware bin width from eq. (1) of the paper). Each model is fitted
by truncated, binned maximum likelihood with bin probabilities renormalised on that region, and two
statistics are then computed on the same bins:

* reduced chi-square, χ²/ν = Σ (c_i − E_i)²/E_i over bins with expected count E_i = n·p_i > 5,
  ν = bins − parameters − 1;
* Kolmogorov–Smirnov distance, KS = max |F_emp − F_mod| between the empirical and model
  cumulative distributions evaluated at the bin edges (a distance, not a test with a p-value).

With ≈ 5·10⁶ samples the critical KS value is ≈ 6·10⁻⁴ and χ²/ν should be ≈ 1, so every
two-parameter model is formally rejected; the statistics are used to rank the models
(log-χ², Gaussian in dB, Gumbel, log-exponential), and the ranking is the same under both statistics
and before/after ridge detrending. Out-of-sample behaviour is checked separately by split-half
validation (odd/even sweeps). See section 13 of the notebook for the worked numbers.

## Citing

Please cite the paper above. A `CITATION.cff` file is included.

## Licence

Code: MIT (see `LICENSE`). Sample data: CC BY 4.0; see `data/README.md`.
