# Sample data — August 2026

One month of ARGUS/ESMD frequency-sweep exports from three fixed monitoring stations
(R&S ESMD receivers). Each mission folder holds one band sub-folder with one `.exp` file per day.

The files contain only what the pipeline reads. The XML descriptors and the `EXP_LIST.EXP` index that ARGUS
writes next to the `.exp` files are not included, and the descriptor text embedded in each `.exp` file
(station identifiers, coordinates, host names, logs) has been overwritten with zero bytes. The measurement
records are unchanged: the notebook reproduces every figure and table of the paper from these files.

| folder   | station                    | band (MHz)  | step    | sweeps | size    |
|----------|----------------------------|-------------|---------|-------:|--------:|
| rRD_02   | Reșița (R)                 | 174–230     | 250 kHz | 31 064 | 174 MB  |
| rFM_04   | Reșița (R)                 | 87.5–108    | 100 kHz |    696 |   4 MB  |
| KRd001   | Timișoara–Kathrein (K)     | 174–230     | 250 kHz |    598 |   4 MB  |
| KFM001   | Timișoara–Kathrein (K)     | 87.5–108    | 100 kHz |    577 |   3 MB  |
| Yrd_00   | Ianova (Y)                 | 174–230     | 250 kHz |    374 |   2 MB  |
| Yfm_01   | Ianova (Y)                 | 80–110      | 100 kHz |    374 |   3 MB  |

Levels are RMS-detected field strengths in dBµV/m as logged by ARGUS; level 0.0 marks an invalid
record. Timestamps are UTC (local time = UTC+3 in August).

Licence: CC BY 4.0.
