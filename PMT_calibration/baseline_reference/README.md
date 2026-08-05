# Median baseline references

Keep one directory per acquisition configuration, especially when the PMT,
oscilloscope bandwidth limit, termination, sampling, or voltage scale changes.
Each directory contains one median-waveform `.npz` artifact per PMT voltage:

```text
baseline_reference/
├── dark_counts/
│   ├── 700V.npz
│   ├── 800V.npz
│   └── 900V.npz
├── no_bw_filter/
│   ├── 800V.npz
│   └── 850V.npz
└── 20MHz/
    ├── 800V.npz
    └── 850V.npz
```

`Baseline_reference_diagnostic.ipynb` creates these files. Set its
`reference_collection` to the directory name and narrow `reference_glob` to one
voltage before running it. `Batch_analysis.ipynb` then resolves the matching
artifact from each dataset's `baseline_reference_dir`.

Preferred filenames are `<voltage>V.npz`, such as `800V.npz`. The resolver also
accepts `<PMT>_<voltage>V.npz` or a full acquisition-name `.npz` when a directory
needs more-specific references.

The `.npz` artifacts are generated data and ignored by Git. The directories and
this naming convention are intentional; do not mix references from incompatible
acquisition configurations.
