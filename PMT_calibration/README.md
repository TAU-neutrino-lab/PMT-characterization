# PMT calibration and dark-count analysis

This directory contains the waveform selection, charge fitting, gain calibration,
and diagnostic notebooks used for PMT characterization. The notebooks support
both LED-triggered voltage scans and dark-count acquisitions. Large HDF5 files are
read in chunks, while reusable event features are stored in versioned pandas
caches.

Run Jupyter from this directory so that relative paths such as `PMT_Data/...`,
`plots/...`, and imports from `pmt` resolve correctly.

## Analysis workflow

```text
raw Keysight HDF5 files
          |
          | optional LED-off reference
          v
Baseline_reference_diagnostic.ipynb
          |
          | baseline_reference/<collection>/<voltage>V.npz
          v
Selection.ipynb  ------------------------------+
          |                                     |
          | versioned pre-cut *_df.pkl cache    | same cache
          v                                     v
Fit.ipynb                                 testing.ipynb
          |                                     |
          | fit-result YAML files               | per-voltage PDFs and
          v                                     | reduced diagnostic caches
Gain_calibration.ipynb                          v
                                        Batch_testing_summaries.ipynb
```

For a voltage sweep, [Batch_analysis.ipynb](Batch_analysis.ipynb) runs Selection
and Fit for each acquisition and then runs Gain calibration. For dark-count QA,
[Batch_testing_summaries.ipynb](Batch_testing_summaries.ipynb) runs the main
Testing analysis once per existing Selection cache.

## Notebook overview

| Notebook | Primary use | Main input | Main output |
| --- | --- | --- | --- |
| [Baseline_reference_diagnostic.ipynb](Baseline_reference_diagnostic.ipynb) | Build and validate an optional LED-off median baseline reference | Random/forced-trigger HDF5 files | One `<voltage>V.npz` reference |
| [Selection.ipynb](Selection.ipynb) | Preprocess one acquisition, find peaks, classify events, and cache features | One acquisition's HDF5 chunks | Versioned pre-cut `*_df.pkl` cache and selection diagnostics |
| [Fit.ipynb](Fit.ipynb) | Fit one voltage using the exact cached Selection configuration | One Selection cache | Fit YAML files and plots |
| [Gain_calibration.ipynb](Gain_calibration.ipynb) | Combine fit results across voltage | Fit-result YAML directory | Gain curves, parameter plots, and CSV summaries |
| [Batch_analysis.ipynb](Batch_analysis.ipynb) | Run Selection/Fit across one or more datasets and voltages | Dataset directories | All selection, fit, sweep-PDF, and gain outputs |
| [testing.ipynb](testing.ipynb) | Diagnose dark-count pedestal/signal classification and charge methods for one voltage | One Selection cache plus referenced HDF5 files | Testing summary PDF, shape diagnostics PDF, and reduced caches |
| [Batch_testing_summaries.ipynb](Batch_testing_summaries.ipynb) | Produce one Testing summary per voltage/date directory | A directory of Selection caches | One Testing PDF per acquisition |
| [Single_waveform_peak_diagnostics.ipynb](Single_waveform_peak_diagnostics.ipynb) | Explain why individual saved waveforms pass or fail `scipy.signal.find_peaks` | Two small `.npy` waveform examples | Inline threshold, width, SNR, and QDC diagnostics |

## Environment and data layout

The analysis requires Python with Jupyter, NumPy, pandas, Matplotlib, SciPy,
PyYAML, h5py, nbformat, and the local `lab_tools` package. In the laboratory
workspace, the existing environment can normally be activated with:

```bash
source ../../Lab-tools/.venv/bin/activate
jupyter lab
```

If a different environment is used, verify this before opening the notebooks:

```bash
python -c "import numpy, pandas, scipy, matplotlib, yaml, h5py, nbformat, lab_tools"
```

Acquisition chunks must share a prefix and end in `-<segment>.h5`:

```text
PMT_Data/Dark_Counts/trig20_20_8/
  WA0089_775V_Dark_20microV_trig-1.h5
  WA0089_775V_Dark_20microV_trig-2.h5
  ...
  WA0089_775V_Dark_20microV_trig-10.h5
```

The acquisition name entered in a notebook is the shared prefix without the
chunk suffix or `.h5` extension:

```python
file_name = "WA0089_775V_Dark_20microV_trig"
```

The voltage must appear as `_<number>V_` in the name. Batch discovery groups all
matching chunks into one acquisition and sorts acquisitions by voltage.

Raw `.h5` files, generated plots, `.pkl` caches, and NumPy artifacts are ignored
by Git. Timing CSVs may be committed because they are small run metadata.

## Quick start: one dark-count voltage

The safest first run is one voltage with a small `max_files` value, followed by
the complete acquisition after the plots look reasonable.

### 1. Run Selection

In the standalone configuration cell of [Selection.ipynb](Selection.ipynb), set
matching raw-data, output, acquisition, and channel values. For example:

```python
data_dir = Path("PMT_Data/Dark_Counts/trig20_20_8")
analysis_dir = Path("plots/dark_counts")
file_name = "WA0089_775V_Dark_20microV_trig"
channel = "Channel 3"

selection_output_dir = analysis_dir / "selection"
fit_inputs_path = analysis_dir / "fit_data" / "20_8"
save_plots = False
baseline_reference_spec = None
```

For a quick input check, use `max_files = 1`. For the production cache, restore
`max_files = None` and run all Selection cells in order.

The final pre-cut cache in this example is:

```text
plots/dark_counts/fit_data/20_8/
  WA0089_775V_Dark_20microV_trig_df.pkl
```

### 2. Optionally run Fit

Set the Fit cache directory to exactly the same directory used by Selection:

```python
analysis_dir = Path("plots/dark_counts")
file_name = "WA0089_775V_Dark_20microV_trig"
fit_inputs_path = analysis_dir / "fit_data" / "20_8"
fit_output_root = analysis_dir / "fit"

charge_method_names = ["fixed_pulse_window"]
fit_model_names = ["poisson"]
```

Then run [Fit.ipynb](Fit.ipynb) from the beginning. Fit does not recalculate
preprocessing. It loads the manifest inside the Selection cache and rebuilds the
same event selections.

### 3. Run Testing diagnostics

At the top of [testing.ipynb](testing.ipynb), select the same cache and choose
where the PDFs should be written:

```python
sample_acquisition = "WA0089_775V_Dark_20microV_trig"
sample_cache_path = (
    Path("plots/dark_counts/fit_data/20_8")
    / f"{sample_acquisition}_df.pkl"
)
testing_summary_pdf_path = (
    Path("plots/dark_counts_20_8")
    / f"{sample_acquisition}_testing_summary.pdf"
)
shape_diagnostics_pdf_path = (
    Path("plots/dark_counts_20_8")
    / f"{sample_acquisition}_waveform_shape_diagnostics.pdf"
)
```

Run only the first cell for the complete per-voltage analysis. The second and
third cells are optional rerun tools described below.

## Baseline reference diagnostic

[Baseline_reference_diagnostic.ipynb](Baseline_reference_diagnostic.ipynb) is
optional. Use it when a separate LED-off/random-trigger acquisition exists and
you want to subtract a pointwise baseline template instead of estimating a
baseline independently for every waveform.

The reference run should match the measurement in PMT, voltage, oscilloscope
channel, bandwidth/filter, termination, sampling, voltage scale, and record
length. Process only one voltage at a time:

```python
reference_data_dir = Path("PMT_Data/Baseline_Reference")
reference_glob = "*_800V_*.h5"
channel = "Channel 3"
reference_collection = "dark_counts"
```

Important controls include:

- `max_files` and `max_events`: optional quick-test limits; `None` uses all data.
- `chunk_size`: number of events read from HDF5 at once.
- `median_block_samples`: time-axis block size used to bound median RAM.
- `pulse_test_snr`: rejects likely asynchronous PMT pulses from the reference.
- `target_false_trigger_rates_hz`: accidental trigger rates for which empirical
  hardware-threshold recommendations are attempted.
- `minimum_expected_exceedances`: prevents extrapolating farther into the noise
  tail than the reference exposure supports.

The notebook uses an iterative pointwise median, diagnoses residual RMS and
frequency content, shows rejected reference traces, and writes:

```text
baseline_reference/dark_counts/800V.npz
```

To use the collection in Selection:

```python
baseline_reference_spec = Path("baseline_reference/dark_counts")
```

To use it in a Batch dataset:

```python
{
    "name": "dark_counts",
    "data_dir": Path("PMT_Data/Dark_Counts/trig20_20_8"),
    "analysis_dir": Path("plots/dark_counts_20_8"),
    "baseline_reference_dir": Path("baseline_reference/dark_counts"),
}
```

The directory is resolved by the voltage in each acquisition name. Setting the
reference to `None` explicitly selects per-waveform baseline subtraction; it
does not mean that a reference file is required.

## Selection notebook

[Selection.ipynb](Selection.ipynb) is the authoritative preprocessing stage.
Changing a parameter that affects the waveform, peak count, classification, or
cached charge requires rerunning Selection.

### What Selection does

1. Discovers all HDF5 chunks for one acquisition.
2. Rejects saturated events and optionally corrupt files.
3. Optionally rejects raw baseline windows containing pulse-like excursions.
4. Optionally learns a global baseline-RMS quantile and rejects noisier events.
5. Subtracts either the per-waveform baseline or a voltage-specific reference.
6. Finds negative PMT pulses by applying `scipy.signal.find_peaks` to `-V`.
7. Calculates peak amplitude, SNR, prominence, FWHM, rise/fall times, timing,
   multiplicity, and several charge estimators.
8. Learns signal-shape and trigger-time bounds from high-SNR dominant single
   pulses without using QDC as a classification input.
9. Optionally learns a common fixed pulse-integration window.
10. Builds named calibration selections and writes the pre-cut cache and manifest.

### Important preprocessing controls

- `baseline_window_ns`: baseline interval for events containing a peak.
- `require_clean_baseline`: checks raw baseline samples before subtraction so a
  pulse in the baseline cannot hide by inflating its own RMS.
- `baseline_rms_max_quantile`: for example, `0.95` removes the noisiest 5% of
  nonsaturated waveforms. Set it to `None` to disable this cut.
- `peak_snr_threshold`: minimum peak height divided by baseline RMS.
- `peak_prominence_snr`: minimum peak prominence divided by baseline RMS.
- `peak_distance_samples`: minimum candidate separation in samples. Convert it
  to time using the acquisition's sample spacing before copying settings between
  sampling rates.
- `peak_width_samples`: accepted FWHM range in samples. `(257, None)` means a
  minimum of 257 samples and no upper-width limit.
- `chunk_size`: bounds HDF5 RAM during the streaming pass.
- `waveform_sample_size`: only this diagnostic sample is retained as full arrays;
  the complete dataset remains feature-only.

### Learned signal classes

The high-SNR reference sample establishes learned ranges for FWHM, 10–90% rise
time, 90–10% fall time, and trigger time. Generic shape ranges may also impose
dataset-independent physical bounds.

The important cached labels are:

- `signal_good`: one detected peak with a dominant qualifying candidate, valid
  learned/generic shape, and trigger alignment.
- `signal_oscillatory`: additional signal-like candidates are too large relative
  to the primary candidate.
- `signal_with_small_additional_peaks`: additional candidates exist but are small
  enough relative to the primary pulse.
- `signal_bad_generic_shape`: dominant candidate outside generic shape bounds.
- `signal_noise_like`: candidate fails the learned pulse-shape bounds.
- `signal_off_time`: shape passes but peak timing is outside the learned interval.
- `n_peaks == 0`: no qualifying peak was found and Testing treats the event as
  pedestal-like.

`signal_classification_max_additional_peak_fraction` controls how large an
additional candidate may be relative to the main pulse. For example, `0.5`
accepts secondary candidates only when their amplitude is at most half the main
candidate amplitude.

### Calibration selection modes

- `standard`: exactly one detected peak plus LED timing.
- `timing_only`: LED timing only; multiple detected peaks are allowed.
- `loose_peak_multiplicity`: one through `max_allowed_peaks` plus LED timing.
- `dark_counts`: exactly one peak with no LED-timing requirement in the
  calibration-selection builder.

For each `cut_thresholds_snr` value, quality requirements apply only at or above
that SNR. Lower-SNR and no-peak events are intentionally retained to preserve the
pedestal needed by charge fits. `include_no_peak_cuts=True` also creates the
`no_peak_cuts` reference selection.

### Fixed pulse-window charge

When `enable_fixed_pulse_window_charge=True`, Selection learns one window from
high-SNR dominant single-pulse references. Its width is the selected quantile of
the CFD-10 falling-crossing time minus rising-crossing time, and its center is
the median CFD-10 midpoint. The exact learned bounds are then applied to every
event, including pedestal-like events. This avoids moving the integration window
to a random noise maximum for pedestals.

The reference controls are:

```python
fixed_pulse_window_reference_snr = 15.0
fixed_pulse_window_width_quantile = 0.95
fixed_pulse_window_min_reference_pulses = 100
fixed_pulse_window_read_chunk_size = 128
```

The final integration rereads retained events in chunks, so lowering the read
chunk size reduces peak RAM at the cost of additional loop overhead.

### Selection outputs

Selection writes:

- `<acquisition>_df.pkl`: the pre-cut event table and complete provenance
  manifest used by Fit and Testing.
- `<acquisition>_df_selected__<selection>.pkl`: self-describing selected tables
  for inspection.
- `<acquisition>_df_selected.pkl`: the default selected table.
- selection plots when `save_plots=True`.

The pre-cut cache is written atomically only after selection succeeds. Fit should
always use this cache rather than an older repository-level `fit_data` directory.

The later Baseline Study cells are optional exploratory plots. They operate on
the retained diagnostic waveform sample, not the complete raw acquisition.

## Fit notebook

[Fit.ipynb](Fit.ipynb) fits one voltage. It deliberately has no preprocessing or
peak-finding controls. Those values are loaded from the Selection cache manifest,
and the named selections are rebuilt exactly.

### Charge methods available to Fit

- `fixed_pulse_window`: the learned trigger-relative window applied identically
  to every event.
- `led_window`: one LED-synchronous window from `led_time_ns - pre_led_ns` through
  `led_time_ns + post_led_ns`; clipped-window coverage is cached and reported.
- `full_waveform`: the baseline-subtracted full recorded waveform, useful as a
  systematic comparison.

Several methods may be fit in one run:

```python
charge_method_names = ["fixed_pulse_window", "led_window"]
fit_model_names = ["poisson", "bellamy"]
maxPE = 6
nbins = 250
```

This produces a fit for every requested method/model/selection combination; it
does not combine different charge methods into one fit.

### Fit models

- `poisson`: faster first-pass SPE model.
- `bellamy`: adds an under-amplified SPE component and is more flexible but more
  expensive.

Fit parameter specifications have the form:

```python
[initial_value, lower_bound, upper_bound, is_fixed]
```

Ranges and parameters can be overridden without editing the model cells:

```python
fit_range_overrides = {
    "fixed_pulse_window": (-10.0, 80.0),
    (
        "fixed_pulse_window",
        "dark_count_quality_above_snr15",
    ): (-5.0, 70.0),
}

fit_parameter_overrides = {
    (
        "poisson",
        "fixed_pulse_window",
        "dark_count_quality_above_snr15",
    ): {
        "q1_mV_ns": [20.0, 5.0, 80.0, False],
    },
}
```

Override precedence is model, then `(model, method)`, then
`(model, method, selection)`. Fit-range precedence is `(method, selection)`, then
method, then the finite sample range.

After the cache and selections have loaded, fit cells may be rerun repeatedly
with new ranges or parameter guesses without rereading the full raw dataset.
Only small requested waveform examples and rejected-event samples access HDF5.

Fit results are stored below:

```text
<fit_output_root>/
  <voltage>V/                 # plots
  fit_results/                # YAML result files
```

## Gain calibration notebook

[Gain_calibration.ipynb](Gain_calibration.ipynb) reads Fit YAML files and builds
consistent voltage scans. Select one model, integration method, and selection for
the main curve:

```python
results_dir = Path("plots/20MHz/fit/fit_results")
output_dir = Path("plots/20MHz/gain_calibration")

fit_model = "poisson"
integration = "fixed_pulse_window"
selection = "pulse_quality_above_snr15"
excluded_voltages_V = []
```

The gain model is

```text
G(V) = G_ref (V / V_ref)^k
```

At least three usable voltages are recommended before interpreting parameter
uncertainties. `excluded_voltages_V` keeps runs in the all-fits CSV but removes
them from voltage plots and power-law calibration.

Outputs include:

- `gain_calibration_all_fits.csv`
- `gain_curve.png`
- `fit_parameters_vs_voltage.png`
- one gain curve per exact analysis choice
- `gain_calibrations_by_analysis.csv`
- compact gain-calibration grids grouped by fit model

Do not mix model, integration, or selection choices within one voltage curve.

## Batch analysis notebook

[Batch_analysis.ipynb](Batch_analysis.ipynb) is the normal voltage-sweep driver.
It executes Selection and Fit in the current kernel, using `batch_*` variables to
override their standalone configuration cells.

### Dataset configuration

Add one dictionary per acquisition family:

```python
batch_datasets = [
    {
        "name": "20MHz",
        "data_dir": Path("PMT_Data/20MHz"),
        "analysis_dir": Path("plots/20MHz"),
        "baseline_reference_dir": None,
    },
]

batch_channel = "Channel 4"
```

`name` is only the batch label. `data_dir` must be the directory containing the
HDF5 files. For the 20 MHz example the correct input is
`Path("PMT_Data/20MHz")`, while acquisition names are discovered from filenames.

Use `file_names` for a quick subset:

```python
file_names = ["WA0089_900V_20MHz_led100"]
```

Use `file_names = None` to discover and process all acquisition prefixes in each
dataset directory.

### Execution controls

- `run_selection=True`: reread raw data and rebuild caches. Required after any
  preprocessing, peak-finding, classification, or fixed-window change.
- `run_selection=False`: reuse existing caches and their stored settings.
- `run_fit=True`: run charge-model fits from the caches.
- `diagnostic_only=True`: ignore `run_selection`/`run_fit`, recreate Fit waveform
  and selection diagnostics from caches, and stop before charge fitting.
- `continue_on_error=True`: record a failed voltage and continue the sweep.
- `batch_save_plots`: save figures under the dataset analysis directory.
- `batch_show_diagnostic_plots`: control inline display during the sweep.

Useful combinations are:

```python
# Rebuild everything after changing peak finding.
run_selection = True
run_fit = True
diagnostic_only = False

# Refit existing caches with new fit ranges or parameter guesses.
run_selection = False
run_fit = True
diagnostic_only = False

# Recreate cache-based diagnostics without fitting.
diagnostic_only = True

# Rebuild only the voltage-sweep PDF from existing caches.
run_selection = False
run_fit = False
diagnostic_only = False
batch_create_voltage_sweep_diagnostics_pdf = True
```

### Voltage-sweep PDF

`voltage_sweep_diagnostics.pdf` overlays unfitted charge histograms and sampled
high-SNR/pedestal waveforms across voltage. The report methods are controlled
independently of the fit grid:

```python
batch_voltage_sweep_charge_methods = [
    "fixed_pulse_window",
    "led_window",
    "full_waveform",
]
batch_voltage_sweep_selection_names = ["no_peak_cuts"]
batch_voltage_sweep_waveforms_per_voltage = 3
```

Listing multiple charge methods creates pages for those methods in one dataset
PDF; it does not create a separate PDF for each method.

The normal batch output tree is:

```text
plots/<dataset>/
  selection/
  fit_data/
  fit/
    <voltage>V/
    fit_results/
  gain_calibration/
  voltage_sweep_diagnostics.pdf
```

The batch summary table shows one row per acquisition. A printed `...` from a
pandas display only abbreviates the table; the numbered acquisition headings and
final status rows determine whether every voltage was processed.

## Testing notebook

[testing.ipynb](testing.ipynb) is the detailed dark-count QA notebook. It uses
Selection's cached features to choose populations, then streams the corresponding
raw waveforms in small chunks. It does not load all waveforms into RAM.

### Main-cell populations

- Pedestal population: cached `n_peaks == 0` events.
- Initial signal population: cached `signal_good` events passing the configured
  Testing sample filters.
- Final blue signal population: initial signals that survive the Testing-only
  normalized-template mismatch cut.

Multi-peak, oscillatory, off-time, learned-shape failures, and generic-shape
failures are shown in diagnostic figures but are not silently added to the blue
signal population. QDC is not used to create the original waveform labels, so it
can be used to evaluate their charge separation.

The Testing mismatch cut affects later main-summary plots and reduced
multi-voltage caches but does not rewrite the Selection cache. Pedestal growth
and local-RMS metrics are currently diagnostic-only in the main cell.

### Charge methods compared in Testing

1. **Learned fixed window**: the exact Selection window applied to both pedestal
   and signal events.
2. **Event-specific method**: signals integrate from strongest-candidate time
   minus one FWHM through peak time plus two FWHM; pedestals integrate 0–80 ns.
   Pedestal waveforms use their full-trace mean and RMS for this 0–80 ns study.
3. **Raw full trace**: integrate `-V_raw` over the entire trace for both
   populations without baseline subtraction.

Testing also compares a pedestal full-trace integral whose baseline is learned
from configurable sidebands. A full-trace mean is not used for that integral,
because subtracting the mean of the same complete trace would force its integral
to zero by construction.

### Main output and diagnostics

The first cell produces the per-voltage `*_testing_summary.pdf`, including:

- noisiest retained waveforms and baseline region
- sampled pedestal and signal waveforms
- linear and logarithmic QDC histograms
- peak amplitude, QDC, SNR, FWHM, and rise-time distributions
- amplitude-versus-QDC and rise-time-versus-QDC density plots
- automatic pedestal/signal QDC-overlap examples
- high-QDC tails and low-QDC signal examples
- multi-peak, oscillatory, generic-shape, and wide-pulse examples
- fixed-window, event-specific, pedestal-sideband, and raw-full-trace comparisons
- automatic QDC valley separators and TP/TN/FP/FN rates for each integration
  method, using waveform classification as the reference label

It also writes a separate waveform-shape diagnostics PDF and two reduced caches:

```text
<testing output>/
  shape_metric_cache/<acquisition>_shape_metrics.pkl
  voltage_qdc_method_cache/<acquisition>_qdc_methods.npz
```

The QDC separator defaults to the valley in a class-balanced smoothed density so
the usually much larger pedestal sample cannot move the separator toward the
signal mode. Set `qdc_separator_balance_classes=False` to use raw combined counts.
TP means signal-labelled and right of the separator; TN means pedestal-labelled
and left; FP means pedestal-labelled and right; FN means signal-labelled and
left. These are agreement metrics relative to waveform-based labels, not an
independent physical truth sample.

### Sampling and RAM controls

```python
max_pedestal_waveforms_for_qdc = 300_000
max_peak_waveforms_for_qdc = 300_000
qdc_read_chunk_size = 128

n_pedestal_waveforms_to_plot = 10
n_peak_waveforms_to_plot = 10
```

The two maximum values cap how many unique events contribute to QDC calculations.
If a requested maximum exceeds the available population, Testing uses every
available event without repetition. Plot counts are independent and only retain
the small number of waveforms shown. Lower `qdc_read_chunk_size` if memory is
tight.

### Timing CSVs and rate units

When `qdc_use_timing_rate_units=True`, Testing looks beside the raw HDF5 files for
a CSV named `<acquisition>_timing.csv`. Required columns are `Filename`,
`Segments`, and the selected time column. For example:

```csv
Filename,Segments,Acquisition_s,Rate_evt_s,Save_s,Batch_s
WA0089_775V_Dark_20microV_trig-1.h5,30000,10.219657,2935.519293,22.729581,32.966744
```

With `qdc_timing_time_column="Acquisition_s"`, histogram weights are reported in
waveforms/s/bin using acquisition live time. `Batch_s` includes save/dead time.
If a matching and complete timing CSV is unavailable, Testing prints a message
and uses event counts rather than mixing units.

### Optional second cell: voltage comparison

The second cell is independent and may be run after a fresh kernel start. It
loads only reduced QDC method caches created by successful first-cell runs:

```python
voltage_qdc_comparison_voltages = (750, 775, 800, 825, 850)
voltage_qdc_comparison_cache_dir = Path(
    "plots/dark_counts/fit_data/20_8"
)
voltage_qdc_comparison_output_dir = Path("plots/dark_counts_20_8")
```

The voltage tuple decides which curves appear; it does not decide which caches
are saved. Every successful main Testing run saves its own reduced cache. Missing
or stale cache versions are skipped with an instruction to run the first cell for
that voltage.

The resulting `normalized_qdc_methods_voltage_comparison.pdf` contains method
comparisons across voltage and separate pedestal/signal views. Despite the
historical filename, whether histograms are normalized is controlled by
`voltage_qdc_comparison_density`.

### Optional third cell: fast shape replot

The third cell rebuilds only the separate waveform-shape diagnostics PDF from
the reduced shape-metric cache. It is useful for changing:

- signal mismatch rejection fraction
- pedestal growth threshold
- pedestal local-RMS variation threshold
- diagnostic histogram bins and number of displayed waveforms
- the fixed-window QDC range used to select comparison examples

It reads raw files only for the few waveforms displayed. Run the full first cell
instead after changing event sampling, baseline/QDC definitions, template
construction, source Selection cache, or any setting that should change the main
testing summary or multi-voltage cache.

## Batch testing summaries

[Batch_testing_summaries.ipynb](Batch_testing_summaries.ipynb) runs the main
Testing cell sequentially for every cache matching a date directory. Configure:

```python
date_tags = ["20_8"]
testing_fit_data_root = Path("plots/dark_counts/fit_data")
testing_summary_output_root = Path("plots")
testing_cache_glob = "*_Dark_20microV_trig_df.pkl"
```

This reads caches from:

```text
plots/dark_counts/fit_data/20_8/
```

and writes per-voltage PDFs below:

```text
plots/dark_counts_20_8/
```

Voltages are processed one at a time, figures are closed after saving, and raw
waveforms remain chunked to avoid accumulating an entire sweep in RAM.

## Single-waveform peak diagnostics

[Single_waveform_peak_diagnostics.ipynb](Single_waveform_peak_diagnostics.ipynb)
is a small, standalone explanation tool. It expects:

```text
time_and_example_wf_pedestal_peak.npy
time_and_example_wf_pedestal_noise.npy
```

Each file contains one time array and one waveform array. A compatible save
format is:

```python
payload = np.empty((1, 2), dtype=object)
payload[0, 0] = time_ns
payload[0, 1] = waveform_mV
np.save("time_and_example_wf_pedestal_peak.npy", payload)
```

Match these controls to the Selection run being diagnosed:

```python
baseline_window_ns = (0.0, 20.0)
peak_snr_threshold = 8.0
peak_prominence_snr = 6.0
peak_distance_samples = 1024
peak_width_samples = (64, None)
```

The notebook plots the baseline, height/prominence thresholds, strongest local
candidate, FWHM, learned fixed charge window, event-specific QDC window, and
pedestal QDC window. It also prints every pass/fail decision used by
`find_peaks`. This notebook is for two hand-picked examples; use `testing.ipynb`
for statistically representative populations.

## Which notebook must be rerun?

| Change | Minimum rerun |
| --- | --- |
| Raw directory, channel, baseline mode/window, saturation, RMS quantile, clean-baseline check | Selection, then any downstream Fit/Testing products |
| Peak SNR, prominence, distance, width, learned shape/timing, fixed pulse window | Selection, then downstream notebooks |
| Selection SNR thresholds or selection mode | Selection to update the authoritative cache manifest, then Fit |
| Fit model, `maxPE`, fit range, bins, starting values, parameter bounds | Fit only |
| Gain scan choice or excluded voltages | Gain calibration only |
| Testing waveform display counts, QDC histogram bins, overlap/tail ranges | Main Testing cell only |
| Testing mismatch fraction used in the main populations and QDC caches | Main Testing cell only; no Selection rerun |
| Shape-diagnostics-only mismatch/growth/RMS thresholds | Optional third Testing cell |
| Voltages, alpha, bins, or axes in the voltage comparison | Optional second Testing cell |
| Batch voltage-sweep PDF display settings | Batch with Selection/Fit disabled, using existing caches |

## Troubleshooting

### Fit says the cache is missing or predates provenance

Check that `fit_inputs_path` exactly matches Selection's output directory. Old
caches without a current manifest must be regenerated by Selection.

### An old cache points to a missing HDF5 path

Testing and Fit resolve cached sources by filename under `PMT_Data` when possible.
If files were renamed or moved outside that tree, restore the original filenames
or rerun Selection so the manifest records the current sources.

### Baseline reference is unexpectedly required

Use:

```python
baseline_reference_spec = None
```

or set a Batch dataset's `baseline_reference_dir` explicitly to `None`. This
selects per-waveform baseline-window subtraction.

### Testing reports no RMS cut

This is valid when Selection used `baseline_rms_max_quantile=None`. Testing prints
that the global RMS cut was disabled and continues.

### Testing has no pedestal events

High-voltage data may classify every retained event as signal-like. Pedestal-only
plots and QDC confusion/separator metrics require both populations and are skipped;
signal diagnostics continue.

### A requested Testing sample is larger than the population

The notebook takes all available events without replacement. It does not raise an
error merely because a QDC cap such as `300_000` is larger than the population.

### RAM use is high

- Lower Selection `chunk_size` or fixed-window read chunk size.
- Lower Testing `qdc_read_chunk_size`.
- Keep waveform plot counts small; QDC calculation caps are separate.
- Do not replace streamed loaders with a one-go HDF5 load.
- Clear old notebook outputs and restart the kernel after an interrupted run.
- Keep the Baseline-reference median cache on disk rather than a RAM-backed
  temporary directory.

### The multi-voltage PDF omits a voltage

Run the main Testing cell for that voltage with the current code and settings.
The optional comparison cell rejects missing or stale reduced-cache versions.

### A PDF was interrupted

The main Testing summary is written through a `.partial` file and replaces the
final PDF only after successful completion. Rerun the main cell; stale partial
files are cleaned at the beginning of the next run.

