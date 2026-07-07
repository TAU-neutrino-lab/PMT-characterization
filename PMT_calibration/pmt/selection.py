import numpy as np
import pandas as pd


# ----------------------------------------------------------------
# Class
# ----------------------------------------------------------------

class CutFlow:
    """
    Helper class to build a cumulative event selection.

    Example
    -------
    cutflow = CutFlow(len(df))

    cutflow.apply(
        "Baseline pulse",
        reject_baseline_pulses(...),
    )

    cutflow.apply(
        "SNR > 5",
        df["snr"] > 5,
    )

    df_sel = df[cutflow.mask].copy()
    waveforms_sel = waveforms[cutflow.mask]
    """

    def __init__(self, n_events):

        self._initial = int(n_events)
        self._mask = np.ones( n_events, dtype=bool )

        self._rows = [
            {
                "cut": "Initial",
                "n_kept": self._initial,
                "relative_efficiency": 1.0,
                "cumulative_efficiency": 1.0,
            }
        ]

    @property
    def mask(self):
        return self._mask

    def apply(self, name, mask):

        mask = np.asarray(mask)

        if mask.shape != self._mask.shape: raise ValueError( "Mask has incorrect length." )

        before = self._mask.sum()
        self._mask &= mask
        after = self._mask.sum()
        self._rows.append(
            {
                "cut": name,
                "n_kept": int(after),
                "relative_efficiency": ( after / before if before > 0 else np.nan ),
                "cumulative_efficiency": ( after / self._initial ),
            }
        )
        return self._mask

    def dataframe(self):

        return pd.DataFrame(self._rows)

    def print(self):

        df = self.dataframe()

        print( f"{'Cut':<25}" f"{'Kept':>10}" f"{'Relative':>12}" f"{'Cumulative':>14}" )
        print("-" * 62)
        for _, row in df.iterrows():
            print(
                f"{row['cut']:<25}"
                f"{row['n_kept']:>10d}"
                f"{100*row['relative_efficiency']:>11.1f}%"
                f"{100*row['cumulative_efficiency']:>13.1f}%"
            )


# ----------------------------------------------------------------
# Individual Cuts
# ----------------------------------------------------------------

def reject_baseline_pulses( df, baseline_window_ns, min_peak_height_mV=0.5 ):

    reject = ( df["peak_time_ns"].between( *baseline_window_ns ) & (df["peak_amplitude_mV"] >= min_peak_height_mV) )

    return ~reject