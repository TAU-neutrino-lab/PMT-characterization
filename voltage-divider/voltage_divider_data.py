"""Utilities for reading voltage-divider measurement and simulation data.

Typical notebook usage:

    from voltage_divider_data import load_all_data
    data = load_all_data()

Then use, for example, ``data.measurement_df``, ``data.sim_df``, or
``data.comparison_df``. Call ``load_all_data(include_intermediate=True)``
to also return ``measurement_raw_df`` and ``sim_raw_df``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
DEFAULT_EXCEL_PATH = DATA_DIR / "VoltageDivider_v2_tests.xlsx"
DEFAULT_SIMULATION_PATH = DATA_DIR / "simulation_v2.csv"

MEASURED_PIN_ORDER = [
    "K-G",
    "G-Dy1",
    "Dy1-Dy2",
    "Dy2-Dy3",
    "Dy3-Dy4",
    "Dy4-Dy5",
    "Dy5-Dy6",
    "Dy6-Dy7",
    "Dy7-Dy8",
    "Dy8-Dy9",
    "Dy9-Dy10",
    "Dy10-P",
]

CUMULATIVE_PIN_ORDER = ["K-G"] + [f"K-Dy{i}" for i in range(1, 11)] + ["K-P"]
SIM_PIN_ORDER = ["HV"] + CUMULATIVE_PIN_ORDER

SIMULATION_PIN_MAPPINGS = {
    "V(net-_g1-pin_1_)": "K-G",
    "V(net-_d1-pin_1_)": "K-Dy1",
    "V(net-_d2-pin_1_)": "K-Dy2",
    "V(net-_d3-pin_1_)": "K-Dy3",
    "V(net-_d4-pin_1_)": "K-Dy4",
    "V(net-_d5-pin_1_)": "K-Dy5",
    "V(net-_d6-pin_1_)": "K-Dy6",
    "V(net-_d7-pin_1_)": "K-Dy7",
    "V(net-_d8-pin_1_)": "K-Dy8",
    "V(net-_d9-pin_1_)": "K-Dy9",
    "V(net-_d10-pin_1_)": "K-Dy10",
    "V(net-_p1-pin_1_)": "K-P",
    "V(net-_hv1-in_)": "HV",
}

SPICE_PREFIX_MULTIPLIERS = {
    "": 1.0,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "\u00b5": 1e-6,
    "\u03bc": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "K": 1e3,
    "meg": 1e6,
    "Meg": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
}


@dataclass(frozen=True)
class VoltageDividerData:
    """Default container returned by ``load_all_data``."""

    measurement_df: pd.DataFrame
    sim_df: pd.DataFrame
    comparison_df: pd.DataFrame


@dataclass(frozen=True)
class VoltageDividerDataWithIntermediate:
    """Container returned by ``load_all_data(include_intermediate=True)``."""

    measurement_raw_df: pd.DataFrame
    measurement_df: pd.DataFrame
    sim_raw_df: pd.DataFrame
    sim_df: pd.DataFrame
    comparison_df: pd.DataFrame


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    return default if path is None else Path(path).expanduser().resolve()


def read_measurement_raw(
    excel_path: str | Path | None = None,
    sheet_name: str = "Sheet1",
    start_row: int = 20,
    end_row: int = 33,
) -> pd.DataFrame:
    """Read the Excel measurement block as a raw dataframe.

    Excel row numbers are 1-based and inclusive. By default this reads rows
    20-33 from all columns.
    """

    excel_path = _resolve_path(excel_path, DEFAULT_EXCEL_PATH)
    nrows = end_row - start_row + 1
    measurement_raw_df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        skiprows=start_row - 1,
        nrows=nrows,
        engine="openpyxl",
    )
    return measurement_raw_df.reindex(range(nrows))


def build_measurement_dataframe(
    measurement_raw_df: pd.DataFrame,
    pin_order: list[str] | None = None,
    infer_missing_kg: bool = True,
) -> pd.DataFrame:
    """Build measured step and cumulative voltage dataframes from ``measurement_raw_df``.

    Returns one dataframe with the measured adjacent-pin voltages and the
    cumulative voltage from K to each pin.
    """

    if pin_order is None:
        pin_order = MEASURED_PIN_ORDER
    pin_order_index = {pin: index for index, pin in enumerate(pin_order)}

    records = []
    for start_col in range(0, measurement_raw_df.shape[1], 3):
        input_label = measurement_raw_df.iat[0, start_col]
        if pd.isna(input_label) or start_col + 1 >= measurement_raw_df.shape[1]:
            continue

        input_match = re.search(r"(\d+(?:\.\d+)?)", str(input_label))
        if input_match is None:
            continue

        block = measurement_raw_df.iloc[2:, [start_col, start_col + 1]].copy()
        block.columns = ["measured_between", "measured_voltage_V"]
        block["measured_voltage_V"] = pd.to_numeric(
            block["measured_voltage_V"], errors="coerce"
        )
        block = block.dropna(subset=["measured_between", "measured_voltage_V"])
        block["input_voltage_V"] = float(input_match.group(1))
        records.append(block)

    if not records:
        raise ValueError("No measurement blocks were found in the raw Excel data.")

    parsed_df = pd.concat(records, ignore_index=True)
    unknown_pins = sorted(set(parsed_df["measured_between"]) - set(pin_order))
    if unknown_pins:
        raise ValueError(f"Unexpected pin labels in Excel data: {unknown_pins}")

    frames = []
    for input_voltage, group in parsed_df.groupby("input_voltage_V", sort=True):
        group = group.copy()

        # New workbooks include measured K-G. This fallback supports older files.
        if infer_missing_kg and "K-G" not in set(group["measured_between"]):
            inferred_kg = input_voltage - group["measured_voltage_V"].sum()
            group = pd.concat(
                [
                    group,
                    pd.DataFrame(
                        {
                            "input_voltage_V": [input_voltage],
                            "measured_between": ["K-G"],
                            "measured_voltage_V": [inferred_kg],
                        }
                    ),
                ],
                ignore_index=True,
            )

        group["pin_order"] = group["measured_between"].map(pin_order_index)
        group = (
            group.drop_duplicates(subset="measured_between", keep="last")
            .sort_values("pin_order")
            .drop(columns="pin_order")
            .reset_index(drop=True)
        )
        group["cumulative_voltage_from_K_V"] = group["measured_voltage_V"].cumsum()
        group["computed_between"] = "K-" + group["measured_between"].str.split("-").str[-1]
        frames.append(group)

    parsed_df = pd.concat(frames, ignore_index=True)
    measurement_df = parsed_df[
        [
            "input_voltage_V",
            "measured_between",
            "measured_voltage_V",
            "computed_between",
            "cumulative_voltage_from_K_V",
        ]
    ].copy()
    measurement_df[["measured_voltage_V", "cumulative_voltage_from_K_V"]] = measurement_df[
        ["measured_voltage_V", "cumulative_voltage_from_K_V"]
    ].round(6)

    return measurement_df


def read_measurements(
    excel_path: str | Path | None = None,
    sheet_name: str = "Sheet1",
    start_row: int = 20,
    end_row: int = 33,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read measured Excel data.

    Returns ``(measurement_raw_df, measurement_df)``.
    """

    measurement_raw_df = read_measurement_raw(
        excel_path=excel_path,
        sheet_name=sheet_name,
        start_row=start_row,
        end_row=end_row,
    )
    measurement_df = build_measurement_dataframe(measurement_raw_df)
    return measurement_raw_df, measurement_df


def convert_spice_value(value: object) -> float:
    """Convert a SPICE value such as ``7.27273uA`` or ``19.9273V`` to float."""

    text = str(value).strip()
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)",
        text,
    )
    if match is None:
        warnings.warn(f"Could not convert simulation value: {text!r}")
        return np.nan

    number = float(match.group(1))
    suffix = match.group(2)

    # Values are currents or voltages, so strip the physical unit and keep the SI prefix.
    prefix = suffix[:-1] if suffix.endswith(("A", "V")) else suffix
    if prefix not in SPICE_PREFIX_MULTIPLIERS:
        warnings.warn(f"Unexpected unit prefix {prefix!r} in simulation value: {text!r}")
        return np.nan

    return number * SPICE_PREFIX_MULTIPLIERS[prefix]


def format_hv_column(value: float) -> int | float:
    """Use integer HV labels when the simulated HV is effectively an integer."""

    return int(round(value)) if np.isclose(value, round(value)) else value


def read_sim(
    filename: str | Path | None = None,
    pin_mappings: dict[str, str] | None = None,
    hv_parameter: str = "V(net-_hv1-in_)",
) -> pd.DataFrame:
    """Read the simulation CSV into a dataframe indexed by simulation parameter."""

    filename = _resolve_path(filename, DEFAULT_SIMULATION_PATH)
    if pin_mappings is None:
        pin_mappings = SIMULATION_PIN_MAPPINGS

    df = pd.read_csv(
        filename,
        sep=":",
        header=None,
        names=["Parameter", "Value"],
        usecols=[0, 1],
        skip_blank_lines=True,
    ).dropna(subset=["Parameter", "Value"])

    df["Parameter"] = df["Parameter"].astype(str).str.strip()
    df["Value"] = df["Value"].astype(str).str.strip().apply(convert_spice_value)

    parameter_order = df["Parameter"].drop_duplicates()
    df["run_index"] = df.groupby("Parameter").cumcount()
    final_df = (
        df.pivot(index="Parameter", columns="run_index", values="Value")
        .reindex(parameter_order)
        .rename_axis(index=None, columns=None)
    )

    if hv_parameter in final_df.index:
        hv_values = final_df.loc[hv_parameter].dropna().to_list()
        if len(hv_values) == final_df.shape[1]:
            final_df.columns = [format_hv_column(value) for value in hv_values]
            final_df.columns.name = "simulation_setting"
        else:
            warnings.warn(
                "Mismatch between simulation runs and HV values; keeping default column labels."
            )
    else:
        warnings.warn(f"HV parameter {hv_parameter!r} was not found; keeping default column labels.")

    final_df["pins"] = [pin_mappings.get(parameter) for parameter in final_df.index]
    return final_df


def build_sim_dataframe(
    sim_raw_df: pd.DataFrame,
    sim_pin_order: list[str] | None = None,
) -> pd.DataFrame:
    """Return the simulation rows relevant to the measured voltage pins."""

    if sim_pin_order is None:
        sim_pin_order = SIM_PIN_ORDER
    return (
        sim_raw_df[sim_raw_df["pins"].notna()]
        .set_index("pins", drop=False)
        .reindex(sim_pin_order)
    )


def build_comparison_df(
    measurement_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    pin_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build a tidy measured-vs-simulated dataframe for plotting or residuals."""

    if pin_order is None:
        pin_order = CUMULATIVE_PIN_ORDER

    measured_plot_df = measurement_df.rename(
        columns={"computed_between": "pins"}
    )[["input_voltage_V", "pins", "cumulative_voltage_from_K_V"]].rename(
        columns={"cumulative_voltage_from_K_V": "measured_voltage_V"}
    )

    sim_plot_df = (
        sim_df.drop(columns="pins")
        .rename_axis(index="pins", columns="simulation_setting")
        .stack()
        .rename("simulated_voltage_V")
        .reset_index()
        .rename(columns={"simulation_setting": "input_voltage_V"})
    )

    comparison_df = measured_plot_df.merge(
        sim_plot_df,
        on=["input_voltage_V", "pins"],
        how="inner",
    )
    comparison_df["pins"] = pd.Categorical(
        comparison_df["pins"],
        categories=pin_order,
        ordered=True,
    )
    comparison_df = comparison_df.sort_values(["pins", "input_voltage_V"]).reset_index(drop=True)
    comparison_df["delta_V"] = comparison_df["measured_voltage_V"] - comparison_df[
        "simulated_voltage_V"
    ]
    comparison_df["delta_percent"] = (
        100 * comparison_df["delta_V"] / comparison_df["simulated_voltage_V"]
    )
    return comparison_df


def load_all_data(
    excel_path: str | Path | None = None,
    simulation_path: str | Path | None = None,
    sheet_name: str = "Sheet1",
    include_intermediate: bool = False,
) -> VoltageDividerData | VoltageDividerDataWithIntermediate:
    """Load measured and simulated dataframes using default project paths.

    By default, return only the analysis-ready dataframes. Set
    ``include_intermediate=True`` to also expose ``measurement_raw_df`` and ``sim_raw_df``.
    """

    measurement_raw_df, measurement_df = read_measurements(
        excel_path=excel_path,
        sheet_name=sheet_name,
    )
    sim_raw_df = read_sim(simulation_path)
    sim_df = build_sim_dataframe(sim_raw_df)
    comparison_df = build_comparison_df(measurement_df, sim_df)
    if include_intermediate:
        return VoltageDividerDataWithIntermediate(
            measurement_raw_df=measurement_raw_df,
            measurement_df=measurement_df,
            sim_raw_df=sim_raw_df,
            sim_df=sim_df,
            comparison_df=comparison_df,
        )

    return VoltageDividerData(
        measurement_df=measurement_df,
        sim_df=sim_df,
        comparison_df=comparison_df,
    )


__all__ = [
    "CUMULATIVE_PIN_ORDER",
    "DATA_DIR",
    "DEFAULT_EXCEL_PATH",
    "DEFAULT_SIMULATION_PATH",
    "MEASURED_PIN_ORDER",
    "SIM_PIN_ORDER",
    "SIMULATION_PIN_MAPPINGS",
    "VoltageDividerData",
    "VoltageDividerDataWithIntermediate",
    "build_comparison_df",
    "build_measurement_dataframe",
    "build_sim_dataframe",
    "convert_spice_value",
    "load_all_data",
    "read_measurement_raw",
    "read_measurements",
    "read_sim",
]
