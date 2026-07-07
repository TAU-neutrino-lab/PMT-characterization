from dataclasses import dataclass
from typing import Optional
import yaml


from dataclasses import dataclass, fields


@dataclass(frozen=True)
class GeneralConfig:
    baseline_window_width_ns: float
    integration_window_ns: tuple[float, float]
    peak_search_window_ns: tuple[float, float]
    global_peak_voltage_max_mV: float | None
    out_of_window_peak_voltage_max_mV: float
    allowed_peak_window_ns: tuple[float, float]


@dataclass(frozen=True)
class FitGeneralConfig:
    fit_range: tuple[float, float]
    fit_nbins: int


@dataclass(frozen=True)
class RunConfig:
    general: GeneralConfig
    fit_general: FitGeneralConfig

    def __str__(self):
        lines = []

        for section_name, section in [
            ("general", self.general),
            ("fit_general", self.fit_general),
        ]:
            lines.append(f"[{section_name}]")

            for field in fields(section):
                value = getattr(section, field.name)
                lines.append(f"{field.name} = {value}")

            lines.append("")

        return "\n".join(lines)    


def load_config(path: str, run_name: str) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    run = data[run_name]

    return RunConfig(
        general=GeneralConfig(
            baseline_window_width_ns=run["general"]["baseline_window_width_ns"],
            integration_window_ns=tuple(run["general"]["integration_window_ns"]),
            peak_search_window_ns=tuple(run["general"]["peak_search_window_ns"]),
            global_peak_voltage_max_mV=run["general"]["global_peak_voltage_max_mV"],
            out_of_window_peak_voltage_max_mV=run["general"]["out_of_window_peak_voltage_max_mV"],
            allowed_peak_window_ns=tuple(run["general"]["allowed_peak_window_ns"]),
        ),
        fit_general=FitGeneralConfig(
            fit_range=tuple(run["fit_general"]["fit_range"]),
            fit_nbins=run["fit_general"]["fit_nbins"],
        ),
    )