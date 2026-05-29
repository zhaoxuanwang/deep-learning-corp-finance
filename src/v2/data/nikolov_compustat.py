"""Compustat Global cleaning helpers for Nikolov auxiliary policy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = (
    "gvkey",
    "conm",
    "datadate",
    "fyearq",
    "fqtr",
    "fic",
    "curcdq",
    "indfmt",
    "consol",
    "datafmt",
    "gsector",
    "sic",
)

ACCOUNTING_COLUMNS = (
    "atq",
    "ppentq",
    "capxy",
    "oibdpy",
    "dlttq",
    "dlcq",
    "cheq",
    "dvty",
    "prstkcy",
)

SOURCE_COLUMNS = IDENTIFIER_COLUMNS + ACCOUNTING_COLUMNS
REQUIRED_COLUMNS = (
    "gvkey",
    "datadate",
    "fyearq",
    "fqtr",
    "fic",
    "curcdq",
    "indfmt",
    "consol",
    "datafmt",
    "gsector",
    "atq",
    "ppentq",
    "capxy",
    "oibdpy",
    "cheq",
)
OPTIONAL_ZERO_COLUMNS = ("dlttq", "dlcq", "dvty", "prstkcy")

POLICY_VARIABLES = (
    "investment_rate",
    "future_leverage",
    "payout_rate",
    "log_k",
    "profitability",
    "leverage",
)


@dataclass(frozen=True)
class NikolovCompustatConfig:
    """Configuration for the Hong Kong Compustat baseline panel."""

    raw_dir: str | Path = Path("data/raw")
    file_glob: str = "comp_global_HKG_*.csv.gz"
    raw_files: tuple[str | Path, ...] = ()
    currency: str = "HKD"
    excluded_gsectors: tuple[int, ...] = (40, 55)
    min_observations_per_firm: int = 3
    winsor_limits: tuple[float, float] = (0.01, 0.99)
    # Leverage definition. The Nikolov TO/LE/MH models have no cash holdings, so
    # their leverage b/k is GROSS debt over capital and is non-negative. Default
    # to gross (DLTT+DLC)/AT so the data support matches the model's b/k >= 0.
    # Set True for the doc's net measure (DLTT+DLC-CHE)/AT, which can be negative
    # and whose support the model cannot reach.
    leverage_net_of_cash: bool = False

    def __post_init__(self) -> None:
        low, high = self.winsor_limits
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                "winsor_limits must satisfy 0 <= low < high <= 1. "
                f"Got {self.winsor_limits}"
            )
        if self.min_observations_per_firm < 1:
            raise ValueError(
                "min_observations_per_firm must be positive. "
                f"Got {self.min_observations_per_firm}"
            )


def load_compustat_hkg_raw(
    config: NikolovCompustatConfig | None = None,
) -> pd.DataFrame:
    """Load and concatenate the raw Hong Kong Compustat extracts."""

    config = config or NikolovCompustatConfig()
    files = _resolve_raw_files(config)
    frames = [
        pd.read_csv(
            path,
            usecols=lambda column: column in SOURCE_COLUMNS,
            low_memory=False,
        )
        for path in files
    ]
    raw = pd.concat(frames, ignore_index=True)
    return _normalize_raw(raw)


def build_nikolov_policy_panel(
    config: NikolovCompustatConfig | None = None,
    raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the cleaned annual panel for auxiliary policy regressions.

    Returns:
        panel: Final firm-year panel with raw, winsorized, and FE-adjusted
            policy variables.
        diagnostics: Long-form cleaning diagnostics.
    """

    config = config or NikolovCompustatConfig()
    source = load_compustat_hkg_raw(config) if raw is None else _normalize_raw(raw)
    _validate_required_columns(source)

    diagnostics: list[dict[str, float | int | str]] = []
    _add_sample_count(diagnostics, "raw", source)

    panel = _filter(source, diagnostics, "fqtr_4", source["fqtr"].eq(4))
    panel = _filter(panel, diagnostics, "fic_hkg", panel["fic"].eq("HKG"))
    panel = _filter(panel, diagnostics, "currency", panel["curcdq"].eq(config.currency))
    panel = _filter(panel, diagnostics, "industrial", panel["indfmt"].eq("INDL"))
    panel = _filter(panel, diagnostics, "consolidated", panel["consol"].eq("C"))
    panel = _filter(panel, diagnostics, "standard_format", panel["datafmt"].eq("HIST_STD"))
    panel = _filter(
        panel,
        diagnostics,
        "exclude_financial_utility",
        ~panel["gsector"].isin(config.excluded_gsectors),
    )
    panel = _filter(panel, diagnostics, "positive_assets", panel["atq"].gt(0))
    panel = _filter(panel, diagnostics, "positive_ppent", panel["ppentq"].gt(0))
    _add_missing_rates(diagnostics, "source_missing_rate", panel, ACCOUNTING_COLUMNS)
    panel = _filter(
        panel,
        diagnostics,
        "finite_capx_oibdp_cash",
        np.isfinite(panel[["capxy", "oibdpy", "cheq"]]).all(axis=1),
    )

    panel = _construct_policy_variables(panel, config)
    _add_sample_count(diagnostics, "finite_current_variables", panel)

    panel = _add_future_leverage(panel)
    _add_sample_count(diagnostics, "with_future_leverage", panel)

    panel = _require_min_observations(panel, config.min_observations_per_firm)
    _add_sample_count(diagnostics, "min_firm_observations", panel)

    panel = _add_fixed_effect_adjusted_columns(panel, POLICY_VARIABLES)
    panel["default"] = False
    panel["firm_id"] = pd.factorize(panel["gvkey"], sort=True)[0]
    panel["t"] = panel.groupby("gvkey", sort=False).cumcount()

    _add_imputation_rates(diagnostics, panel)
    _add_panel_balance(diagnostics, panel)
    _add_variable_summary(diagnostics, panel, POLICY_VARIABLES)

    ordered_columns = _ordered_output_columns(panel)
    panel = panel.loc[:, ordered_columns].reset_index(drop=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    return panel, diagnostics_frame


def _resolve_raw_files(config: NikolovCompustatConfig) -> list[Path]:
    if config.raw_files:
        files = [Path(path) for path in config.raw_files]
    else:
        files = sorted(Path(config.raw_dir).glob(config.file_glob))
    if not files:
        raise FileNotFoundError(
            f"No Compustat files found in {config.raw_dir!s} matching "
            f"{config.file_glob!r}"
        )
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Raw Compustat files do not exist: {missing}")
    return files


def _normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in SOURCE_COLUMNS:
        if column not in frame.columns and column in OPTIONAL_ZERO_COLUMNS:
            frame[column] = np.nan

    string_columns = ("gvkey", "conm", "fic", "curcdq", "indfmt", "consol", "datafmt")
    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()

    if "datadate" in frame.columns:
        frame["datadate"] = pd.to_datetime(frame["datadate"], errors="coerce")

    numeric_columns = ("fyearq", "fqtr", "gsector", "sic") + ACCOUNTING_COLUMNS
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _validate_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Compustat data are missing required columns: {missing}")


def _filter(
    frame: pd.DataFrame,
    diagnostics: list[dict[str, float | int | str]],
    label: str,
    mask: pd.Series | np.ndarray,
) -> pd.DataFrame:
    if isinstance(mask, pd.Series):
        mask_array = mask.fillna(False).to_numpy(dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool)
    filtered = frame.loc[mask_array].copy()
    _add_sample_count(diagnostics, label, filtered)
    return filtered


def _construct_policy_variables(
    frame: pd.DataFrame,
    config: NikolovCompustatConfig,
) -> pd.DataFrame:
    panel = frame.copy()
    for column in OPTIONAL_ZERO_COLUMNS:
        panel[f"{column}_missing"] = panel[column].isna()

    debt = panel["dlttq"].fillna(0.0) + panel["dlcq"].fillna(0.0)
    payout = panel["dvty"].fillna(0.0) + panel["prstkcy"].fillna(0.0)

    net_debt = debt - panel["cheq"] if config.leverage_net_of_cash else debt
    panel["investment_rate_raw"] = panel["capxy"] / panel["atq"]
    panel["leverage_raw"] = net_debt / panel["atq"]
    panel["profitability_raw"] = panel["oibdpy"] / panel["atq"]
    panel["payout_rate_raw"] = payout / panel["atq"]
    panel["log_k_raw"] = np.log(panel["ppentq"])

    raw_columns = [
        "investment_rate_raw",
        "leverage_raw",
        "profitability_raw",
        "payout_rate_raw",
        "log_k_raw",
    ]
    finite = np.isfinite(panel.loc[:, raw_columns]).all(axis=1)
    panel = panel.loc[finite].copy()

    for final_column in (
        "investment_rate",
        "leverage",
        "profitability",
        "payout_rate",
        "log_k",
    ):
        panel[final_column] = _winsorize(
            panel[f"{final_column}_raw"],
            config.winsor_limits,
        )

    panel = panel.sort_values(["gvkey", "datadate", "fyearq"]).reset_index(drop=True)
    return panel


def _add_future_leverage(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["future_leverage_raw"] = result.groupby("gvkey")["leverage_raw"].shift(-1)
    result["future_leverage"] = result.groupby("gvkey")["leverage"].shift(-1)
    finite = np.isfinite(result["future_leverage"].to_numpy(dtype=np.float64))
    return result.loc[finite].copy()


def _require_min_observations(panel: pd.DataFrame, min_observations: int) -> pd.DataFrame:
    counts = panel.groupby("gvkey")["gvkey"].transform("size")
    return panel.loc[counts >= min_observations].copy()


def _winsorize(series: pd.Series, limits: tuple[float, float]) -> pd.Series:
    low, high = limits
    values = series.astype(float)
    lower = values.quantile(low)
    upper = values.quantile(high)
    return values.clip(lower=lower, upper=upper)


def _add_fixed_effect_adjusted_columns(
    panel: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    result = panel.copy()
    grouped = result.groupby("gvkey", sort=False)
    for column in columns:
        firm_mean = grouped[column].transform("mean")
        sample_mean = result[column].mean()
        result[f"{column}_feadj"] = result[column] - firm_mean + sample_mean
    return result


def _add_sample_count(
    diagnostics: list[dict[str, float | int | str]],
    label: str,
    frame: pd.DataFrame,
) -> None:
    diagnostics.append(
        {
            "section": "sample",
            "metric": label,
            "value": float(len(frame)),
            "rows": int(len(frame)),
            "firms": int(frame["gvkey"].nunique()) if "gvkey" in frame.columns else 0,
        }
    )


def _add_missing_rates(
    diagnostics: list[dict[str, float | int | str]],
    section: str,
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        if column in frame.columns:
            value = float(frame[column].isna().mean()) if len(frame) else np.nan
            diagnostics.append(
                {
                    "section": section,
                    "metric": column,
                    "value": value,
                    "rows": int(len(frame)),
                    "firms": int(frame["gvkey"].nunique()),
                }
            )


def _add_imputation_rates(
    diagnostics: list[dict[str, float | int | str]],
    panel: pd.DataFrame,
) -> None:
    for column in OPTIONAL_ZERO_COLUMNS:
        flag = f"{column}_missing"
        if flag in panel.columns:
            diagnostics.append(
                {
                    "section": "imputation_rate",
                    "metric": column,
                    "value": float(panel[flag].mean()) if len(panel) else np.nan,
                    "rows": int(len(panel)),
                    "firms": int(panel["gvkey"].nunique()),
                }
            )


def _add_panel_balance(
    diagnostics: list[dict[str, float | int | str]],
    panel: pd.DataFrame,
) -> None:
    counts = panel.groupby("gvkey").size()
    if counts.empty:
        summary = {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    else:
        summary = {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "min": float(counts.min()),
            "max": float(counts.max()),
        }
    for metric, value in summary.items():
        diagnostics.append(
            {
                "section": "panel_balance",
                "metric": f"periods_per_firm_{metric}",
                "value": value,
                "rows": int(len(panel)),
                "firms": int(panel["gvkey"].nunique()),
            }
        )


def _add_variable_summary(
    diagnostics: list[dict[str, float | int | str]],
    panel: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    stats = ("mean", "std", "p01", "p50", "p99")
    for column in columns:
        values = panel[column].astype(float)
        quantiles = values.quantile([0.01, 0.5, 0.99])
        summary = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p01": float(quantiles.loc[0.01]),
            "p50": float(quantiles.loc[0.5]),
            "p99": float(quantiles.loc[0.99]),
        }
        for stat in stats:
            diagnostics.append(
                {
                    "section": "variable_summary",
                    "metric": f"{column}_{stat}",
                    "value": summary[stat],
                    "rows": int(len(panel)),
                    "firms": int(panel["gvkey"].nunique()),
                }
            )


def _ordered_output_columns(panel: pd.DataFrame) -> list[str]:
    preferred = [
        "gvkey",
        "firm_id",
        "conm",
        "datadate",
        "fyearq",
        "t",
        "fic",
        "curcdq",
        "gsector",
        "sic",
        "default",
        "investment_rate",
        "future_leverage",
        "payout_rate",
        "log_k",
        "profitability",
        "leverage",
        "investment_rate_feadj",
        "future_leverage_feadj",
        "payout_rate_feadj",
        "log_k_feadj",
        "profitability_feadj",
        "leverage_feadj",
        "investment_rate_raw",
        "future_leverage_raw",
        "payout_rate_raw",
        "log_k_raw",
        "profitability_raw",
        "leverage_raw",
        "dlttq_missing",
        "dlcq_missing",
        "dvty_missing",
        "prstkcy_missing",
        "atq",
        "ppentq",
        "capxy",
        "oibdpy",
        "dlttq",
        "dlcq",
        "cheq",
        "dvty",
        "prstkcy",
    ]
    existing = [column for column in preferred if column in panel.columns]
    extras = [column for column in panel.columns if column not in existing]
    return existing + extras
