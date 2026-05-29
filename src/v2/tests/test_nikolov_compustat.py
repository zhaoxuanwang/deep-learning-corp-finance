from __future__ import annotations

import numpy as np
import pandas as pd

from src.v2.data.nikolov_compustat import (
    POLICY_VARIABLES,
    NikolovCompustatConfig,
    build_nikolov_policy_panel,
    load_compustat_hkg_raw,
)


def test_load_compustat_hkg_raw_reads_configured_gz_files(tmp_path):
    frame_a = pd.DataFrame([_row("001", 2000)])
    frame_b = pd.DataFrame([_row("001", 2001)])
    path_a = tmp_path / "a.csv.gz"
    path_b = tmp_path / "b.csv.gz"
    frame_a.to_csv(path_a, index=False, compression="gzip")
    frame_b.to_csv(path_b, index=False, compression="gzip")

    raw = load_compustat_hkg_raw(
        NikolovCompustatConfig(raw_files=(path_a, path_b)),
    )

    assert len(raw) == 2
    assert set(raw["fyearq"]) == {2000, 2001}


def test_compustat_filters_and_constructs_policy_variables():
    rows = [_row("001", year, dlttq=20.0 + i) for i, year in enumerate(range(2000, 2004))]
    rows.extend(
        [
            _row("002", 2000, fqtr=3),
            _row("003", 2000, curcdq="USD"),
            _row("004", 2000, gsector=40),
            _row("005", 2000, gsector=55),
            _row("006", 2000, atq=-1.0),
            _row("007", 2000, ppentq=0.0),
        ]
    )

    panel, diagnostics = build_nikolov_policy_panel(
        _test_config(),
        raw=pd.DataFrame(rows),
    )

    assert set(panel["gvkey"]) == {"001"}
    assert len(panel) == 3
    first = panel.iloc[0]
    assert first["investment_rate"] == 5.0 / 100.0
    # Default leverage is gross (DLTT+DLC)/AT, no cash subtraction.
    assert first["leverage"] == (20.0 + 2.0) / 100.0
    assert first["profitability"] == 10.0 / 100.0
    assert first["payout_rate"] == 1.0 / 100.0
    assert first["log_k"] == np.log(50.0)
    assert first["future_leverage"] == (21.0 + 2.0) / 100.0
    assert "exclude_financial_utility" in set(diagnostics["metric"])


def test_missing_debt_and_payout_are_zero_imputed_with_flags():
    rows = [
        _row("001", 2000, dlttq=np.nan, dlcq=np.nan, dvty=np.nan, prstkcy=np.nan),
        _row("001", 2001, dlttq=10.0, dlcq=1.0, dvty=2.0, prstkcy=0.0),
        _row("001", 2002, dlttq=11.0, dlcq=1.0, dvty=2.0, prstkcy=0.0),
        _row("001", 2003, dlttq=12.0, dlcq=1.0, dvty=2.0, prstkcy=0.0),
    ]

    panel, _ = build_nikolov_policy_panel(_test_config(), raw=pd.DataFrame(rows))
    first = panel.iloc[0]

    assert bool(first["dlttq_missing"])
    assert bool(first["dlcq_missing"])
    assert bool(first["dvty_missing"])
    assert bool(first["prstkcy_missing"])
    # Gross leverage: missing debt imputed to 0, so (0+0)/100 = 0 (no -cash).
    assert first["leverage"] == 0.0
    assert first["payout_rate"] == 0.0


def test_future_leverage_stays_within_firm_and_min_observations_rule_applies():
    rows = [_row("001", year, dlttq=20.0 + i) for i, year in enumerate(range(2000, 2004))]
    rows.extend(_row("002", year, dlttq=80.0 + i) for i, year in enumerate(range(2000, 2003)))

    panel, _ = build_nikolov_policy_panel(_test_config(), raw=pd.DataFrame(rows))

    assert set(panel["gvkey"]) == {"001"}
    expected = (21.0 + 2.0) / 100.0
    assert panel.loc[panel["fyearq"].eq(2000), "future_leverage"].iloc[0] == expected


def test_leverage_net_of_cash_flag_subtracts_cash():
    rows = [_row("001", year, dlttq=20.0 + i) for i, year in enumerate(range(2000, 2004))]
    config = NikolovCompustatConfig(
        raw_files=(),
        winsor_limits=(0.0, 1.0),
        min_observations_per_firm=3,
        leverage_net_of_cash=True,
    )
    panel, _ = build_nikolov_policy_panel(config, raw=pd.DataFrame(rows))
    # Net leverage subtracts cash: (DLTT+DLC-CHE)/AT.
    assert panel.iloc[0]["leverage"] == (20.0 + 2.0 - 5.0) / 100.0


def test_fixed_effect_adjusted_variables_preserve_sample_means_and_are_finite():
    rows = [_row("001", year, atq=100.0 + 10 * i) for i, year in enumerate(range(2000, 2004))]
    rows.extend(_row("002", year, atq=150.0 + 10 * i) for i, year in enumerate(range(2000, 2004)))

    panel, _ = build_nikolov_policy_panel(_test_config(), raw=pd.DataFrame(rows))

    for column in POLICY_VARIABLES:
        adjusted = f"{column}_feadj"
        assert adjusted in panel.columns
        assert np.isclose(panel[adjusted].mean(), panel[column].mean())
        assert np.isfinite(panel[[column, adjusted]].to_numpy(dtype=np.float64)).all()


def _test_config() -> NikolovCompustatConfig:
    return NikolovCompustatConfig(
        raw_files=(),
        winsor_limits=(0.0, 1.0),
        min_observations_per_firm=3,
    )


def _row(
    gvkey: str,
    year: int,
    *,
    fqtr: int = 4,
    fic: str = "HKG",
    curcdq: str = "HKD",
    indfmt: str = "INDL",
    consol: str = "C",
    datafmt: str = "HIST_STD",
    gsector: int = 20,
    atq: float = 100.0,
    ppentq: float = 50.0,
    capxy: float = 5.0,
    oibdpy: float = 10.0,
    dlttq: float = 20.0,
    dlcq: float = 2.0,
    cheq: float = 5.0,
    dvty: float = 1.0,
    prstkcy: float = 0.0,
) -> dict[str, object]:
    return {
        "gvkey": gvkey,
        "conm": f"Firm {gvkey}",
        "datadate": f"{year}-12-31",
        "fyearq": year,
        "fqtr": fqtr,
        "fic": fic,
        "curcdq": curcdq,
        "indfmt": indfmt,
        "consol": consol,
        "datafmt": datafmt,
        "gsector": gsector,
        "sic": 3000,
        "atq": atq,
        "ppentq": ppentq,
        "capxy": capxy,
        "oibdpy": oibdpy,
        "dlttq": dlttq,
        "dlcq": dlcq,
        "cheq": cheq,
        "dvty": dvty,
        "prstkcy": prstkcy,
    }
