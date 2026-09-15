import pandas as pd
import pytest

from index_futures_stat_arb.contracts import PRODUCTS
from index_futures_stat_arb.execution.sizing import (
    DollarNeutralSizer,
    FixedContracts,
    SizerSpec,
    SizerState,
    VolTargetSizer,
    build_sizer,
)


@pytest.mark.parametrize(
    ("signal", "expected"),
    [(1, (2, -1)), (-1, (-2, 1)), (0, (0, 0))],
)
def test_dollar_neutral_signs(signal: int, expected: tuple[int, int]) -> None:
    sizer = DollarNeutralSizer(es_contracts=2)
    assert sizer.size(signal, 5000.0, 18000.0, 0.75, SizerState()) == expected


def test_dollar_neutral_cap() -> None:
    sizer = DollarNeutralSizer(es_contracts=10, max_contracts=2)
    assert sizer.size(1, 5000.0, 18000.0, 10.0, SizerState()) == (10, -2)


def test_dollar_neutral_unit_is_fractional() -> None:
    sizer = DollarNeutralSizer(es_contracts=1)
    assert sizer.unit(1, 2100.0, 4450.0, 0.75) == pytest.approx((1.0, -0.8848314607))


def test_fixed_contracts() -> None:
    sizer = FixedContracts(n_es=3, n_nq=4)
    assert sizer.size(1, 1.0, 1.0, 1.0, SizerState()) == (3, -4)
    assert sizer.size(-1, 1.0, 1.0, 1.0, SizerState()) == (-3, 4)


def test_vol_target_passes_through_before_lookback() -> None:
    sizer = VolTargetSizer(
        target_daily_vol_usd=1000.0,
        lookback_sessions=3,
        inner=FixedContracts(n_es=2, n_nq=2),
    )
    state = SizerState(pd.Series([10.0, 20.0]))
    assert sizer.size(1, 5000.0, 18000.0, 1.0, state) == (2, -2)


def test_vol_target_scales_to_twice_contracts() -> None:
    sizer = VolTargetSizer(
        target_daily_vol_usd=2.0,
        lookback_sessions=2,
        inner=FixedContracts(),
    )
    state = SizerState(pd.Series([0.0, 1.4142135623730951]))
    assert sizer.size(1, 5000.0, 18000.0, 1.0, state) == (2, -2)


def test_vol_target_respects_cap() -> None:
    sizer = VolTargetSizer(
        target_daily_vol_usd=100.0,
        lookback_sessions=2,
        max_contracts=3,
        inner=FixedContracts(n_es=2, n_nq=2),
    )
    state = SizerState(pd.Series([1.0, 2.0]))
    assert sizer.size(1, 5000.0, 18000.0, 1.0, state) == (3, -3)


def test_vol_target_scales_fractional_dollar_neutral_unit() -> None:
    sizer = VolTargetSizer(
        target_daily_vol_usd=18.384776310850235,
        lookback_sessions=2,
        max_contracts=50,
        inner=DollarNeutralSizer(es_contracts=1, max_contracts=50),
    )
    state = SizerState(pd.Series([0.0, 1.0]))
    assert sizer.size(1, 2100.0, 4450.0, 0.75, state) == (26, -23)


def test_vol_target_respects_leg_notional_cap() -> None:
    sizer = VolTargetSizer(
        target_daily_vol_usd=100000.0,
        lookback_sessions=2,
        max_units_a=50,
        max_leg_notional_usd=2_000_000.0,
        spec_a=PRODUCTS["ES"],
        spec_b=PRODUCTS["NQ"],
        inner=DollarNeutralSizer(
            es_contracts=1,
            spec_a=PRODUCTS["ES"],
            spec_b=PRODUCTS["NQ"],
        ),
    )
    result = sizer.size(1, 5000.0, 18000.0, 1.0, SizerState(pd.Series([0.0, 1.0])))
    assert abs(result[0]) <= 8
    assert abs(result[0] * 5000.0 * 50 - abs(result[1]) * 18000.0 * 20) <= 18000.0 * 20


@pytest.mark.parametrize("name", ["fixed", "dollar_neutral"])
def test_build_sizer(name: str) -> None:
    assert build_sizer(SizerSpec(name=name)).size(1, 5000.0, 18000.0, 0.75, SizerState())[0] > 0


def test_build_sizer_nested_inner() -> None:
    sizer = build_sizer(
        SizerSpec(
            name="vol_target",
            kwargs={
                "target_daily_vol_usd": 5000.0,
                "inner": {"name": "dollar_neutral", "es_contracts": 1, "max_contracts": 50},
            },
        )
    )
    assert isinstance(sizer, VolTargetSizer)
    assert sizer.inner.size(1, 5000.0, 18000.0, 0.75, SizerState()) == (1, -1)


def test_unknown_sizer_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_sizer(SizerSpec(name="unknown"))
