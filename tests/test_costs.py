import pytest

from index_futures_stat_arb.contracts import PRODUCTS
from index_futures_stat_arb.execution.costs import CostModel


@pytest.mark.parametrize(
    ("side", "expected"),
    [(1, 5000.5), (-1, 4999.5)],
)
def test_fill_rounds_away_from_trader(side: int, expected: float) -> None:
    model = CostModel(slippage_ticks=1.0, half_spread_ticks=0.5)
    assert model.fill_price(PRODUCTS["ES"], 5000.10, side) == expected


def test_buy_fill_is_not_below_reference() -> None:
    assert CostModel().fill_price(PRODUCTS["NQ"], 18000.0, 1) > 18000.0


def test_sell_fill_is_not_above_reference() -> None:
    assert CostModel().fill_price(PRODUCTS["NQ"], 18000.0, -1) < 18000.0


@pytest.mark.parametrize("quantity", [-3, -1, 0, 1, 3])
def test_fees_use_absolute_quantity(quantity: int) -> None:
    model = CostModel(commission_per_contract_usd=2.0, exchange_fees_per_contract_usd=3.0)
    assert model.fees_usd(PRODUCTS["ES"], quantity) == abs(quantity) * 5.0


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        CostModel().fill_price(PRODUCTS["ES"], 5000.0, 0)  # type: ignore[arg-type]
