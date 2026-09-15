from datetime import date

from index_futures_stat_arb.contracts import (
    adjust_expiry_date,
    list_contracts,
    parse_contract,
)
from index_futures_stat_arb.sessions import cme_holidays


def test_parse_contract_and_expiry():
    es = parse_contract("ESH6", pivot_year=2026)
    nq = parse_contract("NQZ26")
    assert es.expiry_date == date(2026, 3, 20)
    assert nq.expiry_date == date(2026, 12, 18)
    assert list_contracts("ES", date(2026, 1, 1), date(2026, 3, 20))[0].symbol == "ESH6"


def test_expiry_holiday_adjustment():
    holiday = {date(2026, 3, 20)}
    assert adjust_expiry_date(date(2026, 3, 20), holiday) == date(2026, 3, 19)
    assert date(2026, 4, 3) in cme_holidays(2026)
