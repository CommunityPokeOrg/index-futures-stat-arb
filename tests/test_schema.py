from index_futures_stat_arb.schema import (
    FileEntry,
    Manifest,
    make_dataset_id,
    product_from_contract,
)


def test_manifest_json_round_trip(tmp_path):
    manifest = Manifest(
        dataset_id="fixture-id",
        source="synthetic",
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        stype_in="raw_symbol",
        symbols=["ESH6"],
        start="2026-01-01",
        end="2026-01-02",
        retrieved_at="2026-01-01T00:00:00+00:00",
        client_version="synthetic-fixture",
        price_scale=1e-9,
        timezone_in="UTC",
        files=[FileEntry("part.parquet", 2, "abc")],
        row_count=2,
        notes="offline fixture",
    )
    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    assert Manifest.from_json(path) == manifest


def test_dataset_id_is_order_independent():
    a = make_dataset_id("synthetic", "D", "S", ["NQH6", "ESH6"], "2026-01-01", "2026-02-01")
    b = make_dataset_id("synthetic", "D", "S", ["ESH6", "NQH6"], "2026-01-01", "2026-02-01")
    assert a == b


def test_product_from_contract():
    assert product_from_contract("ESH6") == "ES"
    assert product_from_contract("NQZ26") == "NQ"
    for bad in ("ESH", "ESX6", "ES6"):
        try:
            product_from_contract(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
