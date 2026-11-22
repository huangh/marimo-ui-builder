import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import my_ops  # noqa: E402

from anything_analyzer import Param, Pipeline, Registry, Step  # noqa: E402

SOURCE = pl.LazyFrame(
    {
        "region": ["north", "south", "east", None, "north", "west"],
        "units": [10, 4, None, 5, 12, 8],
        "price": [1.2, 0.5, 1.1, 1.0, 0.6, 0.55],
    }
)

CHAINS = [
    [Step("drop_nulls")],
    [Step("filter_compare", {"column": "units", "op": ">=", "value": 5})],
    [Step("filter_contains", {"column": "region", "pattern": "orth"})],
    [Step("keep_columns", {"keep": ["region", "price"]})],
    [Step("sort_by", {"column": "price", "descending": True}), Step("top_n", {"n": 3})],
    [Step("drop_nulls"), Step("sum_by", {"by": "region", "value": "units"})],
]


@pytest.mark.parametrize("steps", CHAINS)
def test_polars_and_duckdb_agree(steps):
    pipe = Pipeline(SOURCE, my_ops.ops, steps)
    expected = pipe.result.collect()
    got = pipe.run_duckdb()
    assert_frame_equal(expected, got, check_dtypes=False, check_row_order="sort_by" in str(steps) or "sum_by" in str(steps))


def test_polars_code_is_runnable():
    steps = CHAINS[4]
    pipe = Pipeline(SOURCE, my_ops.ops, steps)
    ns = {"ops": my_ops.ops, "source": SOURCE}
    exec(pipe.to_polars(), ns)
    assert_frame_equal(ns["result"].collect(), pipe.result.collect())


def test_decorator_returns_plain_function():
    assert my_ops.drop_nulls is my_ops.ops.drop_nulls
    assert my_ops.drop_nulls(SOURCE).collect().height == 4


def test_missing_sql_passes_through():
    reg = my_ops.ops.copy()

    @reg.register("Double")
    def double(lf, column: str):
        return lf.with_columns(pl.col(column) * 2)

    pipe = Pipeline(SOURCE, reg, [Step("double", {"column": "units"}), Step("drop_nulls")])
    assert pipe.missing_sql() == ["double"]
    assert "no SQL template for double" in pipe.to_sql()
    pipe.run_duckdb()  # still valid SQL
    assert "double" not in my_ops.ops  # copy did not mutate the original


def test_params_must_match_signature():
    reg = Registry()
    with pytest.raises(TypeError):
        reg.add(lambda lf: lf, name="bad", params={"nope": Param.number()})
