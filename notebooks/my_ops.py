"""Example ops. Each one is a plain polars function plus a little config.

    @ops.register("Button label", params={...}, sql="... {input} ...")
    def my_op(lf: pl.LazyFrame, **params) -> pl.LazyFrame: ...

`sql` is optional. It is a str.format template (or a callable returning SQL)
used only for the DuckDB view of the chain. Placeholders: {input} is the
previous step, {columns} is every current column, and one per param (columns
come quoted, text comes as a SQL string literal, choices are inserted as-is).
"""

import polars as pl

from anything_analyzer import Param, Registry, quote_ident

ops = Registry("ops")

_COMPARE = {
    ">": pl.Expr.gt,
    ">=": pl.Expr.ge,
    "<": pl.Expr.lt,
    "<=": pl.Expr.le,
    "=": pl.Expr.eq,
    "!=": pl.Expr.ne,
}


@ops.register("Drop rows with nulls", sql="SELECT * FROM {input} WHERE COLUMNS(*) IS NOT NULL")
def drop_nulls(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove every row that has a null in any column."""
    return lf.drop_nulls()


@ops.register(
    "Filter by value",
    params={
        "column": Param.column(numeric_only=True),
        "op": Param.choice(list(_COMPARE), default=">"),
        "value": Param.number(0),
    },
    sql="SELECT * FROM {input} WHERE {column} {op} {value}",
)
def filter_compare(lf: pl.LazyFrame, column: str, op: str, value: float) -> pl.LazyFrame:
    """Keep rows where `column <op> value`."""
    return lf.filter(_COMPARE[op](pl.col(column), value))


@ops.register(
    "Filter text contains",
    params={"column": Param.column(), "pattern": Param.text("")},
    sql="SELECT * FROM {input} WHERE contains(CAST({column} AS VARCHAR), {pattern})",
)
def filter_contains(lf: pl.LazyFrame, column: str, pattern: str) -> pl.LazyFrame:
    """Keep rows whose `column` contains `pattern` (literal, case sensitive)."""
    return lf.filter(pl.col(column).cast(pl.String).str.contains(pattern, literal=True))


@ops.register(
    "Keep columns",
    params={"keep": Param.columns("columns")},
    sql="SELECT {keep} FROM {input}",
)
def keep_columns(lf: pl.LazyFrame, keep: list[str]) -> pl.LazyFrame:
    """Select only the chosen columns."""
    return lf.select(keep)


def _sort_sql(input: str, columns: list[str], column: str, descending: bool) -> str:
    return f"SELECT * FROM {input} ORDER BY {quote_ident(column)} {'DESC' if descending else 'ASC'} NULLS LAST"


@ops.register("Sort", params={"column": Param.column(), "descending": Param.bool(True)}, sql=_sort_sql)
def sort_by(lf: pl.LazyFrame, column: str, descending: bool) -> pl.LazyFrame:
    """Sort by one column."""
    return lf.sort(column, descending=descending, nulls_last=True)


@ops.register("Top N rows", params={"n": Param.number(10)}, sql="SELECT * FROM {input} LIMIT {n}")
def top_n(lf: pl.LazyFrame, n: int) -> pl.LazyFrame:
    """Keep the first `n` rows."""
    return lf.head(int(n))


@ops.register(
    "Sum by group",
    params={"by": Param.column("group by"), "value": Param.column("sum of", numeric_only=True)},
    sql="SELECT {by}, SUM({value}) AS {value} FROM {input} GROUP BY {by} ORDER BY {by}",
    kind="success",
)
def sum_by(lf: pl.LazyFrame, by: str, value: str) -> pl.LazyFrame:
    """Group by `by` and sum `value`."""
    return lf.group_by(by).agg(pl.col(value).sum()).sort(by)
