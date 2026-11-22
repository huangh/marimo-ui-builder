"""A chain of applied ops, and its polars-code / DuckDB-SQL representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import polars as pl

from .registry import Registry


@dataclass(frozen=True)
class Step:
    op: str
    params: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        source: pl.LazyFrame | pl.DataFrame,
        registry: Registry,
        steps: Sequence[Step] = (),
        source_name: str = "source",
    ) -> None:
        self.source = source.lazy()
        self.registry = registry
        self.steps = list(steps)
        self.source_name = source_name

    # Execution ----------------------------------------------------------------
    def apply(self, upto: int | None = None) -> pl.LazyFrame:
        lf = self.source
        for step in self.steps[:upto]:
            lf = self.registry[step.op](lf, **step.params)
        return lf

    @property
    def result(self) -> pl.LazyFrame:
        return self.apply()

    def columns(self, upto: int | None = None) -> list[str]:
        return self.apply(upto).collect_schema().names()

    def schema(self, upto: int | None = None) -> pl.Schema:
        return self.apply(upto).collect_schema()

    # Polars representation ----------------------------------------------------
    def to_polars(self, result_name: str = "result") -> str:
        """Python code that reproduces the chain using the registered functions."""
        ns = self.registry.name
        if not self.steps:
            return f"{result_name} = {self.source_name}"
        lines = [f"{result_name} = (", f"    {self.source_name}"]
        for step in self.steps:
            args = "".join(f", {k}={v!r}" for k, v in step.params.items())
            lines.append(f"    .pipe({ns}.{step.op}{args})")
        lines.append(")")
        return "\n".join(lines)

    def explain(self) -> str:
        """Polars' own optimized query plan for the chain."""
        return self.apply().explain()

    # DuckDB representation ----------------------------------------------------
    def to_sql(self) -> str:
        """DuckDB SQL: one CTE per step, reading from a relation named `source_name`.

        Ops without an SQL template render as a commented passthrough so the
        output stays valid SQL; check `missing_sql()` to detect that.
        """
        if not self.steps:
            return f"SELECT * FROM {self.source_name}"
        ctes = []
        prev = self.source_name
        for i, step in enumerate(self.steps):
            op = self.registry[step.op]
            name = f"step_{i + 1}_{step.op}"
            body = op.render_sql(prev, self.columns(i), step.params)
            if body is None:
                body = f"-- no SQL template for {step.op}({_fmt_params(step.params)}), passing through\n    SELECT * FROM {prev}"
            else:
                body = body.strip().replace("\n", "\n    ")
            ctes.append(f"{name} AS (\n    {body}\n)")
            prev = name
        return "WITH\n" + ",\n".join(ctes) + f"\nSELECT * FROM {prev}"

    def missing_sql(self) -> list[str]:
        return [s.op for s in self.steps if self.registry[s.op].sql is None]

    def run_duckdb(self) -> pl.DataFrame:
        """Run the SQL representation in DuckDB against the collected source."""
        import duckdb

        con = duckdb.connect()
        con.register(self.source_name, self.source.collect().to_arrow())
        return con.sql(self.to_sql()).pl()

    def __repr__(self) -> str:
        return self.to_polars()


def _fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items())
