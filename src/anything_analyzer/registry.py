"""Registry of LazyFrame -> LazyFrame operations.

An *op* is a plain Python function ``fn(lf: pl.LazyFrame, **params) -> pl.LazyFrame``
plus a little config (button label, parameter specs, optional SQL template).
Registering a function does not wrap it: the decorated name is still the
original function, so it can be called directly from notebook code.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import polars as pl

OpFn = Callable[..., pl.LazyFrame]
# A SQL template is either a str.format template, or a callable that returns SQL.
# Template placeholders: {input} (the previous step's relation name), {columns}
# (all current columns, quoted, comma separated) and one per parameter.
SqlSpec = str | Callable[..., str]

ParamKind = Literal["column", "columns", "number", "text", "choice", "bool"]


@dataclass(frozen=True)
class Param:
    """Describes one parameter of an op and how to render its input widget."""

    kind: ParamKind
    default: Any = None
    label: str | None = None
    options: Sequence[Any] | None = None  # for kind="choice"
    numeric_only: bool = False  # for kind="column"/"columns": only offer numeric columns

    # Convenience constructors -------------------------------------------------
    @staticmethod
    def column(label: str | None = None, *, numeric_only: bool = False) -> "Param":
        return Param("column", label=label, numeric_only=numeric_only)

    @staticmethod
    def columns(label: str | None = None, *, numeric_only: bool = False) -> "Param":
        return Param("columns", default=(), label=label, numeric_only=numeric_only)

    @staticmethod
    def number(default: float = 0, label: str | None = None) -> "Param":
        return Param("number", default=default, label=label)

    @staticmethod
    def text(default: str = "", label: str | None = None) -> "Param":
        return Param("text", default=default, label=label)

    @staticmethod
    def choice(options: Sequence[Any], default: Any = None, label: str | None = None) -> "Param":
        return Param("choice", default=default if default is not None else options[0], label=label, options=tuple(options))

    @staticmethod
    def bool(default: bool = False, label: str | None = None) -> "Param":
        return Param("bool", default=default, label=label)

    # SQL rendering ------------------------------------------------------------
    def to_sql(self, value: Any) -> str:
        """Render a parameter value as a SQL fragment for templates."""
        if self.kind == "column":
            return quote_ident(value)
        if self.kind == "columns":
            return ", ".join(quote_ident(v) for v in value)
        if self.kind == "text":
            return quote_literal(value)
        if self.kind == "bool":
            return "TRUE" if value else "FALSE"
        if self.kind == "number":
            return repr(value)
        # choice: inserted raw, so options can be SQL keywords/operators (e.g. "ASC", ">").
        # Options come from the op's config, never from free text, so this is safe.
        return str(value)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class Op:
    name: str
    fn: OpFn
    label: str
    params: dict[str, Param] = field(default_factory=dict)
    sql: SqlSpec | None = None
    description: str | None = None
    kind: Literal["neutral", "success", "warn", "danger"] = "neutral"

    def __call__(self, lf: pl.LazyFrame, **params: Any) -> pl.LazyFrame:
        return self.fn(lf, **params)

    def defaults(self) -> dict[str, Any]:
        return {k: p.default for k, p in self.params.items()}

    def render_sql(self, input_name: str, columns: Sequence[str], params: Mapping[str, Any]) -> str | None:
        if self.sql is None:
            return None
        if callable(self.sql):
            return self.sql(input=input_name, columns=list(columns), **params)
        fmt = {k: self.params[k].to_sql(v) if k in self.params else v for k, v in params.items()}
        return self.sql.format(input=input_name, columns=", ".join(quote_ident(c) for c in columns), **fmt)


class Registry:
    """A named collection of ops. Also works as a namespace: ``ops.my_filter(lf)``."""

    def __init__(self, name: str = "ops") -> None:
        # `name` is how the registry is referred to in generated polars code,
        # so it should match the variable name you bind it to in the notebook.
        self.name = name
        self._ops: dict[str, Op] = {}

    # Registration -------------------------------------------------------------
    def register(
        self,
        label: str | None = None,
        *,
        params: Mapping[str, Param] | None = None,
        sql: SqlSpec | None = None,
        description: str | None = None,
        kind: Literal["neutral", "success", "warn", "danger"] = "neutral",
        name: str | None = None,
    ) -> Callable[[OpFn], OpFn]:
        """Decorator. Returns the original function unchanged."""

        def deco(fn: OpFn) -> OpFn:
            self.add(fn, label=label, params=params, sql=sql, description=description, kind=kind, name=name)
            return fn

        return deco

    def add(
        self,
        fn: OpFn,
        *,
        label: str | None = None,
        params: Mapping[str, Param] | None = None,
        sql: SqlSpec | None = None,
        description: str | None = None,
        kind: Literal["neutral", "success", "warn", "danger"] = "neutral",
        name: str | None = None,
    ) -> Op:
        op_name = name or fn.__name__
        if not op_name.isidentifier():
            raise ValueError(f"op name {op_name!r} must be a valid Python identifier")
        params = dict(params or {})
        _check_signature(fn, params)
        op = Op(
            name=op_name,
            fn=fn,
            label=label or op_name.replace("_", " ").capitalize(),
            params=params,
            sql=sql,
            description=description or inspect.getdoc(fn),
            kind=kind,
        )
        self._ops[op_name] = op
        return op

    # Lookup ---------------------------------------------------------------------
    def __getitem__(self, name: str) -> Op:
        return self._ops[name]

    def __getattr__(self, name: str) -> OpFn:
        # Namespace access: ops.filter_rows(lf, ...) calls the raw function.
        ops = self.__dict__.get("_ops", {})
        if name in ops:
            return ops[name].fn
        raise AttributeError(name)

    def __contains__(self, name: object) -> bool:
        return name in self._ops

    def __iter__(self):
        return iter(self._ops.values())

    def __len__(self) -> int:
        return len(self._ops)

    def __dir__(self):
        return [*super().__dir__(), *self._ops]

    def copy(self, name: str | None = None) -> "Registry":
        """A new registry with the same ops, to extend in a notebook cell without mutating the original."""
        new = Registry(name or self.name)
        new._ops = dict(self._ops)
        return new

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, ops={list(self._ops)})"


def _check_signature(fn: OpFn, params: Mapping[str, Param]) -> None:
    sig = inspect.signature(fn)
    positional = [
        p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        raise TypeError(f"{fn.__name__} must take the LazyFrame as its first argument")
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    for key in params:
        if key not in sig.parameters and not accepts_kwargs:
            raise TypeError(f"{fn.__name__} has no parameter {key!r} declared in params config")


# A default registry for quick use: `from anything_analyzer import ops, Param`.
ops = Registry("ops")
