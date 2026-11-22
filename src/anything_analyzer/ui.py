"""marimo widgets generated from a Registry.

Usage in a notebook (each line in its own cell, results bound to globals):

    panel = OpPanel(ops)                      # depends only on the registry
    buttons = panel.buttons(set_steps)        # never re-created when the chain changes
    inputs = panel.inputs(pipe)               # re-created as columns change
    panel.layout(buttons, inputs)             # display

The buttons and the param inputs are kept in separate UI elements on purpose:
marimo can replay a click onto a freshly re-created button container, which
would apply an op twice. Buttons that live in a cell that never re-runs avoid that.
"""

from __future__ import annotations

import html
from typing import Any, Callable

import marimo as mo
import polars as pl

from .pipeline import Pipeline, Step
from .registry import Op, Param, Registry

SetSteps = Callable[..., None]


def _key(op: Op, pname: str) -> str:
    return f"{op.name}__{pname}"


def _widget(param: Param, name: str, schema: pl.Schema, previous: Any = None) -> mo.ui.UIElement:
    label = param.label or name.replace("_", " ")
    if param.kind in ("column", "columns"):
        cols = [c for c, t in schema.items() if t.is_numeric()] if param.numeric_only else list(schema)
        if param.kind == "column":
            value = previous if previous in cols else (cols[0] if cols else None)
            return mo.ui.dropdown(cols, value=value, label=label, disabled=not cols)
        chosen = previous if previous is not None else param.default
        return mo.ui.multiselect(cols, value=[c for c in chosen if c in cols], label=label)
    value = param.default if previous is None else previous
    if param.kind == "number":
        return mo.ui.number(value=value, label=label)
    if param.kind == "text":
        return mo.ui.text(value=value, label=label)
    if param.kind == "choice":
        options = list(param.options or [])
        return mo.ui.dropdown(options, value=value if value in options else param.default, label=label)
    if param.kind == "bool":
        return mo.ui.checkbox(value=bool(value), label=label)
    raise ValueError(f"unknown param kind {param.kind!r}")


def _coerce(param: Param, value: Any) -> Any:
    if param.kind == "columns":
        return list(value)
    if param.kind == "number" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class OpPanel:
    """Generates one button per registered op, plus input widgets for its params."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._inputs: mo.ui.dictionary | None = None
        self._last_inputs: dict[str, Any] = {}  # kept via on_change; marimo forbids reading .value in the creating cell

    def buttons(self, set_steps: SetSteps) -> mo.ui.dictionary:
        """Op buttons plus Undo / Reset. Clicking appends a Step via `set_steps` (from `mo.state`)."""

        def make_click(op: Op) -> Callable[[Any], Any]:
            def on_click(count: Any) -> Any:
                current = self._inputs.value if self._inputs is not None else {}
                params = {k: _coerce(p, current.get(_key(op, k), p.default)) for k, p in op.params.items()}
                set_steps(lambda steps: [*steps, Step(op.name, params)])
                return (count or 0) + 1

            return on_click

        def undo(count: int) -> int:
            set_steps(lambda steps: steps[:-1])
            return count + 1

        def reset(count: int) -> int:
            set_steps([])
            return count + 1

        elements: dict[str, mo.ui.UIElement] = {
            op.name: mo.ui.button(label=op.label, kind=op.kind, value=0, on_click=make_click(op), tooltip=op.description)
            for op in self.registry
        }
        elements["_undo"] = mo.ui.button(label="Undo last", kind="warn", value=0, on_click=undo)
        elements["_reset"] = mo.ui.button(label="Reset", kind="danger", value=0, on_click=reset)
        return mo.ui.dictionary(elements)

    def inputs(self, pipeline: Pipeline) -> mo.ui.dictionary:
        """Param widgets for the current data. Keeps the user's previous choices where still valid."""
        schema = pipeline.schema()
        previous = self._last_inputs
        elements = {
            _key(op, pname): _widget(param, pname, schema, previous.get(_key(op, pname)))
            for op in self.registry
            for pname, param in op.params.items()
        }
        self._inputs = mo.ui.dictionary(elements, on_change=self._last_inputs.update)
        return self._inputs

    def layout(self, buttons: mo.ui.dictionary, inputs: mo.ui.dictionary) -> mo.Html:
        rows = []
        for op in self.registry:
            widgets = [inputs[_key(op, p)] for p in op.params if _key(op, p) in inputs]
            rows.append(
                mo.hstack(
                    [mo.Html(f'<div style="min-width:11rem">{mo.as_html(buttons[op.name]).text}</div>'), *widgets],
                    justify="start",
                    align="center",
                    gap=0.75,
                    wrap=True,
                )
            )
        rows.append(mo.hstack([buttons["_undo"], buttons["_reset"]], justify="start", gap=0.5))
        return mo.vstack(rows, gap=0.4)


def chain_view(pipeline: Pipeline, mode: str = "polars") -> mo.Html:
    """The applied chain as polars code or DuckDB SQL."""
    note = None
    if mode == "duckdb":
        code, lang = pipeline.to_sql(), "sql"
        missing = sorted(set(pipeline.missing_sql()))
        if missing:
            note = mo.callout(
                mo.md(f"No SQL template for `{', '.join(missing)}`, so those steps pass through unchanged in this SQL."),
                kind="warn",
            )
    else:
        code, lang = pipeline.to_polars(), "python"
    n = len(pipeline.steps)
    header = mo.md(f"**Applied chain** ({n} step{'s' if n != 1 else ''})")
    return mo.vstack([header, mo.md(f"```{lang}\n{code}\n```"), *([note] if note else [])])


def mode_toggle(value: str = "polars") -> mo.ui.radio:
    """Radio toggle between the polars and DuckDB SQL views; its .value is "polars" or "duckdb"."""
    return mo.ui.radio(
        {"Polars": "polars", "DuckDB SQL": "duckdb"},
        value="Polars" if value == "polars" else "DuckDB SQL",
        inline=True,
    )


def preview(pipeline: Pipeline, n: int = 200) -> mo.Html:
    """A table of the first `n` rows of the current result, with the total row count."""
    lf = pipeline.result
    total = lf.select(pl.len()).collect().item()
    df = lf.head(n).collect()
    caption = f"{total:,} rows × {df.width} columns" + (f" (showing first {n})" if total > n else "")
    return mo.vstack([mo.md(f"<small>{html.escape(caption)}</small>"), mo.ui.table(df, selection=None)])
