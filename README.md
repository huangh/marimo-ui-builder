# anything_analyzer

Turn polars `LazyFrame -> LazyFrame` functions into marimo buttons. Each click applies
the function to the current data, and the accumulated chain is always shown at the
bottom of the notebook, toggleable between **polars code** and **DuckDB SQL**.

```bash
uv sync
uv run marimo edit notebooks/demo.py   # or: uv run marimo run notebooks/demo.py
uv run pytest
```

## Adding an op

```python
import polars as pl
from anything_analyzer import Param, Registry

ops = Registry("ops")   # the name used in generated code; bind it to the same variable

@ops.register(
    "Filter by value",                                   # button label
    params={                                             # optional, one widget each
        "column": Param.column(numeric_only=True),       # dropdown of current columns
        "op": Param.choice([">", "<", "="]),
        "value": Param.number(0),
    },
    sql="SELECT * FROM {input} WHERE {column} {op} {value}",  # optional, for the DuckDB view
)
def filter_compare(lf: pl.LazyFrame, column: str, op: str, value: float) -> pl.LazyFrame:
    ...
```

The decorator returns your function unchanged, so it is still callable in code:
`filter_compare(lf, ...)`, `ops.filter_compare(lf, ...)` or `lf.pipe(ops.filter_compare, ...)`.

Param kinds: `column`, `columns`, `number`, `text`, `choice`, `bool`.

### SQL templates

`sql=` is either a `str.format` template or a callable
`(input, columns, **params) -> str`. Placeholders: `{input}` (the previous step),
`{columns}` (all current columns, quoted) and one per param. Columns are inserted as
quoted identifiers, text as SQL string literals, numbers as-is, and choices raw (so
options can be operators or keywords). An op without `sql=` still works; its step shows
as a commented pass-through in the SQL view with a warning.

## Notebook wiring

See `notebooks/demo.py`. The pieces:

```python
get_steps, set_steps = mo.state([])
pipe = Pipeline(source, ops, get_steps())   # pipe.result is the current LazyFrame
panel = OpPanel(ops); buttons = panel.buttons(set_steps)   # own cell
inputs = panel.inputs(pipe)                                # own cell
panel.layout(buttons, inputs)
mode = mode_toggle()                                       # own cell
mo.vstack([mode, chain_view(pipe, mode.value)])            # last cell
```

Buttons and param inputs live in separate cells on purpose: if the buttons are
re-created after every click, marimo can replay the click onto the new buttons and
apply the op twice.

## Pipeline API

- `Pipeline.to_polars()` runnable code using `.pipe(ops.<name>, ...)`
- `Pipeline.to_sql()` DuckDB SQL, one CTE per step, reading from `source`
- `Pipeline.run_duckdb()` runs that SQL in DuckDB (used by the tests to check both agree)
- `Pipeline.explain()` polars' own optimized plan
