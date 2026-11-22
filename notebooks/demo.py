import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from anything_analyzer import OpPanel, Param, Pipeline, chain_view, mode_toggle, preview
    import my_ops

    return OpPanel, Param, Pipeline, chain_view, mo, mode_toggle, my_ops, pl, preview


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Anything analyzer demo

    Each button below comes from a registered polars function (see `notebooks/my_ops.py`).
    Clicking it applies that function to the current data. The chain of calls is shown at
    the bottom, as polars code or DuckDB SQL.
    """)
    return


@app.cell
def _(pl):
    # Source data. Swap this for pl.scan_parquet(...) / pl.scan_csv(...) etc.
    source = pl.LazyFrame(
        {
            "region": ["north", "south", "east", "west", "north", "south", "east", None, "west", "north"],
            "product": ["apple", "banana", "apple", "cherry", "banana", "apple", "cherry", "apple", "banana", "cherry"],
            "units": [10, 4, 7, None, 12, 3, 9, 5, 8, 15],
            "price": [1.2, 0.5, 1.1, 3.0, 0.6, 1.3, 2.8, 1.0, 0.55, 3.2],
        }
    ).with_columns(revenue=pl.col("units") * pl.col("price"))
    return (source,)


@app.cell
def _(Param, my_ops, pl):
    # Ops can also be added right in the notebook. Copying keeps my_ops.ops untouched,
    # and because `ops` is defined in this cell, the buttons update when you edit it.
    ops = my_ops.ops.copy("ops")

    @ops.register("Round numbers", params={"column": Param.column(numeric_only=True), "decimals": Param.number(0)})
    def round_column(lf: pl.LazyFrame, column: str, decimals: int) -> pl.LazyFrame:
        """No `sql=` here, so the DuckDB view flags this step."""
        return lf.with_columns(pl.col(column).round(int(decimals)))

    return (ops,)


@app.cell
def _(mo):
    get_steps, set_steps = mo.state([])
    return get_steps, set_steps


@app.cell
def _(Pipeline, get_steps, ops, source):
    pipe = Pipeline(source, ops, get_steps(), source_name="source")
    result = pipe.result  # the current LazyFrame, usable in any cell below
    return pipe, result


@app.cell
def _(OpPanel, ops, set_steps):
    # Buttons depend only on the registry, so they are never re-created by a click.
    panel = OpPanel(ops)
    buttons = panel.buttons(set_steps)
    return buttons, panel


@app.cell
def _(panel, pipe):
    # Param inputs follow the current columns (e.g. after "Keep columns").
    inputs = panel.inputs(pipe)
    return (inputs,)


@app.cell
def _(buttons, inputs, panel):
    panel.layout(buttons, inputs)
    return


@app.cell
def _(pipe, preview):
    preview(pipe)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Advanced: call the same functions in code

    They are plain functions, available as `ops.<name>` (or import them from `my_ops`).
    """)
    return


@app.cell
def _(ops, source):
    ops.filter_compare(source, column="units", op=">", value=5).pipe(ops.sum_by, by="product", value="revenue").collect()
    return


@app.cell
def _(mode_toggle):
    mode = mode_toggle()
    return (mode,)


@app.cell(hide_code=True)
def _(chain_view, mo, mode, pipe):
    mo.vstack([mode, chain_view(pipe, mode.value)])
    return


if __name__ == "__main__":
    app.run()
