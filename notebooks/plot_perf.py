# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.26.0",
#     "polars>=2.2.0",
#     "numpy>=2.5",
#     "pandas>=3.0",
#     "pyarrow>=26.0.1",
#     "anywidget>=0.11",
#     "plotly>=7.2",
#     "altair>=6.3",
#     "vegafusion>=2.0",
#     "vl-convert-python>=1.9",
#     "matplotlib>=3.11",
#     "bokeh>=3.10",
#     "datashader>=0.19",
#     "holoviews>=1.23",
#     "hvplot>=0.13",
#     "jupyter-scatter>=1.0",
#     "lonboard>=0.17",
#     "traitlets>=5.16",
# ]
#
# # marimo refuses cell outputs over 8 MB by default. Plotly's HTML output passes that at
# # roughly 750k points, so raise it here to let the bench go further.
# [tool.marimo.runtime]
# output_max_bytes = 500_000_000
# ///
"""Scatter plot performance test bench: 100k to 10M points across marimo plotting libraries.

Standalone on purpose (not wired into the anything_analyzer package). Run with:

    uvx marimo edit --sandbox notebooks/plot_perf.py
"""

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import time

    import altair as alt
    import anywidget
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import traitlets

    return alt, anywidget, mo, np, pd, pl, re, time, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scatter plot performance bench

    Pick a point count (log scale, 100k to 10M), a display mode and the libraries to try,
    then press **Render**. Each library shows where it draws (browser GPU via WebGL, browser
    CPU, or server CPU), how long Python took to build and serialize it, and how many
    bytes go to the browser.

    The meter below runs in the browser. When a plot is heavy the page freezes, and the
    freeze shows up as a long frame. Click the meter to reset its peak, then render **one
    library at a time** to see each one's browser cost, and pan or zoom to feel it.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class FrameMeter(anywidget.AnyWidget):
        """Browser-side frame timer. Long frames = main thread blocked (parsing, uploading, drawing)."""

        _esm = r"""
        function render({ el }) {
          const box = document.createElement("div");
          box.style.cssText = "font:13px ui-monospace,monospace;padding:6px 10px;border-radius:6px;" +
            "border:1px solid #8884;display:inline-block;cursor:pointer;user-select:none";
          box.title = "Click to reset the peak";
          box.textContent = "browser frame meter: starting...";
          el.appendChild(box);
          let last = performance.now(), start = last, frames = 0, worst = 0, peak = 0, alive = true;
          box.onclick = () => { peak = 0; };
          function tick(now) {
            if (!alive) return;
            const dt = now - last; last = now; frames++;
            worst = Math.max(worst, dt); peak = Math.max(peak, dt);
            if (now - start >= 1000) {
              const fps = frames * 1000 / (now - start);
              box.textContent = `browser: ${fps.toFixed(0)} fps | longest frame, last 1s: ${worst.toFixed(0)} ms | ` +
                `peak since reset: ${(peak / 1000).toFixed(2)} s`;
              frames = 0; worst = 0; start = now;
            }
            requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
          return () => { alive = false; };
        }
        export default { render };
        """
        _tick = traitlets.Int(0).tag(sync=True)

    return (FrameMeter,)


@app.cell(hide_code=True)
def _(FrameMeter, mo):
    meter = mo.ui.anywidget(FrameMeter())
    meter
    return


@app.cell(hide_code=True)
def _():
    # Renderer labels answer "does it use the GPU?". GPU here always means the viewer's
    # browser GPU via WebGL; nothing below uses a server GPU unless noted.
    LIBS = {
        "plotly": dict(
            name="Plotly Scattergl (mo.ui.plotly)",
            renderer="Browser GPU (WebGL)",
            limit=3_000_000,
        ),
        "bokeh": dict(
            name="Bokeh scatter, output_backend='webgl'",
            renderer="Browser GPU (WebGL)",
            limit=3_000_000,
        ),
        "jscatter": dict(
            name="jupyter-scatter (anywidget, regl)",
            renderer="Browser GPU (WebGL)",
            limit=None,
        ),
        "lonboard": dict(
            name="lonboard ScatterplotLayer (anywidget, deck.gl)",
            renderer="Browser GPU (WebGL); points placed over Boston as lon/lat",
            limit=None,
        ),
        "altair": dict(
            name="Altair mark_circle (mo.ui.altair_chart)",
            renderer="Browser CPU (Vega canvas)",
            limit=500_000,
        ),
        "altair_vf": dict(
            name="Altair 2D-binned heatmap + VegaFusion",
            renderer="Server CPU aggregation, then browser CPU (Vega canvas)",
            limit=None,
        ),
        "matplotlib": dict(
            name="matplotlib (mo.ui.matplotlib, box/lasso select)",
            renderer="Server CPU (Agg raster PNG)",
            limit=None,
        ),
        "datashader": dict(
            name="datashader to PNG",
            renderer="Server CPU (numba); can use a server GPU via cuDF/CuPy on CUDA",
            limit=None,
        ),
        "holoviews": dict(
            name="HoloViews rasterize (Bokeh + Panel), hvplot-compatible",
            renderer="Server CPU raster (datashader), re-rasterizes on zoom",
            limit=None,
        ),
    }
    return (LIBS,)


@app.cell(hide_code=True)
def _(LIBS, mo, np):
    def _log_steps(lo_exp, hi_exp, per_decade=10):
        exps = np.linspace(lo_exp, hi_exp, int((hi_exp - lo_exp) * per_decade) + 1)
        return sorted({int(round(10**e, -3)) for e in exps})

    n_points = mo.ui.slider(
        steps=_log_steps(5, 7), value=1_000_000, label="Points (log scale)", show_value=True, full_width=True
    )
    mode = mo.ui.radio(
        {
            "All points": "all",
            "Sample, keep every outlier": "outlier_sample",
            "Plain random sample (for comparison)": "random_sample",
        },
        value="All points",
        label="Display mode",
    )
    budget = mo.ui.slider(
        steps=_log_steps(4, 6), value=200_000, label="Sample size", show_value=True
    )
    z_cut = mo.ui.number(start=2, stop=10, step=0.5, value=4, label="Outlier rule: robust distance >")
    highlight = mo.ui.checkbox(value=True, label="Draw outliers on top in red")
    dtype = mo.ui.dropdown(["float32", "float64"], value="float32", label="Coordinate dtype")
    libs = mo.ui.multiselect(
        {v["name"]: k for k, v in LIBS.items()},
        value=[LIBS[k]["name"] for k in ("plotly", "jscatter", "datashader")],
        label="Libraries",
        full_width=True,
    )
    force = mo.ui.checkbox(value=False, label="Ignore safety limits (may freeze the tab)")
    run = mo.ui.run_button(label="Render", kind="success")

    mo.vstack(
        [
            n_points,
            mo.hstack([mode, mo.vstack([budget, z_cut, highlight, dtype])], justify="start", gap=3),
            libs,
            mo.hstack([run, force], justify="start"),
        ]
    )
    return budget, dtype, force, highlight, libs, mode, n_points, run, z_cut


@app.cell(hide_code=True)
def _(dtype, mo, np, pl, time):
    N_MAX = 10_000_000

    def make_points(n, dt, seed=0):
        """Mixture of blobs plus planted outliers. Rows are in random order, so any prefix is a fair sample."""
        rng = np.random.default_rng(seed)
        #            weight   cx    cy    sx    sy
        comps = [
            (0.55, 0.0, 0.0, 1.0, 1.0),
            (0.30, 3.0, 1.0, 0.5, 1.5),
            (0.1499, -2.5, 3.0, 0.3, 0.3),
            (0.00005, 12.0, -9.0, 0.1, 0.1),  # tiny far cluster: lost by plain sampling
        ]
        w = np.array([c[0] for c in comps] + [0.0])
        w[-1] = 1 - w.sum()  # remaining 0.00005: scattered in a ring well outside the blobs
        comp = rng.choice(len(w), size=n, p=w).astype(np.int8)
        cx, cy, sx, sy = (np.array([c[i] for c in comps] + [0.0], dtype=dt) for i in range(1, 5))
        x = cx[comp] + sx[comp] * rng.standard_normal(n, dtype=dt)
        y = cy[comp] + sy[comp] * rng.standard_normal(n, dtype=dt)
        far = comp == len(w) - 1
        ang = rng.uniform(0, 2 * np.pi, far.sum())
        rad = rng.uniform(11, 20, far.sum())
        x[far] = (0.5 + rad * np.cos(ang)).astype(dt)
        y[far] = (1.0 + rad * np.sin(ang)).astype(dt)
        return pl.DataFrame({"x": x, "y": y, "planted": comp >= 3})

    _t0 = time.perf_counter()
    full = make_points(N_MAX, getattr(np, dtype.value))
    gen_s = time.perf_counter() - _t0
    mo.md(f"Generated {N_MAX:,} {dtype.value} points in {gen_s:.2f} s (once per dtype).")
    return (full,)


@app.cell(hide_code=True)
def _(budget, full, highlight, mo, mode, n_points, pl, time, z_cut):
    def flag_outliers(lf: pl.LazyFrame, k: float) -> pl.LazyFrame:
        """Robust distance from the center: per-axis z-scores from median/MAD, combined as
        sqrt(zx^2 + zy^2). A LazyFrame -> LazyFrame op, so it could be registered with the
        anything_analyzer Registry as-is."""

        def rz(c):
            med = pl.col(c).median()
            mad = (pl.col(c) - med).abs().median() * 1.4826
            return ((pl.col(c) - med) / mad).abs()

        return lf.with_columns(outlier=(rz("x") ** 2 + rz("y") ** 2).sqrt() > k)

    _t0 = time.perf_counter()
    _lf = full.lazy().head(n_points.value)
    _need_flags = mode.value == "outlier_sample" or highlight.value
    if _need_flags:
        _lf = flag_outliers(_lf, z_cut.value)
    data = _lf.collect()

    if mode.value == "outlier_sample":
        _out = data.filter(pl.col("outlier"))
        _inl = data.filter(~pl.col("outlier"))
        _k = max(0, min(budget.value - _out.height, _inl.height))
        shown = pl.concat([_inl.sample(_k, seed=0), _out])
    elif mode.value == "random_sample":
        shown = data.sample(min(budget.value, data.height), seed=0)
    else:
        shown = data
    outliers = shown.filter(pl.col("outlier")) if highlight.value else shown.clear()
    prep_s = time.perf_counter() - _t0

    xs, ys = shown["x"].to_numpy(), shown["y"].to_numpy()
    xo, yo = outliers["x"].to_numpy(), outliers["y"].to_numpy()
    _planted_total = int(data["planted"].sum())
    _planted_shown = int(shown["planted"].sum())
    _flagged = int(data["outlier"].sum()) if _need_flags else None
    mo.callout(
        mo.md(
            f"**{shown.height:,} of {data.height:,} points shown** ({mode.value}), prep {prep_s:.2f} s. "
            + (f"Outliers flagged: {_flagged:,}. " if _flagged is not None else "")
            + f"Planted outliers visible: **{_planted_shown:,} / {_planted_total:,}**."
        ),
        kind="success" if _planted_shown == _planted_total else "warn",
    )
    return outliers, shown, xo, xs, yo, ys


@app.cell(hide_code=True)
def _(LIBS, force, libs, mo, re, run, shown, time):
    def gate(key):
        """None if this library should render now, else what to show in its place."""
        lib = LIBS[key]
        if key not in libs.value:
            return mo.md("")
        if not run.value:
            return mo.md(f"*{lib['name']}: press Render.*")
        lim = lib["limit"]
        if lim is not None and shown.height > lim and not force.value:
            return mo.callout(
                mo.md(
                    f"**{lib['name']}** skipped: {shown.height:,} points is above its safety limit of "
                    f"{lim:,}. Tick *Ignore safety limits* to try anyway, or switch to a sample mode."
                ),
                kind="warn",
            )
        return None

    def html_payload(text):
        """Bytes the browser fetches for an HTML output: the HTML itself plus any marimo
        virtual files it points at (their URLs start with @file/<size>-)."""
        urls = set(re.findall(r"@file/(\d+-[\w.-]+)", text))
        return len(text) + sum(int(u.split("-", 1)[0]) for u in urls)

    class Timer:
        def __init__(self):
            self.t0 = time.perf_counter()

        def s(self):
            return time.perf_counter() - self.t0

    def card(key, view, build_s, payload_bytes, note=""):
        lib = LIBS[key]
        row = dict(
            library=lib["name"],
            renderer=lib["renderer"],
            points=shown.height,
            python_build_s=round(build_s, 3),
            payload_mb=round(payload_bytes / 1e6, 2),
        )
        head = mo.md(
            f"### {lib['name']}\n"
            f"**Renderer:** {lib['renderer']} · **points:** {shown.height:,} · "
            f"**Python build + serialize:** {build_s:.2f} s · **to browser:** {payload_bytes / 1e6:,.2f} MB"
            + (f"  \n{note}" if note else "")
        )
        return mo.vstack([head, view]), row

    return Timer, card, gate, html_payload


@app.cell
def _(Timer, card, gate, html_payload, mo, xo, xs, yo, ys):
    plotly_row = None
    _view = gate("plotly")
    if _view is None:
        import plotly.graph_objects as _go

        _t = Timer()
        _fig = _go.Figure(
            [
                _go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=2, opacity=0.35), name="points"),
                _go.Scattergl(x=xo, y=yo, mode="markers", marker=dict(size=4, color="red"), name="outliers"),
            ]
        )
        _fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        _el = mo.ui.plotly(_fig)
        _view, plotly_row = card(
            "plotly", _el, _t.s(), html_payload(_el.text), "Arrays go as base64 typed arrays inside the cell output, which counts against marimo's output_max_bytes."
        )
    _view
    return (plotly_row,)


@app.cell
def _(Timer, card, gate, html_payload, mo, xo, xs, yo, ys):
    bokeh_row = None
    _view = gate("bokeh")
    if _view is None:
        from bokeh.plotting import figure as _figure

        _t = Timer()
        _p = _figure(output_backend="webgl", height=450, sizing_mode="stretch_width")
        _p.scatter(xs, ys, size=2, alpha=0.35, line_color=None)
        _p.scatter(xo, yo, size=4, color="red")
        _html = mo.as_html(_p)  # serialize once here so the timing is real, and reuse it as the output
        _view, bokeh_row = card("bokeh", _html, _t.s(), html_payload(_html.text), "Arrays go as base64 inside the HTML.")
    _view
    return (bokeh_row,)


@app.cell
def _(Timer, card, gate, mo, np, pd, xo, xs, yo, ys):
    jscatter_row = None
    _view = gate("jscatter")
    if _view is None:
        import jscatter as _jscatter

        _t = Timer()
        _pdf = pd.DataFrame(
            {
                "x": np.concatenate([xs, xo]),
                "y": np.concatenate([ys, yo]),
                "kind": pd.Categorical.from_codes(
                    np.r_[np.zeros(len(xs), np.int8), np.ones(len(xo), np.int8)], ["point", "outlier"]
                ),
            }
        )
        _sc = _jscatter.Scatter(
            data=_pdf, x="x", y="y", color_by="kind", color_map={"point": "#3b82f6", "outlier": "#ff0000"},
            size=2, opacity=0.5, height=450,
        )
        _el = mo.ui.anywidget(_sc.widget)
        _view, jscatter_row = card(
            "jscatter", _el, _t.s(), _sc.widget.points.nbytes,
            "Binary widget transfer, not HTML. Lasso selection is built in (shift+drag).",
        )
    _view
    return (jscatter_row,)


@app.cell
def _(Timer, card, gate, mo, np, xo, xs, yo, ys):
    lonboard_row = None
    _view = gate("lonboard")
    if _view is None:
        import pyarrow as _pa
        from lonboard import Map as _Map
        from lonboard import ScatterplotLayer as _Layer

        def _points_table(x, y):
            # Fake lon/lat around Boston so the bench runs on any x/y data.
            lon = -71.06 + x.astype(np.float64) * 0.02
            lat = 42.36 + y.astype(np.float64) * 0.015
            arr = _pa.FixedSizeListArray.from_arrays(_pa.array(np.column_stack([lon, lat]).ravel()), 2)
            meta = {"ARROW:extension:name": "geoarrow.point", "ARROW:extension:metadata": '{"crs":"OGC:CRS84"}'}
            return _pa.Table.from_arrays([arr], schema=_pa.schema([_pa.field("geometry", arr.type, metadata=meta)]))

        _t = Timer()
        _base = _points_table(xs, ys)
        _layers = [_Layer(_base, get_fill_color=[59, 130, 246, 90], radius_min_pixels=1, radius_max_pixels=2)]
        if len(xo):
            _layers.append(_Layer(_points_table(xo, yo), get_fill_color=[255, 0, 0, 255], radius_min_pixels=3))
        _el = mo.ui.anywidget(_Map(_layers, height=450))
        _view, lonboard_row = card(
            "lonboard", _el, _t.s(), _base.nbytes,
            "Binary widget transfer (Arrow, sent as Parquet chunks). Built for geo data such as MBTA vehicle positions.",
        )
    _view
    return (lonboard_row,)


@app.cell
def _(Timer, alt, card, gate, html_payload, mo, outliers, shown):
    altair_row = None
    _view = gate("altair")
    if _view is None:
        _t = Timer()
        _pts = alt.Chart(shown.select("x", "y")).mark_circle(size=3, opacity=0.35).encode(x="x:Q", y="y:Q")
        _out = alt.Chart(outliers.select("x", "y")).mark_circle(size=12, color="red").encode(x="x:Q", y="y:Q")
        _el = mo.ui.altair_chart(
            (_pts + _out).properties(width="container", height=420), chart_selection=False, legend_selection=False
        )
        _view, altair_row = card(
            "altair", _el, _t.s(), html_payload(_el.text),
            "Every point is a Vega mark drawn on a canvas by the CPU.",
        )
    _view
    return (altair_row,)


@app.cell
def _(Timer, alt, card, gate, html_payload, mo, outliers, shown):
    altair_vf_row = None
    _view = gate("altair_vf")
    if _view is None:
        _t = Timer()
        with alt.data_transformers.enable("vegafusion"):
            _heat = (
                alt.Chart(shown.select("x", "y"))
                .mark_rect()
                .encode(
                    alt.X("x:Q", bin=alt.Bin(maxbins=160)),
                    alt.Y("y:Q", bin=alt.Bin(maxbins=90)),
                    alt.Color("count():Q", scale=alt.Scale(type="log"), legend=None),
                )
            )
            _out = alt.Chart(outliers.select("x", "y")).mark_circle(size=12, color="red").encode(x="x:Q", y="y:Q")
            _el = mo.ui.altair_chart(
                (_heat + _out).properties(width="container", height=420), chart_selection=False, legend_selection=False
            )
        _view, altair_vf_row = card(
            "altair_vf", _el, _t.s(), html_payload(_el.text),
            "VegaFusion runs the binning in Python, so only bins (plus outlier points) reach the browser. "
            "marimo does not support chart selections with VegaFusion yet.",
        )
    _view
    return (altair_vf_row,)


@app.cell
def _(Timer, card, gate, html_payload, mo, xo, xs, yo, ys):
    matplotlib_row = None
    mpl_el = None
    _view = gate("matplotlib")
    if _view is None:
        import matplotlib.pyplot as _plt

        _t = Timer()
        _fig, _ax = _plt.subplots(figsize=(9, 4.5), dpi=100)
        _ax.plot(xs, ys, ",", alpha=0.3)  # pixel marker: far faster than ax.scatter at millions of points
        _ax.plot(xo, yo, "o", color="red", markersize=2)
        _fig.tight_layout()
        mpl_el = mo.ui.matplotlib(_ax)
        _view, matplotlib_row = card(
            "matplotlib", mpl_el, _t.s(), html_payload(mpl_el.text),
            "Static image, but box select (drag) and lasso (shift+drag) map back to rows in Python.",
        )
        _plt.close(_fig)
    _view
    return matplotlib_row, mpl_el


@app.cell(hide_code=True)
def _(mo, mpl_el, xs, ys):
    mo.stop(mpl_el is None)
    _sel = int(mpl_el.value.get_mask(xs, ys).sum()) if mpl_el.value else 0
    mo.md(f"matplotlib selection: **{_sel:,}** points selected (drag on the image above).")
    return


@app.cell
def _(Timer, card, gate, mo, pd, shown, xo, yo):
    datashader_row = None
    _view = gate("datashader")
    if _view is None:
        import io as _io

        import datashader as _ds
        import datashader.transfer_functions as _tf

        _t = Timer()
        # datashader takes pandas/dask/cuDF, not polars
        _pdf = pd.DataFrame({"x": shown["x"].to_numpy(), "y": shown["y"].to_numpy()})
        _cvs = _ds.Canvas(plot_width=900, plot_height=450)
        _img = _tf.shade(_cvs.points(_pdf, "x", "y"), how="eq_hist", cmap=["#c6dbef", "#08306b"])
        if len(xo):
            _o = _cvs.points(pd.DataFrame({"x": xo, "y": yo}), "x", "y")
            _img = _tf.stack(_img, _tf.spread(_tf.shade(_o, cmap=["red", "red"]), px=2))
        _buf = _io.BytesIO()
        _tf.set_background(_img, "white").to_pil().save(_buf, format="PNG")
        _png = _buf.getvalue()
        _view, datashader_row = card(
            "datashader", mo.image(_png), _t.s(), len(_png),
            "Fixed-size PNG whatever the point count. The first run in a session includes about 1 s of numba JIT.",
        )
    _view
    return (datashader_row,)


@app.cell
def _(Timer, card, gate, html_payload, mo, shown, xo, yo):
    holoviews_row = None
    _view = gate("holoviews")
    if _view is None:
        import holoviews as _hv
        from holoviews.operation.datashader import rasterize as _rasterize

        _hv.extension("bokeh")
        _t = Timer()
        _ras = _rasterize(_hv.Points((shown["x"].to_numpy(), shown["y"].to_numpy()))).opts(
            cmap="Blues", cnorm="eq_hist", responsive=True, height=450
        )
        _plot = _ras * _hv.Points((xo, yo)).opts(color="red", size=3)
        _html = mo.as_html(_plot)  # goes through marimo's Panel bridge, so it stays live
        _view, holoviews_row = card(
            "holoviews", _html, _t.s(), html_payload(_html.text),
            "hvplot gives the same thing in one line: `df.hvplot.points('x', 'y', rasterize=True)`. "
            "Zooming sends the new range to Python, which re-rasterizes at full detail.",
        )
    _view
    return (holoviews_row,)


@app.cell(hide_code=True)
def _(
    altair_row,
    altair_vf_row,
    bokeh_row,
    datashader_row,
    holoviews_row,
    jscatter_row,
    lonboard_row,
    matplotlib_row,
    mo,
    pl,
    plotly_row,
):
    _rows = [
        r
        for r in (
            plotly_row, bokeh_row, jscatter_row, lonboard_row, altair_row,
            altair_vf_row, matplotlib_row, datashader_row, holoviews_row,
        )
        if r
    ]
    mo.vstack(
        [
            mo.md("## Python-side timings for this render\nBrowser time is not in this table. Read it off the frame meter at the top."),
            mo.ui.table(pl.DataFrame(_rows), selection=None, pagination=False) if _rows else mo.md("*Nothing rendered yet.*"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
