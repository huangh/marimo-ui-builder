# Scatter plot performance: findings

Bench notebook: `notebooks/plot_perf.py` (standalone, not part of the package).

```bash
uvx marimo edit --sandbox notebooks/plot_perf.py
```

## Short version

- **Sending every point to the browser stops working between 1M and 10M.** At 10M the payload is
  100 to 320 MB and the tab freezes for 10 to 35 s before anything shows, whatever the library.
- **Up to about 1M points, WebGL scatter is fine.** jupyter-scatter and Plotly Scattergl both draw
  1M points with a freeze of 1 to 3 s and pan/zoom smoothly afterwards.
- **For 10M, draw on the server.** datashader, HoloViews `rasterize` and matplotlib all send a
  ~50 to 100 KB image, stay under 2 s of Python once warm, and never freeze the browser. HoloViews
  re-rasterizes at full detail when you zoom. Altair with VegaFusion bins in Python and sends
  only the bins.
- **Outlier-preserving sampling makes any library usable at 10M.** Showing a 200k random sample
  plus every point flagged as an outlier took 0.2 s to prepare, sent 2 MB to Plotly and froze the
  tab for 1 s, and kept all 1,009 planted outliers. A plain 200k random sample kept only 29 of them.

## Which ones use the GPU

Only the browser's GPU, through WebGL: Plotly `Scattergl`, Bokeh with `output_backend="webgl"`,
jupyter-scatter (regl) and lonboard (deck.gl). Altair/Vega draws on a canvas with the CPU.
matplotlib, datashader and HoloViews `rasterize` draw on the server CPU. datashader is the only
one that can use a server GPU, via cuDF/CuPy on an NVIDIA/CUDA machine.

The GPU only speeds up drawing. At 10M the time goes to serializing, transferring and parsing
100+ MB, which the GPU doesn't help with.

## Numbers

Measured in a cloud container: Python 3.13, marimo 0.26, float32 coordinates, and headless
Chromium with **software WebGL (SwiftShader, no real GPU)**. Browser numbers for the WebGL
libraries are therefore pessimistic, so run the notebook on a real machine to see yours.
"Freeze" is the longest single frame the in-page meter saw after pressing Render.

Each cell: Python build + serialize (s) / sent to browser (MB) / browser freeze (s).

| Library | Renderer | 100k | 1M | 10M |
|---|---|---|---|---|
| Plotly Scattergl (`mo.ui.plotly`) | Browser GPU | 0.07 / 1.1 / 0.8 | 0.30 / 10.7 / 3.2 | 3.2 / 107 / 34 (64 s until visible) |
| jupyter-scatter (anywidget) | Browser GPU | 0.07 / 3.2 / 0.3 | 0.47 / 32 / 1.1 | 5.9 / 320 / 10 (28 s until visible) |
| Bokeh, webgl | Browser GPU | 0.07 / 1.0 / n/a | 0.49 / 9.9 / n/a | 4.2 / 99 / n/a |
| lonboard (anywidget) | Browser GPU | 0.15 / 1.6 / n/a | 0.42 / 16 / n/a | 3.7 / 160 / n/a |
| Altair `mark_circle` | Browser CPU | 0.06 / 0.8 / 1.1 | 0.06 / 8.0 / 9.9 (15 s until settled) | not run |
| Altair heatmap + VegaFusion | Server CPU bins, then browser | 0.60 / 0.1 / 0.07 | 0.66 / 0.1 / 0.03 | 1.35 / 0.2 / 0.05 |
| matplotlib (`mo.ui.matplotlib`) | Server CPU | 0.11 / <0.1 / 0.05 | 0.23 / <0.1 / 0.03 | 1.8 / 0.05 / 0.02 |
| datashader to PNG | Server CPU | 3.1* / <0.1 / 0.03 | 3.2* / <0.1 / 0.02 | 3.4* / 0.05 / 0.03 |
| HoloViews `rasterize` | Server CPU | 1.0 / 0.1 / n/a | 1.0 / 0.1 / n/a | 1.3 / 0.1 / n/a |

\* Each run used a fresh server, so these include about 3 s of numba JIT compile. Warm, datashader
aggregates and shades 10M points in about 0.15 s.

n/a: Bokeh, HoloViews/Panel and lonboard load JavaScript or WASM from CDNs (cdn.bokeh.org,
cdn.holoviz.org, jsdelivr) that the test container can't reach, so their browser side wasn't measured.

10M rows with "Sample, keep every outlier" (200k shown):

| Library | Python (s) | MB | Freeze (s) |
|---|---|---|---|
| Plotly Scattergl | 0.07 + 0.2 prep | 2.2 | 1.0 |
| jupyter-scatter | 0.11 + 0.2 prep | 6.4 | 0.4 |
| Altair `mark_circle` | 0.05 + 0.2 prep | 1.6 | 2.0 |

## Gotchas found

- **marimo's 8 MB output cap.** `mo.ui.plotly` embeds the data in the cell output, so above about
  750k points marimo replaces the plot with "output too large". The notebook raises
  `output_max_bytes` in its script header. Altair, Bokeh and images go through marimo's virtual
  files and anywidgets through binary messages, so those don't hit the cap.
- **Altair's raw scatter is CPU-only and slow.** 1M points froze the tab for about 10 s. VegaFusion
  doesn't help a raw scatter because there's nothing to aggregate, but it works well for binned
  views. marimo doesn't support chart selections with VegaFusion yet (marimo issue 4601).
- **jupyter-scatter sends 32 bytes per point** (it keeps several float64 columns), so its payload is
  3x Plotly's. It still renders fastest among the WebGL options here and has lasso selection.
- **Plain random sampling hides exactly what you're looking for.** The far-away cluster of
  rare points disappears.
- **float32 halves the payload** for every library that ships raw points.
- **mo.ui.matplotlib** gives box and lasso selection on a static image, mapped back to rows in
  Python. It worked at 10M points with a 50 KB PNG, which makes it a cheap way to select from huge data.

## Suggestion for anything_analyzer (the "1M+ points" item in the design plan)

1. Up to about 500k rows: a WebGL scatter (jupyter-scatter or Plotly Scattergl) with all points.
2. Above that: a server-rasterized overview (datashader / HoloViews `rasterize`, zoom re-renders),
   plus an overlay of a sample that keeps every outlier, so single rows stay hoverable and selectable.
3. Make the outlier rule a registered `LazyFrame -> LazyFrame` op. The notebook's `flag_outliers`
   (robust distance from the median, scaled by MAD) is already in that shape, so groups can swap in
   domain rules such as "headway over 20 min".
4. Show the "N of M points shown, K outliers kept" line on every plot so users know what they're seeing.
