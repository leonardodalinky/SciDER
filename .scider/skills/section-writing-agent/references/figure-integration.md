# Figure Integration

Conventions for including figures in the LaTeX paper, per the Section
Writing Agent prompt (App. F.1 p.48, item 5 "Figures and Visual Fidelity").

## Where the figures live

After Step 2 (Plotting Agent), figures are at:

```
workspace/figures/
├── fig_framework_overview.png
├── fig_main_results.png
├── fig_ablation_temperature.png
└── captions.json
```

The Section Writing Agent must reference them with the exact filenames,
including the `.png` extension:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_framework_overview.png}
\caption{Overview of the proposed pipeline. Raw video frames flow into the
frozen SAM encoder; aligned audio cues are projected through the temporal
modality fusion layer before being injected into the mask decoder.}
\label{fig:framework_overview}
\end{figure}
```

## Single-column vs full-width

The prompt is explicit: in 2-column conference templates, prefer
`\begin{figure}` (single-column) unless the figure is very wide. Use
`\begin{figure*}` only for cross-column figures.

| Figure aspect ratio | Recommended environment |
|---|---|
| `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4` | `figure` (single column) |
| `16:9`, `21:9`, `4:1` | `figure*` (cross column) |
| `9:16`, `1:4` | `figure` (very tall, hangs over multiple text lines) |

## Caption placement

For figures, `\caption` goes **AFTER** `\includegraphics`:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_main_results.png}
\caption{Comparison of three temporal-attention variants on the Ref-AVS
Seen split. Bars show Jaccard index; error bars are 95% confidence
intervals over five seeds.}
\label{fig:main_results}
\end{figure}
```

(For tables, `\caption` goes BEFORE `\begin{tabular}`. Different
conventions; both correct in their respective contexts.)

## Caption text

- Pull from `workspace/figures/captions.json` keyed by `figure_id`.
- Do **not** include `Figure N:` in the caption text. LaTeX adds the prefix
  via `\caption`.
- Plain text only. No markdown bold/italic.
- 1-3 sentences. State what the figure shows AND why (the takeaway).

## Reference in prose

Every figure must be referenced in the prose:

```latex
As shown in Figure~\ref{fig:main_results}, our method outperforms both
baselines across all five splits.
```

Use `~` (non-breaking space) before `\ref{...}`. Use `Figure` capitalized
when starting a sentence; lowercase `figure` mid-sentence.

## DO NOT merge figures

The prompt forbids combining multiple figures into one display. Each
`figure_id` from the outline corresponds to exactly one `\begin{figure}`
environment.

## All figures before Conclusion

Per the prompt: "all figures must appear before the Conclusion section,
unless they are placed in an Appendix." If you have a figure that
contextually belongs in the Appendix, move it there explicitly — do not
leave it floating after the Conclusion in the main body.

## Multi-panel figures

If a figure has multiple panels (a, b, c) — e.g., a 1×3 grid showing
ablations across three settings — the panels are part of the **same** PNG
file (rendered by Step 2). Use a single `\includegraphics` and reference
sub-panels in the caption text:

```latex
\caption{Ablation study. (a) Effect of temperature. (b) Effect of dropout.
(c) Effect of layer count. Bars show validation accuracy on the Seen split.}
```

If you really need separate sub-figure environments (`subcaption` package),
that's allowed but adds complexity — prefer single-PNG multi-panel from
Step 2.
