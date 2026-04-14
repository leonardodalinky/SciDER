# Conceptual Diagram Patterns

For `plot_type == "diagram"` figures: framework overviews, system pipelines,
algorithmic flows, architectural block diagrams.

The paper's Fig. 1 (PaperOrchestra overview) is a canonical example: boxes
representing agents, arrows showing data flow, grouped sub-systems with
labeled inputs and outputs.

## Tools

- **matplotlib patches** — best for hand-controlled layouts (boxes-and-arrows
  with exact positioning, labels, and group rectangles). No external deps.
- **graphviz** — best for DAGs where layout doesn't matter; requires the
  `graphviz` Python binding and the `dot` system binary. Use only if the
  diagram is purely topological.

This skill defaults to matplotlib because it ships in `requirements.txt` and
needs no system binary.

## Block diagram pattern (boxes + arrows)

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# 16:9 frame
fig, ax = plt.subplots(figsize=(7.0, 3.94))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis('off')

PALETTE = {
    'input':   '#cfe2f3',
    'agent':   '#9fc5e8',
    'output':  '#b6d7a8',
    'control': '#ead1dc',
    'border':  '#2060cc',
}

def box(x, y, w, h, text, color):
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                        ec=PALETTE['border'], fc=color, lw=0.8)
    ax.add_patch(bb)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=8, fontweight='bold')

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', mutation_scale=10,
                        color=PALETTE['border'], lw=0.9)
    ax.add_patch(a)

# Inputs (left column)
box(0.2, 4.8, 1.6, 0.7, "Idea (I)",            PALETTE['input'])
box(0.2, 3.6, 1.6, 0.7, "Exp Log (E)",         PALETTE['input'])
box(0.2, 2.4, 1.6, 0.7, "Template (T)",        PALETTE['input'])
box(0.2, 1.2, 1.6, 0.7, "Guidelines (G)",      PALETTE['input'])

# Agents (middle)
box(2.6, 3.6, 1.8, 1.0, "Outline\nAgent",      PALETTE['agent'])
box(5.0, 4.8, 1.8, 0.9, "Plotting\nAgent",     PALETTE['agent'])
box(5.0, 3.0, 1.8, 0.9, "Lit Review\nAgent",   PALETTE['agent'])
box(5.0, 1.2, 1.8, 0.9, "Section\nWriter",     PALETTE['agent'])
box(7.5, 2.4, 2.0, 1.6, "Refinement\nAgent",   PALETTE['control'])

# Output
box(7.5, 0.4, 2.0, 0.9, "paper.tex\n+ paper.pdf", PALETTE['output'])

# Arrows
for y_in in (5.15, 3.95, 2.75, 1.55):
    arrow(1.8, y_in, 2.6, 4.1)
arrow(4.4, 4.1, 5.0, 5.25)
arrow(4.4, 4.1, 5.0, 3.45)
arrow(4.4, 4.1, 5.0, 1.65)
for y in (5.25, 3.45, 1.65):
    arrow(6.8, y, 7.5, 3.2)
arrow(8.5, 2.4, 8.5, 1.3)

fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()
```

## Pipeline / flowchart pattern (linear with branches)

Same primitives, just arranged left-to-right with `arrow(x1, y1, x2, y2)`
between successive boxes. For parallel branches (Steps 2 ∥ 3 in
PaperOrchestra), draw two parallel rows with a fork-and-join.

## Algorithm-as-graph pattern

For diagrams that are *just* a topological graph (no labels of importance,
just nodes-and-edges), use graphviz:

```python
import graphviz
g = graphviz.Digraph(format='png')
g.attr(dpi='300', rankdir='LR', fontname='Times-Roman', fontsize='8')
g.attr('node', shape='box', style='rounded,filled',
       fillcolor='#cfe2f3', color='#2060cc', fontname='Times-Roman', fontsize='8')
g.attr('edge', color='#2060cc', fontname='Times-Roman', fontsize='7')
g.edge('Input', 'Encoder')
g.edge('Encoder', 'Decoder')
g.edge('Decoder', 'Output')
g.render(out_path.replace('.png', ''), cleanup=True)
```

## Anti-patterns

- Never use clip-art icons or emoji.
- Never use color as the *only* signal (always pair with shape or label).
- Never let arrows cross labels — re-route around.
- Never use a bitmap background.
- Always `axis('off')` for diagrams; no spines, no ticks, no grid.
