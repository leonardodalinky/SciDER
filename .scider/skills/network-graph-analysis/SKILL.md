---
name: network-graph-analysis
description: Graph and network analysis — graph construction, centrality measures, community detection, graph ML basics, and visualization. Use when data has relational/network structure.
allowed_agents: [data, experiment]
---

# Network Graph Analysis

## Overview

Graph and network analysis extracts structural information from relational data. Use this skill when your data describes connections between entities — citations, protein interactions, social ties, trade flows, co-occurrence matrices, or any adjacency structure. The key insight: if relationships carry information that tabular rows cannot, model the structure explicitly.

## When to Use This Skill

Use this skill when:
- Your data is fundamentally relational (entities linked by edges)
- You need to identify influential nodes, bridging nodes, or tightly-knit communities
- You are building or evaluating a graph neural network (GNN)
- You need to visualize network structure for publication or exploration
- Your dataset is a co-occurrence matrix, adjacency matrix, or edge list

Do **not** reach for graph methods if rows in your tabular dataset are independent (no meaningful pairwise relationships). First run EDA (EDA skill) to understand data shape, then apply this skill.

---

## Graph Construction from Data

### From a NumPy/SciPy Adjacency Matrix

```python
import numpy as np
import networkx as nx
import scipy.sparse as sp

# Dense adjacency matrix
A = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])

G = nx.from_numpy_array(A)                          # undirected
G_directed = nx.from_numpy_array(A, create_using=nx.DiGraph())  # directed

# SciPy sparse matrix (memory-efficient for large graphs)
A_sparse = sp.csr_matrix(A)
G_sparse = nx.from_scipy_sparse_array(A_sparse)
```

### From an Edge List CSV

```python
import pandas as pd

# CSV with columns: source, target (and optionally weight)
edges_df = pd.read_csv('edges.csv')

# Unweighted, undirected
G = nx.from_pandas_edgelist(edges_df, source='source', target='target')

# Weighted, directed
G_weighted = nx.from_pandas_edgelist(
    edges_df,
    source='source',
    target='target',
    edge_attr='weight',
    create_using=nx.DiGraph()
)
```

### From Bipartite Data (e.g., users x items)

```python
from networkx.algorithms import bipartite

# Build bipartite graph
B = nx.Graph()
B.add_nodes_from(['u1', 'u2', 'u3'], bipartite=0)  # user nodes
B.add_nodes_from(['i1', 'i2'], bipartite=1)          # item nodes
B.add_edges_from([('u1','i1'), ('u2','i1'), ('u2','i2'), ('u3','i2')])

# Project onto one set (user-user co-occurrence)
user_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
G_projected = bipartite.weighted_projected_graph(B, user_nodes)
```

### Weighted vs Unweighted / Directed vs Undirected

| Choice | When to use |
|--------|-------------|
| **Undirected** | Symmetric relationships (co-authorship, friendship) |
| **Directed** | Asymmetric relationships (citations, follows, flows) |
| **Unweighted** | Only presence/absence of edge matters |
| **Weighted** | Edge strength matters (call frequency, correlation strength) |

Use `nx.DiGraph()` for directed, `nx.Graph()` for undirected. Add `weight` edge attributes for weighted graphs — most centrality functions respect the `weight` keyword.

---

## Basic Graph Statistics

```python
G = nx.karate_club_graph()  # example

print(f"Nodes:               {G.number_of_nodes()}")
print(f"Edges:               {G.number_of_edges()}")
print(f"Density:             {nx.density(G):.4f}")         # edges / possible edges

degrees = [d for _, d in G.degree()]
print(f"Average degree:      {sum(degrees)/len(degrees):.2f}")
print(f"Max degree:          {max(degrees)}")

# Connected components
components = list(nx.connected_components(G))
print(f"Connected components: {len(components)}")
print(f"Largest component:   {max(len(c) for c in components)} nodes")

# Diameter (only valid for connected graphs)
if nx.is_connected(G):
    print(f"Diameter:            {nx.diameter(G)}")
else:
    # Diameter of largest component
    largest = G.subgraph(max(components, key=len))
    print(f"Diameter (LCC):      {nx.diameter(largest)}")

# Clustering coefficient
print(f"Global clustering:   {nx.transitivity(G):.4f}")
print(f"Avg local clustering:{nx.average_clustering(G):.4f}")
```

---

## Centrality Measures

### Degree Centrality — Who Has the Most Connections?

The simplest measure. Node importance = fraction of other nodes it connects to. Use as a baseline; fast to compute.

```python
degree_cent = nx.degree_centrality(G)

# For weighted graphs, use strength (sum of weights)
strength = dict(G.degree(weight='weight'))

# Top 5 nodes by degree centrality
top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top nodes by degree centrality:", top_degree)
```

### Betweenness Centrality — Bridges Between Communities

Fraction of shortest paths passing through a node. High betweenness = bottleneck / broker. Expensive: O(VE) for unweighted, slower for weighted. Sample for large graphs.

```python
# Exact
between_cent = nx.betweenness_centrality(G, normalized=True, weight='weight')

# Approximate for large graphs (sample k sources)
between_approx = nx.betweenness_centrality(G, k=200, normalized=True)

top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top nodes by betweenness:", top_between)
```

### Closeness Centrality — Reaches Others Quickly

Average shortest path distance to all other nodes (inverted). High closeness = spreads information fast. Meaningful only in connected graphs (or apply per component).

```python
closeness_cent = nx.closeness_centrality(G)

top_close = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top nodes by closeness:", top_close)
```

### Eigenvector Centrality — Connected to Important Nodes

Score proportional to scores of neighbors. Being connected to high-scoring nodes raises your own score. PageRank is a damped variant. Use for undirected or directed networks where prestige propagates.

```python
eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')

# PageRank for directed graphs
pagerank = nx.pagerank(G, alpha=0.85, weight='weight')

top_eigen = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top nodes by eigenvector centrality:", top_eigen)
```

### When to Use Each

| Measure | Best for |
|---------|----------|
| **Degree** | Quick scan, finding hubs, baseline |
| **Betweenness** | Finding brokers, bridges, critical infrastructure nodes |
| **Closeness** | Information diffusion, finding broadcast centers |
| **Eigenvector/PageRank** | Influence, authority, prestige propagation |

Compute all four and compare — they often disagree and the disagreement is informative.

---

## Community Detection

### Louvain — Fast, Large Graphs (Recommended Default)

Maximizes modularity greedily. Non-deterministic; run multiple times and pick highest modularity. Available via `python-louvain` (`community` package) or `igraph`.

```python
# Using python-louvain
import community as community_louvain

partition = community_louvain.best_partition(G, weight='weight', random_state=42)
modularity = community_louvain.modularity(partition, G)

num_communities = len(set(partition.values()))
print(f"Communities: {num_communities}, Modularity: {modularity:.4f}")

# Assign as node attribute
nx.set_node_attributes(G, partition, 'community')

# Using igraph (faster on very large graphs)
import igraph as ig
G_ig = ig.Graph.from_networkx(G)
louvain_result = G_ig.community_multilevel(weights='weight')
print(f"Modularity (igraph): {louvain_result.modularity:.4f}")
```

### Girvan-Newman — Hierarchical, Small Graphs

Removes edges with highest betweenness iteratively, revealing community hierarchy. Slow (O(E²V)); use only when n < 1000 and you need a dendrogram.

```python
from networkx.algorithms.community import girvan_newman

comp = girvan_newman(G)
# Get partitions at different resolution levels
for communities in itertools.islice(comp, 3):
    print(f"k={len(communities)}: {[len(c) for c in communities]}")
```

### Spectral Clustering — Exactly k Communities

Use when you know k in advance. Builds graph Laplacian, finds k eigenvectors, runs k-means in spectral space.

```python
from sklearn.cluster import SpectralClustering
import numpy as np

A = nx.to_numpy_array(G)
sc = SpectralClustering(
    n_clusters=5,
    affinity='precomputed',
    assign_labels='kmeans',
    random_state=42
)
labels = sc.fit_predict(A)
```

### Label Propagation — Very Fast, Non-Deterministic

Each node inherits the most common label of its neighbors. Converges in O(E). Results vary between runs; use when speed matters more than stability.

```python
from networkx.algorithms.community import label_propagation_communities

communities = list(label_propagation_communities(G))
print(f"Found {len(communities)} communities")
```

---

## Graph ML Basics

### When Does Graph Structure Help?

Graph structure adds value **beyond tabular features** when:
- Nodes are connected in ways that predict the target (social influence, protein proximity)
- Labels propagate through edges (homophily: connected nodes share labels)
- Global topology matters (scale-free vs random network properties)

If nodes have no meaningful connections or edges are arbitrary, stick with tabular models.

### GNN Overview: Message Passing

At each layer, each node aggregates feature vectors from its neighbors, then updates its own representation. After k layers, a node sees its k-hop neighborhood.

Key architectures:
- **GCN** (Graph Convolutional Network): normalized sum aggregation — simple, effective
- **GraphSAGE**: sample-and-aggregate — scales to large graphs via mini-batching
- **GAT** (Graph Attention Network): learns attention weights over neighbors — best when neighbor importance varies

### Task Types

| Task | Input | Output | Example |
|------|-------|--------|---------|
| **Node classification** | Node features + graph | Label per node | Classify paper topics in citation network |
| **Link prediction** | Node features + partial graph | Edge probability | Recommend friends, predict protein interactions |
| **Graph classification** | Full graph | Label per graph | Classify molecules, detect fraud subgraphs |

### Libraries

```python
# PyTorch Geometric (PyG) — most popular, extensive model zoo
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch

data = Data(
    x=node_features,       # [num_nodes, num_features]
    edge_index=edge_index, # [2, num_edges] — COO format
    y=labels               # [num_nodes] for node classification
)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# DGL — alternative with strong support for heterogeneous graphs
import dgl
g = dgl.from_networkx(G)
```

---

## Visualization

### NetworkX + Matplotlib — Small Graphs (<500 nodes)

```python
import matplotlib.pyplot as plt
import networkx as nx

fig, ax = plt.subplots(figsize=(12, 8))

# Layout algorithms
pos = nx.spring_layout(G, seed=42)           # force-directed — general purpose
# pos = nx.kamada_kawai_layout(G)            # minimizes edge crossings — slower but cleaner
# pos = nx.circular_layout(G)               # nodes on a circle — good for small n
# pos = nx.spectral_layout(G)               # based on graph Laplacian eigenvectors

# Color by community
communities = community_louvain.best_partition(G)
colors = [communities[n] for n in G.nodes()]

# Size by degree
sizes = [300 * nx.degree_centrality(G)[n] + 50 for n in G.nodes()]

nx.draw_networkx(
    G, pos,
    node_color=colors,
    node_size=sizes,
    cmap=plt.cm.tab20,
    with_labels=True,
    font_size=8,
    edge_color='gray',
    alpha=0.8,
    ax=ax
)
ax.set_title("Network with community coloring")
plt.tight_layout()
plt.savefig('network.png', dpi=150, bbox_inches='tight')
```

### Pyvis — Interactive HTML Visualization

```python
from pyvis.network import Network

net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
net.from_nx(G)

# Customize nodes
for node in net.nodes:
    node['size'] = degree_cent[node['id']] * 50 + 5
    node['title'] = f"Node {node['id']}, degree={G.degree(node['id'])}"

net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100
    },
    "solver": "forceAtlas2Based"
  }
}
""")
net.save_graph('network.html')
```

### Gephi — Large Graphs (>5000 nodes)

Export from networkx and open in Gephi for GPU-accelerated layout and interactive exploration:

```python
# Export to GEXF (Gephi's native format)
nx.write_gexf(G, 'network.gexf')

# Or GraphML
nx.write_graphml(G, 'network.graphml')
```

In Gephi: use ForceAtlas2 layout, color by modularity class (run it under Statistics), size by degree. Export as SVG for publication.

---

## Evaluation

### Community Detection

```python
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# When ground truth labels are known
true_labels = [ground_truth[n] for n in G.nodes()]
pred_labels = [partition[n] for n in G.nodes()]

nmi = normalized_mutual_info_score(true_labels, pred_labels)
ari = adjusted_rand_score(true_labels, pred_labels)
print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")

# When no ground truth — use modularity
modularity = community_louvain.modularity(partition, G)
print(f"Modularity Q: {modularity:.4f}")
# Q > 0.3 is generally considered significant community structure
```

### Link Prediction

```python
from sklearn.metrics import roc_auc_score
import random

# Create train/test split by hiding edges
edges = list(G.edges())
test_edges = random.sample(edges, int(0.2 * len(edges)))
G_train = G.copy()
G_train.remove_edges_from(test_edges)

# Generate negative samples (non-edges)
non_edges = list(nx.non_edges(G))
test_non_edges = random.sample(non_edges, len(test_edges))

# Score with common neighbors as baseline predictor
def score_pairs(G, pairs):
    return [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs]

scores_pos = score_pairs(G_train, test_edges)
scores_neg = score_pairs(G_train, test_non_edges)

y_true = [1]*len(test_edges) + [0]*len(test_non_edges)
y_score = scores_pos + scores_neg
auc = roc_auc_score(y_true, y_score)
print(f"Link prediction AUC (common neighbors): {auc:.4f}")

# networkx also has Jaccard, Adamic-Adar, resource allocation predictors built-in
```

### Node Classification

Use standard classification metrics (accuracy, F1, AUC) from sklearn. Evaluate on a held-out set of nodes — **do not leak graph structure** from test nodes into training via multi-hop neighborhoods unless you explicitly account for it.

---

## Best Practices

1. **Inspect the degree distribution first** — scale-free (power-law) vs Poisson shapes imply different generative processes
2. **Always check for isolated nodes and giant component fraction** — disconnected graphs require careful handling
3. **Self-loops and multi-edges** — decide explicitly: `nx.Graph()` removes multi-edges, use `nx.MultiGraph()` to keep them
4. **Large graphs (>100k nodes)** — switch to igraph or graph-tool; networkx is Python-native and slow at scale
5. **Reproducibility** — set `random_state`/`seed` for any non-deterministic algorithm (Louvain, spring layout)
6. **Null models** — compare your graph properties against an Erdos-Renyi or configuration model null to validate findings

---

## Common Pitfalls

- **Treating directed graphs as undirected**: PageRank on undirected graphs is meaningless — check `nx.is_directed(G)`
- **Diameter on disconnected graphs**: Always check `nx.is_connected(G)` first
- **Betweenness on large graphs without sampling**: It will run for hours — use `k` parameter to approximate
- **Comparing modularity across algorithms**: Modularity is maximized differently by each algorithm; comparisons are not apples-to-apples
- **Forgetting to normalize centrality**: Raw degree vs degree centrality (0–1) depend on graph size — always normalize for cross-graph comparison
