# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt",
#     "jax[cuda]",
#     "kaleido",
#     "plotly",
# ]
# ///

import numpy as np
from differt_core.rt import DiGraph

# BEGIN_adjacency_matrix
adjacency_matrix = np.array(
  [
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
).astype(bool)

# We can test that our adjacency matrix, without TX and RX,
# is indeed symmetric
sub_adjacency_matrix = adjacency_matrix[1:-1, 1:-1]
assert np.all(sub_adjacency_matrix == sub_adjacency_matrix.T)

di_graph = DiGraph.from_adjacency_matrix(adjacency_matrix)
# END_adjacency_matrix

# BEGIN_enumerate_paths_di_graph
from_ = 0  # TX
to = 13  # RX

for i, path in enumerate(
  di_graph.all_paths(from_, to, depth=4)
):
  print(f"#{i + 1:03d}: {path}")
  # END_enumerate_paths_di_graph

from differt_core.rt import CompleteGraph

# BEGIN_enumerate_paths_complete_graph
complete_graph = CompleteGraph(
  12  # number of objects
)

from_ = 12  # Can be anything >= 12
to = 13  # Can be anything >= 12 and != from_

for i, path in enumerate(
  complete_graph.all_paths(from_, to, depth=4)
):
  print(f"#{i + 1:03d}: {path}")
# END_enumerate_paths_complete_graph

# BEGIN_count_paths
for depth in [2, 3, 4, 5, 6, 7]:
  num_paths_di_graph = sum(
    1 for _ in di_graph.all_paths(0, 13, depth=depth)
  )
  num_paths_complete_graph = len(
    complete_graph.all_paths(12, 13, depth=depth)
  )
  print(
    f"{depth = }: {num_paths_di_graph:6d} (DiGraph) "
    f"vs {num_paths_complete_graph:6d} (CompleteGraph)",
  )
  # END_count_paths
