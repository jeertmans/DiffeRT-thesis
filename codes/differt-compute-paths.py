scene = TriangleScene(
  mesh=mesh,
  transmitters=tx,
  receivers=rx,
)
paths = scene.compute_paths(
  order=...,
)
