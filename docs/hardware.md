# Hardware Guidance

SNAPpy is designed to run on ordinary CPU hardware. The primary publication benchmark did not use a GPU for SNAPpy.

## Publication Benchmark Hardware

The primary SNAPpy timing results were produced on AWS `r7i.large` CPU workers:

| Resource | Specification |
|---|---|
| vCPUs | 2 |
| Physical cores | 1 core with 2 hardware threads |
| Memory | 16 GiB RAM |
| CPU | 4th-generation Intel Xeon Scalable/Sapphire Rapids class |
| GPU | None |

This means a modern laptop, desktop, or small workstation with at least 16 GB RAM is a reasonable starting point for routine 3D z-stack detection, especially when processing images one at a time.

## Practical Recommendations

For routine detection:

- Minimum practical CPU-only system: 2 CPU threads and 8-16 GB RAM.
- Recommended CPU-only system: 4 or more CPU threads and at least 16 GB RAM.
- GPU: not required for SNAPpy detection.
- Storage: use local SSD storage when processing many TIFF stacks; network drives can be slower for large batches.

For model optimization or repeated benchmarking:

- Use at least 16 GB RAM.
- Prefer more CPU threads if running many images or trials.
- Use local scratch storage for temporary outputs when possible.
- Keep benchmark result trees outside the public package repository.

Runtime depends on image dimensions, candidate density, selected recipe, and whether background correction and local fitting are enabled.

## Comparison With GPU Methods

Spotiflow exploratory GPU runs used AWS `g6.xlarge` workers with one NVIDIA L4 GPU, but those runs were not used for the primary SNAPpy timing claim. The main timing comparison in the manuscript used CPU-matched `r7i.large` workers for SNAPpy, RS-FISH, and Spotiflow.
