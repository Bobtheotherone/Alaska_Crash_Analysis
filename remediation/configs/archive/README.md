# configs/archive/ — retired configurations (not executable study configs)

`final_benchmark_synthetic_2016_2017.yml` (formerly `configs/final_benchmark.yml`) is the
**retired** configuration of the pre-Route-A synthetic harness era: it names synthetic
2016–2017 final years, `primary_baseline: ordinal_logistic` (a model name that no longer exists
in the registry), and a `synthetic_n` fixture size. It never described the real study and is kept
only as an archival record (its content is referenced by `research/FINAL_BENCHMARK_PROTOCOL.md`'s
change table). It is preserved byte-for-byte — do not update it.

The **canonical** executable configuration for the study is `configs/route_a_09_12.yml`
(see its header for the Route-A-filename vs Route-R-study naming note, CFG-001). The smoke config
`configs/_smoke.yml` is test-only.
