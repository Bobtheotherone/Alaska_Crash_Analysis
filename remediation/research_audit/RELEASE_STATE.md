# RELEASE STATE (machine-generated — do not hand-edit; regenerate at packaging)

Generated 2026-07-12T05:28:00Z by `research_audit/make_release_state.py`.

| Field | Value |
|---|---|
| commit | `3add4282b74e4b1ed521a073400c76d86a841f55` (`3add428`) |
| branch | `portfolio-research-finalization-v4` |
| tag `portfolio-v3` | `e06e8037b62b` |
| tag `portfolio-v4` | `4aa0751a8607` |
| canonical config | `configs/route_r_09_12.yml` (v4 protocol; strict tier, posterior-median) |
| test suite | **66 passed** (live run at generation time) |
| claim scan | PASS |
| PDF layout audit | PASS |

## Governed runs (committed skeletons under `evidence_release/`)

| run_id | protocol | tier | rule | generator commit |
|---|---|---|---|---|
| `final_10517423399d` | v4-structural-isolation | broad | posterior_median | `f2f73ed17` |
| `final_8af9d5bc23d8` | v4-structural-isolation | strict | posterior_median | `458f784ee` |
| `final_d612d030bc51` | v4-structural-isolation | strict | posterior_median | `3add4282b` |
| `final_f27613102c96` | ? | ? | ? | `e55d6f160` |

## Key artifact hashes (SHA-256)

| artifact | sha256 |
|---|---|
| `experiment/final_results.json` | `6578e7e1ef967040a47176ad051672dbb897580940397d015aecdec71a79089b` |
| `experiment/development_report.json` | `941170ed03d34d56abf857c1edceb142b9027c19a1a79b94726a88061f03b86f` |
| `configs/route_r_09_12.yml` | `73fc9cc6f7e65887338ae585b27d964a15546f5c5cbac9412debcb77b370d4dd` |
| `data/feature_availability_ledger.csv` | `a6293c81f4b0c9ae7650bc48d6f22a40dca65404215a5344b0d492aff2a68231` |
| `data/target_mapping.yml` | `806d02652e348d1462c9f650e25f762df05a8bc50b9d3a480d61d7df454f3301` |
| `requirements-lock.txt` | `ca3aaa32eede769b423b6df8759ee4f4b7bccd2f2eb438b2ce0673fe0b5d191d` |

Historical/planning documents (`FINAL_BENCHMARK_PROTOCOL.md` body, Route-A-era configs,
prior release notes) are retained as records of what was specified when; where any prose
conflicts with this generated file, **this file is authoritative**.
