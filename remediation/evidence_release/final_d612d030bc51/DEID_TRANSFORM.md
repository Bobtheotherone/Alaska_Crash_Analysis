# De-identification transform witness — `final_d612d030bc51` (TRACE-001)

Generated 2026-07-12T05:25:34Z by `evidence_release/make_deid_witness.py` (script SHA-256 `000f2f7b1031eb88c61520c97b9ac30d9d6ad1f192d87fb1dce1011fad0bb66a`).

**Transform (v4.1, PRIV-001).** Column `group_id` (the real `Crash Number`) is replaced
by an opaque surrogate: surrogates `T00000..` are assigned over a seeded PERMUTATION
(seed 20260712, documented for bundle-holders' re-runs) of the sorted original-id set,
identically across all prediction files (pairing across models is preserved); every
other byte of every row is unchanged. Earlier releases assigned surrogates in sorted-id
order, which preserved source-id ordering structure — a precautionary weakness, not a
demonstrated re-identification path. The id mapping is **discarded, never persisted** —
the transform is deterministic and re-runnable by anyone holding the local bundle, and
*witnessable* by anyone holding this table.

**Chain.** `manifest.json` records each ORIGINAL file's SHA-256 (`artifact_sha256`);
this table records the matching DE-IDENTIFIED file's SHA-256; the published handoff's
`04_MANIFEST_SHA256.txt` hashes the shipped copies. Original -> de-identified ->
published is therefore a closed hash chain even though the originals stay local.

**BUNDLE-CRLF-001 (disclosed).** The v3.x bundle writer hashed its in-memory LF content
but materialised files with platform newlines (CRLF on Windows), so `artifact_sha256`
matches the on-disk originals only after CRLF→LF normalisation. Both hashes are recorded
below; the manifest match is evaluated on the normalised bytes. The v4 bundle writer
writes the exact bytes it hashes.

| prediction file | original SHA-256 (on disk) | original SHA-256 (LF-normalised) | matches manifest via | de-identified SHA-256 | rows | oMAE (recomputed) | metrics vs manifest | published copy |
|---|---|---|---|---|---|---|---|---|
| `predictions_decision_tree.csv` | `b5146422c50ab1535265ba489e8fa20a1be151ea400505017056533a90dd1014` | `53e6619b785195c757076079fecf7617c468c691a14f4751a32ff57d32de1348` | raw | `01d1901e46111c5776d5465b893f2bed17a005d4358a004bf7c73322dba7fdf4` | 11630 | 0.603955 | ok | - |
| `predictions_ebm.csv` | `bebbdd9963954cf3f2f4872992f8fe5d1509d1538d8b7072587cfdc651d6577d` | `bf91ddef8c1d459f8772febcd777072c641dc5908d5991af08c4c5000bad1e92` | raw | `d5a0bb8ab5541d5a8838aa588f9227788b161c8e50cc600dc6427adc7b6f2e6e` | 11630 | 0.494067 | ok | - |
| `predictions_frank_hall_logistic.csv` | `71152e7f20f88dfdddbbfece86084a2d40439d0f5c3a818c54d2863a39514f57` | `2f5ff17c723126b01e491471638d3fb05471f10a6a9807facb7259fbb1048f51` | raw | `d0035a44677e2e9f393bf9100b5f0c3a70abccf63f26240ada24901691e68d3f` | 11630 | 0.553396 | ok | - |
| `predictions_majority.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_multinomial_logistic.csv` | `3f2bff8a0280226856160f0cb88f8f20133c03c8462687bff37112760f525d99` | `2a521a8d885f2be3dd828b32145b1bf57b81aa9a69fb345dbe7d00ff5e733a86` | raw | `39c01c3e7716e1b3a1a448ea7cf42777c13d1d496fdaa06b74f0cc2ce8c011eb` | 11630 | 0.544884 | ok | - |
| `predictions_ordinal_median.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_ordinal_random_forest.csv` | `bbf12f58aa93540c65be49e159edd69c186c8b102a8266fa936f3fe72c175f50` | `a4c6aaf440b049c4c94d6de8233f78fea58f4fe9930f919c509bf80f1d42117d` | raw | `e7f857c14ef735d1a6a4449e186cb37455756ceae3852197e0bb60a24f8f39fc` | 11630 | 0.352279 | ok | - |
| `predictions_ordinal_random_forest_unweighted.csv` | `48e35b0e43ee62228bc7f1a97fbaaa839a82708d21ff4b225ce690f57cd7ffc4` | `27542cae572408c65243c3c078fbe9e5d7f2bbbf9154ca2574a5c052fae9ec16` | raw | `b0fede946671ca4d5c4c1bcc980ff38b20c6a3ee95a5c5036a575eb2f2c9a7be` | 11630 | 0.338779 | ok | - |
| `predictions_prior_probability.csv` | `60a0de5ac75c185bf229f5c070f8efbe024e6d88200d26ea7580bf3191992581` | `7f5c2c3b200b47c4981ca0ed6a8b53ad93130a6f5a73cae1ac4dcf4a0d6c3a93` | raw | `7ac4020a1159acb95e56befbcaaa44c53b96f9e109bd4bebf2812e82cf4b6abc` | 11630 | 0.361393 | ok | - |
| `predictions_proportional_odds.csv` | `fb455a16e970b93ce1e2ce8bfe5b23639c8b9b11269975b1b695c5b8f2fecffc` | `eb2eedd7115337e8441d429664dd87c502b1ff1c2accf5b7ccc09cd30645a52e` | raw | `b85a2524b3267802bdbb2f9197ddf98480d62226997963a09050b43c402e1abe` | 11630 | 0.536973 | ok | - |
| `predictions_random_forest.csv` | `baaa62aad33a07cd202b3d3e294c1b5a3d8db5e124704fbe329618b5e5464025` | `ab69e5d523af666c4c972cb4f0f54528f4467c2fe1ed0b90e90d447d7cd5c0aa` | raw | `eb6382b024a33c59dda109d3b17869645b7ca91dea32cf76c8b5d1227508a090` | 11630 | 0.381685 | ok | - |
| `predictions_random_forest_unweighted.csv` | `fdf506ce5615894d5a0f46ceebf7bf1677af159bca3a07085d181d8aaa1c34ca` | `53a16852cf319ea1b0d6d728dbc3855ee5fbf4c8e849a34a95670abec865bd4b` | raw | `db85418d2ef32d3e6898522f3fe36edea66560646670eb613a0ad669bed60ab1` | 11630 | 0.338779 | ok | - |
| `predictions_shallow_tree.csv` | `2bb3195c81c5d133d482b8784c15bce1f263d6424a459c2292e20c1a9fa3bcbb` | `9fc169066ef7ec95ad7dfd3c86a1cb4c97e0063ed23e6c293aea90074e24d03c` | raw | `d4b2f5e542134fe9c89399146eaefec11ee2e7345b41acd60114f4fbb943dee1` | 11630 | 0.627601 | ok | - |
| `predictions_xgboost.csv` | `8e900acc8b53f4583ed19c5ce0c841b00d55c1f9faf27d4bc009a9bf7f770904` | `cde59ee5495cf8ea89bf8d392ccca9921f9890293074368c9d643017d3c9d5cc` | raw | `eff1e79a4d6bc7c884c826f0d14de97cec69c2218b6f2ae2a7dfac36ce9e5bec` | 11630 | 0.455718 | ok | - |

Surrogate domain: `T00000..T11629` (11630 crashes).
Verification for a reviewer WITHOUT the local bundle: hash the shipped de-identified
files and compare to column 3; recompute any metric from their rows and compare to
`manifest.json -> results`. Verification WITH the local bundle: rerun this script.
