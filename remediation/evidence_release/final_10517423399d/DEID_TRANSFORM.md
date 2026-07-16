# De-identification transform witness — `final_10517423399d` (TRACE-001)

Generated 2026-07-12T05:25:33Z by `evidence_release/make_deid_witness.py` (script SHA-256 `000f2f7b1031eb88c61520c97b9ac30d9d6ad1f192d87fb1dce1011fad0bb66a`).

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
| `predictions_decision_tree.csv` | `7cef5169aa333175c01de7965a9ab6c5b2ceeb15d462c2fc12f260e5b969245c` | `4a8511fae6a61a32d1c4700528346c42d6039cc1a5bc2605751d0a97c60347da` | raw | `e80536095c15efa091137feb2cb3133d389b786bdee0fd55d4a34d13641a404b` | 11630 | 0.587876 | ok | - |
| `predictions_ebm.csv` | `0c46a6642145eb57abb7c2ad5a75373690e2efb21be9b3581dd627a1ab0afdbf` | `2228b90a03be5c438cb34bd457e437900e902a4127fc73efa9f72da796bde6ff` | raw | `67553b69bbafa138639d3a9bc051ae6427d5044ca80ec949398440f1d4c47cb5` | 11630 | 0.420980 | ok | - |
| `predictions_frank_hall_logistic.csv` | `f7282cd742e26f6df6448e0f26b3f1cb45095cef1332beb368edabc92bea3d34` | `3f8b2d3e62ed41e12a72824eee12ad87caa214103868adc5a1410949899ff3ea` | raw | `3c116846004186add96bfb101bc8ce8662996b6c13147c0611f31ed1a5422f6b` | 11630 | 0.474291 | ok | - |
| `predictions_majority.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_multinomial_logistic.csv` | `9bd0147bcc0d182b47f245dc84186bf254fbb39b35c6a1aaee7f6be800994395` | `96700dcc58e9d7f9d168921dc4bec4bfad3c6ac7ff4fce1edfd78099a1f0e201` | raw | `be3af951abfa6919979d46986c9341872ea097af3f5e172b8d20a3c6516f7462` | 11630 | 0.479278 | ok | - |
| `predictions_ordinal_median.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_ordinal_random_forest.csv` | `f872c1297f2ca9ad2470644fffa3ac93fbca128128f0622826d77e5ff392ca04` | `e04c1890a9443bc1c7aed1611ada7687cc23a6fa4e54776f82c25155a2a7999e` | raw | `4f8f25261b60336ab4bbcc3d000ebdc2e673138b971d14b153782348ddb34888` | 11630 | 0.332674 | ok | - |
| `predictions_ordinal_random_forest_unweighted.csv` | `b6ae3c757c20cb547af980c479e5f5d54d2b8fa5b1e2702b4307d02fd0fcf0d3` | `411e660beffd8bde870fd55e67815d7ac61d98878530b78d02e49e659836c9ce` | raw | `b95843894638775e784fd2bdee28dd081a5e2d1a91332332045ac7304207da3a` | 11630 | 0.325279 | ok | - |
| `predictions_prior_probability.csv` | `60a0de5ac75c185bf229f5c070f8efbe024e6d88200d26ea7580bf3191992581` | `7f5c2c3b200b47c4981ca0ed6a8b53ad93130a6f5a73cae1ac4dcf4a0d6c3a93` | raw | `7ac4020a1159acb95e56befbcaaa44c53b96f9e109bd4bebf2812e82cf4b6abc` | 11630 | 0.361393 | ok | - |
| `predictions_proportional_odds.csv` | `7a15121f9b5d949d1b0131385d294f415817464ebca06c42419cf8c8666e1751` | `ad7302e1341b4ccf7d2aecc25294cee4e3d7050d045ec42d6465d6049f9e09f4` | raw | `55a46e24a8ce652b5ff63cea67f8b72bb8658261b6692f0eb4bb10b30a9b693e` | 11630 | 0.481083 | ok | - |
| `predictions_random_forest.csv` | `0cb8d1884639b46cc1510f3dfee25fcb3dbe729c5d9b986af722d3180bfa107d` | `0d1d9042f39a874392c9fd5178ab4b1b5b3a29945217cbe5d6b1eb13526851e0` | raw | `ef1353da637273e2d8196dda054f56b7b3a91a34d926f6e097be4828647e8c2b` | 11630 | 0.355202 | ok | - |
| `predictions_random_forest_unweighted.csv` | `324419999b9dcc7ed3392144cd13a83d6d83ba35d7f2f4cd54822a4efd6e3a8a` | `132f1ea1561a83da3b4932e168641b022d1e517b76dce676f53c08e0c7a926d8` | raw | `9612af77252e9070c9cc766f526ed89585b653519c8d53a819a027bb43374679` | 11630 | 0.325795 | ok | - |
| `predictions_shallow_tree.csv` | `187edaf58db37e0a5335717213349b0fd4dec5b44d123de80af7c7450817cad1` | `72a810eabd1f7f93d78d430a590ec77222d2a57f2c47074524ec8d988cda3630` | raw | `56e69470b20b6c467b7ccf70ba2f5ef9746c88d3cefc53430265c60d44c2d2b3` | 11630 | 0.578160 | ok | - |
| `predictions_xgboost.csv` | `fee2d4bcbfe4cb556d6f08b06281e80fb4da4689d662c3bd86252253ac41b55c` | `f4fb019e825f597f3fdeddc3bd7510e182304cd76b43d10d652b18711e8be4a9` | raw | `387793845c76d2e48ad28627000fcc3dde59b0a17211954128e3719036b35dac` | 11630 | 0.392777 | ok | - |

Surrogate domain: `T00000..T11629` (11630 crashes).
Verification for a reviewer WITHOUT the local bundle: hash the shipped de-identified
files and compare to column 3; recompute any metric from their rows and compare to
`manifest.json -> results`. Verification WITH the local bundle: rerun this script.
