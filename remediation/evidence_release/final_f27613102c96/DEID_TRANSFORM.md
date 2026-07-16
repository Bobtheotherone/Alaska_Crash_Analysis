# De-identification transform witness — `final_f27613102c96` (TRACE-001)

Generated 2026-07-12T05:25:32Z by `evidence_release/make_deid_witness.py` (script SHA-256 `000f2f7b1031eb88c61520c97b9ac30d9d6ad1f192d87fb1dce1011fad0bb66a`).

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
| `predictions_decision_tree.csv` | `439a9ef61dd5133939bf54d5b66ec1dbb2681bc464ca093654d8f771eca41570` | `596dcdb8041c48b462b8e9a507a0eff317c0a3302f922dae80e71355e6591c75` | LF-normalised | `87706e3878d9e898a67563c74f58dcb712cb64045bbc1bce3fcdba3cf889c580` | 11630 | 0.528977 | ok | - |
| `predictions_ebm.csv` | `fd0c192e47d4dd6f22a4509cd3ef83846f68de82e0147676529f4a01c6d9f165` | `eefeabba3912e5c3e860aae60df1f14721f0de40cb7169da35c8adeb56f4423a` | LF-normalised | `364df7b0bf67eadbb11e2fab3f1743016b45c7ae8f94870e74d41a7f67d07001` | 11630 | 0.425451 | ok | - |
| `predictions_empirical_prior.csv` | `682434c18aca36a646545179bfd0869879a7829a056586c523a01fcdbdd621ba` | `c8f1dbf630b0ba989caba1c9527a57b3eebdbb497bda2ff6f7e0e5f3e87ef233` | LF-normalised | `09776b668e26d756a9603ad86f80ce1f0758760f2113b3c00d38292590724435` | 11630 | 0.500946 | ok | - |
| `predictions_frank_hall_logistic.csv` | `73dd340b9bd48fc0ece3ad2aef5568cf8c980e69fa0567de401dcdc7dc2b81b7` | `3c335e0be7ec1f1c5f4a8808401fc5420aa0c722f584da2ca15582a814ac10cd` | LF-normalised | `75f7ba56e3049ac9c58107c1b07f62f719a2d8f2c20d40a126d9face327fe538` | 11630 | 0.484437 | ok | - |
| `predictions_majority.csv` | `0271453f1028cb17d0271e940851a0e1a5605d0f743524100608a6287ab13e64` | `417fe3665ef95875d87a48b921342ea6f9d3cc2d6fe9413369cdc5570f7e4d49` | LF-normalised | `354718e783516b7f0efabadf868c9dd6d157c02d328ba14761791ca61823eb99` | 11630 | 0.361393 | ok | - |
| `predictions_multinomial_logistic.csv` | `e6fbe72ddf9d37fd52dc065c1a010b43ea790cc2beeaa220426118fb3753f0be` | `4cea5dcf807d7dcee409dbab176d78876eaa393c069a211c9b97d40fd912e394` | LF-normalised | `f1dac84fcee579fa8cf378938e59652045ff8bb49f8a2698817884517204a064` | 11630 | 0.480739 | ok | - |
| `predictions_ordinal_median.csv` | `0271453f1028cb17d0271e940851a0e1a5605d0f743524100608a6287ab13e64` | `417fe3665ef95875d87a48b921342ea6f9d3cc2d6fe9413369cdc5570f7e4d49` | LF-normalised | `354718e783516b7f0efabadf868c9dd6d157c02d328ba14761791ca61823eb99` | 11630 | 0.361393 | ok | - |
| `predictions_ordinal_random_forest.csv` | `37a0308fb7d764e9feb54bb48b975d62ee30434ce9d2cbfc68022641b63fc7e6` | `56a3e50a8ae164b723ad6d26ae7e8c6d6a7cc1c5d80f400558ef0a2f5cde80c5` | LF-normalised | `531911b2fed8cfa65cfeba91d7f0880b250fa3a0f338fe4064eb5ea33c4d57fb` | 11630 | 0.331040 | ok | - |
| `predictions_ordinal_random_forest_unweighted.csv` | `dc4a8873506c6401da471505d9a0bee0653dbe9cd94a4c819f0249a607608101` | `e48e249cd9697a2c544cecceb1b356d485dc9938f864765884a6208ce8a79d9b` | LF-normalised | `466f29b58651a14eff28799832cab57fa0413a4f9c3027f0c5cd2e12cf290bee` | 11630 | 0.330610 | ok | - |
| `predictions_proportional_odds.csv` | `21e13b9fcfa6d0637f553db05cb3b87507fb16965304c39b13fe5b4b5165e876` | `83ff8778815c74b528c0a3b15eda910093d58851a65ecaea0014687e5e124f80` | LF-normalised | `a72ae55ba073b1f5d0d37b22eb27b54c87c0d954c0abc65e44e5a84aced95d53` | 11630 | 0.465692 | ok | - |
| `predictions_random_forest.csv` | `7ebfaa81e3fac528e33565f41cce0accf3d84f0e6abee2299bd846fab814968e` | `50157f10dbf11a3f8b8483f129b7e1c39d4e39d5c4c7699af44873ab6f500ee6` | LF-normalised | `213532d69a7f2d83077c99707d027357c9db214f7f36ddfac0e7bbcab1d32e9d` | 11630 | 0.343594 | ok | - |
| `predictions_random_forest_unweighted.csv` | `15c79142441ed884623c1c468ba5bbe96a59466484a6b271b290e232db420bb0` | `54d9ef00c455811d949a2e243a3115246c690026ea625d83e6c5953d0146ca85` | LF-normalised | `062fabeb0c3445db03b2256c6d09bad24df0dae1f241474656da662b2c8b765c` | 11630 | 0.327773 | ok | - |
| `predictions_shallow_tree.csv` | `3f1f546213d8c988f4e214760261a491169bcbdd55f210b850866b5e21caac65` | `49e52981d139e7628bb8905c46b437162b8b730c0fb82b730d05075f55dd45c9` | LF-normalised | `6d0b37affe844b76515d68788a06da0ba50e73958b457ab68bd1e615b86ec428` | 11630 | 0.743508 | ok | - |
| `predictions_xgboost.csv` | `870a40e1536b206c83c1033b5ed6a199d184956ac3b51fbcbd6f292a26cd0e89` | `74cbe809b8f8f1b6a093a22751ab52a353af66a81adc2ad5d33d95cfd8563e6f` | LF-normalised | `ac976783cb03e882c354aa2c1127c32e98c5f0f5a9e310c3057107c7296630de` | 11630 | 0.377644 | ok | - |

Surrogate domain: `T00000..T11629` (11630 crashes).
Verification for a reviewer WITHOUT the local bundle: hash the shipped de-identified
files and compare to column 3; recompute any metric from their rows and compare to
`manifest.json -> results`. Verification WITH the local bundle: rerun this script.
