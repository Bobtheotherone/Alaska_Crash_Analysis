# De-identification transform witness — `final_8af9d5bc23d8` (TRACE-001)

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
| `predictions_decision_tree.csv` | `8fe05c95f9f22fdc4885b199f849a47507ce3cc23695c27a48e40a861ea37c04` | `f21a623e346d1a34423bfab0c47c2e7b8cf3028ed04d8203decfaae5d448ea51` | raw | `85a7c4cce706f7785fcefffaf125e9dd9001a047982da309d04a90800580b634` | 11630 | 0.603955 | ok | - |
| `predictions_ebm.csv` | `719e18c057947a0a35f4284884ac40e806ca8fbb04f448ba14de565c1417d493` | `f97ec6ad234a4c8a925097df0585b51f69a26cea5de975af1fa4ce5406b68ab9` | raw | `4995506fa5e27c76dbd33bbc8e7a7712ca3077d6c3be8738347945535be247b3` | 11630 | 0.479966 | ok | - |
| `predictions_frank_hall_logistic.csv` | `35c1dbcb6eda26f85d1f7de7e3c7c0190cb139420cfd4598b2901e64e236f482` | `dd43ea28872b3fff265b46547127df26069f2a37e692692f785130e89f544d1b` | raw | `f887f46496d7f4c00f43384eaa7ae0ebd7498b7433596e5320d1df643dc1eed5` | 11630 | 0.519433 | ok | - |
| `predictions_majority.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_multinomial_logistic.csv` | `2a8ab41c948d4eff36a99f0522ff6f7aa64cecc336914a5e5e57bc14e5a705a6` | `1b652166535ae1bfaf0ff48c90cc7f9530b5c538b99c9df048d946d2b4e3973b` | raw | `53d9b2e75c9b77ff3ee404fa56945410400512407b35a743e9ed994ee69beffe` | 11630 | 0.520808 | ok | - |
| `predictions_ordinal_median.csv` | `9c6b921b018d9ae641f3a917d1d403775249f4b897998ed980a769310bd2fcb0` | `41d1b7db1e0236630a74e3e86b7ff365567ee49a3503adb910ab77749a37aea9` | raw | `04ebb27dca1ecf18d033468d3cabd681f61c63638630729dd632df6bb5915bdb` | 11630 | 0.361393 | ok | - |
| `predictions_ordinal_random_forest.csv` | `c12f10984a037e8efae57ab58531728a535a4e6546b6e372d9fbc51279e9b860` | `4a0396943a2bfee940eef1fab46ac28b47684c48fb4c00fb357b244ad9c0ec75` | raw | `c146abc4b9758b6096cbc4626f236e2bf348adfcc6be6a4f101cad0956dbae81` | 11630 | 0.346948 | ok | - |
| `predictions_ordinal_random_forest_unweighted.csv` | `f2f6a6b81aa73220104a03d4b59ba87de69cddea3fef15596184db972dbed9c5` | `8fab4a0316772d2e80642f09d2325aaa4c2990776cc3e96d268206377c2e0a37` | raw | `d5340bb25089dfc0bfe1b0a394538b2c17407cdae9afbacbd6c5d1a5a64471a3` | 11630 | 0.342304 | ok | - |
| `predictions_prior_probability.csv` | `60a0de5ac75c185bf229f5c070f8efbe024e6d88200d26ea7580bf3191992581` | `7f5c2c3b200b47c4981ca0ed6a8b53ad93130a6f5a73cae1ac4dcf4a0d6c3a93` | raw | `7ac4020a1159acb95e56befbcaaa44c53b96f9e109bd4bebf2812e82cf4b6abc` | 11630 | 0.361393 | ok | - |
| `predictions_proportional_odds.csv` | `d3e64f28a1b5f5e0afca2700bed1f43ad1b8e12035a37c36257b2ce1a09b2ca9` | `09852291812085d89fc32bb1741e15ce09dff501ea7475391b1d7e9171487e0f` | raw | `2dfacc6fe0f05f4dd557eed797eb012a5b8dd22738b942567d55df0cbc44adab` | 11630 | 0.511780 | ok | - |
| `predictions_random_forest.csv` | `3152638dc320409533b10c2143605c49adecb9df0dae232969de020b82cfeec9` | `6fa8cbc112a0ed8cdb65dd584463ba506f0dbb8163b176b852f8f366ba66029b` | raw | `b375f17c99dcee542448e943351e4fc82fdc1f32b48f71b0356fb97fffed1831` | 11630 | 0.376440 | ok | - |
| `predictions_random_forest_unweighted.csv` | `c1df65de7ff2de62cbe980a2969f459e363c36209ea3089d29b03f9da81d73eb` | `19a031528c827ee33e4297b364850ff0dd908ccf735d8678e90e97231e8af317` | raw | `c258f7bb2d0dc445610e66f661f45164064b20a56245fa8b63690f5a5c538371` | 11630 | 0.340069 | ok | - |
| `predictions_shallow_tree.csv` | `43c5c9927c1291dfdccb303cbae38549ea5b206710bf28603448bf3e9f78a88b` | `675a5dae9fdd8ef55e23411c41c46011678f1a87cbde6eb8a6b9d936575202e2` | raw | `dc60e05ffbd45b66ae603ac586a7feab074d25bb5d937277e4818e721e1e3890` | 11630 | 0.627171 | ok | - |
| `predictions_xgboost.csv` | `7deaee37cc5d13e4671081938807b6793478b3516e8325d87efadefffef982ce` | `c218e160256339e9314eb7ab3efae30f9b940b9fcbfa85fdde32658d258a2ded` | raw | `60cf43c88a2eed5ab4c31e7d6645f0b2d23c60c3dc7f72655da9b85dc2938d19` | 11630 | 0.444196 | ok | - |

Surrogate domain: `T00000..T11629` (11630 crashes).
Verification for a reviewer WITHOUT the local bundle: hash the shipped de-identified
files and compare to column 3; recompute any metric from their rows and compare to
`manifest.json -> results`. Verification WITH the local bundle: rerun this script.
