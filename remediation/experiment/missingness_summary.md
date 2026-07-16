# Strict-tier missingness structure — development years only (MISS-002, v4.1)

Fields: 49 (the v4 primary strict tier after invariant drop). Missing = NaN / blank / documented missing token.

**Prespecified high-missingness rule:** fields with > 50% development missingness are removed in the low-missingness sensitivity benchmark (`configs/route_r_09_12_lowmiss_sensitivity.yml`). Flagged: `Unit 1 Person 1 Residence State`, `Direction`.

| field | overall | 2009 | 2010 | 2011 | class 0 | class 1 | class 2 |
|---|---|---|---|---|---|---|---|
| Unit 1 Person 1 Residence State | 0.831 | 1.000 | 0.995 | 0.496 | 0.825 | 0.836 | 0.894 |
| Direction | 0.639 | 0.651 | 0.637 | 0.629 | 0.641 | 0.641 | 0.582 |
| AADT | 0.322 | 0.330 | 0.333 | 0.304 | 0.334 | 0.294 | 0.311 |
| Posted Speed | 0.306 | 0.322 | 0.298 | 0.296 | 0.350 | 0.219 | 0.131 |
| Unit 1 Person 1 Gender | 0.230 | 0.244 | 0.218 | 0.227 | 0.280 | 0.130 | 0.048 |
| Temperature | 0.212 | 0.232 | 0.201 | 0.204 | 0.260 | 0.117 | 0.036 |
| Temperature Range | 0.212 | 0.232 | 0.201 | 0.204 | 0.260 | 0.117 | 0.036 |
| Roadway Characteristics | 0.198 | 0.218 | 0.189 | 0.186 | 0.246 | 0.101 | 0.032 |
| Distance From Intersection | 0.067 | 0.030 | 0.070 | 0.103 | 0.078 | 0.043 | 0.045 |
| Unit 1 Direction of Travel | 0.066 | 0.056 | 0.049 | 0.094 | 0.076 | 0.046 | 0.048 |
| Unit 1 License Plate State | 0.063 | 0.062 | 0.041 | 0.086 | 0.060 | 0.066 | 0.108 |
| Unit 1 Traffic Control | 0.058 | 0.054 | 0.034 | 0.087 | 0.066 | 0.045 | 0.027 |
| Unit 1 Model Year | 0.050 | 0.043 | 0.042 | 0.067 | 0.045 | 0.059 | 0.078 |
| Unit 1 Action | 0.048 | 0.047 | 0.027 | 0.070 | 0.048 | 0.047 | 0.062 |
| Unit 1 Person 1 Age Range | 0.048 | 0.047 | 0.044 | 0.052 | 0.050 | 0.045 | 0.030 |
| Unit 1 Person 1 Raw Age | 0.048 | 0.047 | 0.044 | 0.052 | 0.050 | 0.045 | 0.030 |
| Roadway Junction | 0.026 | 0.031 | 0.026 | 0.021 | 0.032 | 0.014 | 0.007 |
| Unit 1 Non-CV Configuration | 0.022 | 0.009 | 0.006 | 0.050 | 0.023 | 0.020 | 0.012 |
| Unit 1 CV Issuing Authority | 0.018 | 0.022 | 0.017 | 0.016 | 0.021 | 0.013 | 0.012 |
| Weather | 0.014 | 0.017 | 0.013 | 0.012 | 0.016 | 0.008 | 0.014 |
| Unit 1 Person 1 Type | 0.010 | 0.002 | 0.002 | 0.027 | 0.011 | 0.009 | 0.008 |
| Lighting | 0.010 | 0.011 | 0.009 | 0.008 | 0.012 | 0.005 | 0.008 |
| Unit 1 CV Placard | 0.007 | 0.011 | 0.008 | 0.004 | 0.009 | 0.006 | 0.002 |
| Census Area | 0.005 | 0.001 | 0.006 | 0.007 | 0.005 | 0.004 | 0.009 |
| City | 0.004 | 0.001 | 0.006 | 0.007 | 0.004 | 0.004 | 0.008 |
| *(… 24 further fields below 25th rank, all lower missingness)* | | | | | | | |

**Reading.** Class-conditional missingness differences flag potentially informative
missingness (a value's absence correlating with the outcome or with reporting practice);
year-conditional differences flag reporting drift. The low-missingness sensitivity
(paper §7.8) tests whether the primary contrast survives removing the flagged fields.

*Generated 2026-07-12T04:47:43Z by `crashsev/missingness_summary.py`; development rows only; aggregates only.*
