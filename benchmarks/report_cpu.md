# CPU FP-Growth benchmark

Date (UTC): 2026-10-28T16:15:20.082775+00:00
Platform: Linux-5.15.185-tegra-aarch64-with-glibc2.35
Baseline: origin/main

Five fresh-process measurements per implementation and case, after one warm-up. Each case's stdout was verified byte-for-byte before timing. Elapsed time includes startup, loading, preprocessing, mining, CPU rule counting, and formatting.

| Dataset | Implementation | Median elapsed (s) | Min–max (s) | Median peak RSS (MiB) |
|---|---|---:|---:|---:|
| iris | main | 0.0097 | 0.0090–0.0100 | 9.21 |
| iris | integer_ids | 0.0098 | 0.0085–0.0102 | 9.21 |
| bank | main | 39.3511 | 39.0530–39.5380 | 136.93 |
| bank | integer_ids | 38.2941 | 37.8467–39.4321 | 98.84 |
