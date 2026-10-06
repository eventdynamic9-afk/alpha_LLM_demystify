# Data licensing (§5.10) — fill in on the run date

Raw licensed data is **never committed**. The repository ships rebuild scripts (`data/rebuild/`),
file hashes (`data/MANIFEST.csv`), coverage reports (`data/coverage/`) and derived aggregates only
where the license allows.

| Source | License / terms | Redistribution of raw data | What we release |
|---|---|---|---|
| chenditc/investment_data | Repo Apache-2.0; vendor tables (Wind, Caihui) undocumented | No (vendor-derived parts) | Rebuild scripts + derived signals |
| BaoStock / AkShare / Tushare | Provider terms | Check | Scripts |
| Yahoo (yfinance) | Yahoo terms of service | No | Scripts |
| fja05680/sp500 | See repository | Check | Script + pointer |
| Tiingo / EODHD / FMP free tiers | Vendor terms (free tier) | No | Coverage audit numbers only, if terms allow |
| OSAP / Ken French / q-factors | Free academic | Usually yes with citation — check | Citation + download script |
| JKP | Non-commercial | Check | Script |
| Stambaugh CH-3/CH-4 | Free academic (Wind-derived) | Check | Citation + loader |
| Kaggle competition data (JPX, ...) | Competition rules | Usually no | Script + pointer |
| CRSP / Compustat / CSMAR / Wind `[PAID]` | Subscription | No | Code + derived aggregates only if license allows |

## Paid-grade data that is legitimately free (§5.8)

Use categories 1–6 (free academic releases derived from licensed data; community-merged datasets
with disclosure; use-in-place platforms such as QuantConnect / WorldQuant BRAIN for in-platform
performance re-checks; competition data within its rules; vendor free tiers and trials; institutional
access via your library). **Never** use category 7 — unauthorized re-uploads of CRSP / Compustat /
Wind / CSMAR files on Kaggle, GitHub, cloud drives or chat groups (license violation, unknown
provenance, cannot be cited or released, takedowns break reproducibility).

## Data tiers and the claims they support (§5.1)

| Claim type | Minimum acceptable source |
|---|---|
| C1 mechanistic, C6 identity | any clean OHLCV panel `[FREE]` |
| C2/C3 behavioral, originality | same-panel references `[FREE]`; OSAP/JKP/French for US return-level checks |
| C4 performance | survivorship-aware PIT universe + post-cutoff window `[PAID]` preferred; `[FREE]` with disclosed bias |

Every result table produced by `paper/make_tables.py` carries the data tier it was computed on.
