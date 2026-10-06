# data/

* `panel.py` — in-memory dates × instruments panel with point-in-time membership.
* `synthetic.py` — deterministic GBM fixture panels with planted effects and data defects.
* `labels.py` — forward returns (kept separate from the expression engine, §6.3 layer 4).
* `qlib_bin.py` — reader/writer for Qlib's binary layout (no `pyqlib` required).
* `membership.py` — PIT membership from intervals, dated snapshots, and the fja05680 S&P 500 history.
* `validation.py` — cross-source mismatch (5 bp), coverage by year, Shumway delisting bound, §5.9 QA checklist.
* `manifest.py` / `MANIFEST.csv` — SHA-256, URL and access date of every downloaded file.
* `rebuild/` — CN (chenditc bins + BaoStock), US (yfinance + S&P 500 history + free-tier delisted fills),
  reference libraries (Ken French, OSAP, q-factors, JKP, CH-3/CH-4), optional JPX / crypto.
* `coverage/` — coverage, mismatch and QA reports produced by the rebuild scripts.
* `raw/`, `processed/` — local only (git-ignored).

```bash
python -m data rebuild cn --universe csi500
python -m data rebuild us
python -m data refs us
python -m data qa --panel data/processed/cn_csi500.npz --second data/processed/cn_csi500_baostock.npz
```
