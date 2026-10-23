# CTO-Style Stock Ranking Dashboard

This is a local Streamlit dashboard for ranking technology, AI, semiconductor, cloud, cybersecurity, and infrastructure stocks using transparent technical signals.

It is informational only. It does not place trades, does not connect to a broker, and does not require or store API keys. The first version uses `yfinance` as its only market data source.

## Chatgptfolio Lab

The `Chatgptfolio Lab` dashboard view adds an auditable paper-trading workflow using OpenAI Responses API models. It is an experimental research tool, not investment advice and it never contacts a broker.

- Default decision model: `gpt-5.5`; selectable alternatives: `gpt-5.4`, `gpt-5.4-mini`.
- Set `OPENAI_API_KEY` in your environment before running a model analysis. On macOS, the app also automatically reads a Keychain entry named `chatgptfolio-openai` for the current user. The application never writes API keys to DuckDB, logs, snapshots, or source control.
- Alpha Vantage is the primary recent-news source when `ALPHA_VANTAGE_API_KEY` is available in the environment or macOS Keychain entry `chatgptfolio-alpha-vantage`. Each Lab analysis makes one bounded 48-hour multi-ticker request, ranks ticker-linked articles by relevance and sentiment strength, stores only public evidence URLs, and never stores the credential. GDELT supplies fallback discovery only for symbols without fresh Alpha Vantage coverage.
- Research packets, model configuration, raw structured responses, validation results, evidence, factor scores, and paper orders are persisted locally for audit.
- The research packet can enrich technical candidates with bounded, timestamped SEC/EDGAR filings and facts, FRED macro observations, 48-hour ticker news, and the latest advisory breadth snapshot for every fixed seed universe. The complete packet remains frozen for audit while a compact, de-duplicated view is sent to the model; missing or failed sources remain explicit.
- Model outputs express conviction and underwriting only. A deterministic allocator applies the holding, position, sector, turnover, and cash constraints before any paper order can be approved.
- Every approved run creates a separate $100,000 paper experiment. Experiments never share capital, positions, fills, NAV, or benchmark history; approvals are not rebalances of any existing experiment.
- Pending approvals fill independently at the first cached opening price after their frozen research date. The Paper Portfolio view compares normalized experiments, their available SPY/QQQ/GLD/BTC-USD/TECHNICAL baselines, and each experiment's own fill and audit trail.

### Weekly packet CLI

Prepare a reviewable packet from a CSV containing at least `symbol` and `score`:

```bash
python -m stockpicker.weekly --candidates data/candidates.csv --refresh-research --output /tmp/chatgptfolio-packet.json
```

To freeze and technically screen the current public S&P 500 constituent master for a research run, use `--sp500-master` instead of `--candidates` (the first run downloads the required local price cache).

The command never submits broker orders. Set `CHATGPTFOLIO_USER_AGENT` to a contact-bearing value before using SEC sources at scale.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
python -m compileall .
```

## Project Layout

- `app.py`: Streamlit dashboard.
- `stockpicker/data.py`: yfinance downloader plus DuckDB cache.
- `stockpicker/signals.py`: trend, moving-average, drawdown, extension, and Larsson-proxy calculations.
- `stockpicker/scoring.py`: 0-100 ranking model.
- `stockpicker/universe.py`: seed CSV loading and manual ticker parsing.
- `data/universe_*.csv`: editable seed universe presets with future fundamental/catalyst fields.
- `tests/`: focused pytest coverage.

## Data And Cache

The dashboard fetches 10 years of adjusted daily OHLCV by default:

```python
yfinance.download(..., auto_adjust=True)
```

Rows are cached locally in:

```text
data/market_data.duckdb
```

The cache has two tables:

- `prices`: adjusted daily OHLCV by `symbol` and `date`.
- `fetch_log`: fetch time, row counts, success/failure status, and messages.

Use the dashboard's refresh button to redownload the active universe and benchmark.

Example benchmark symbols include `QQQ`, `SMH`, `SOXX`, `SPY`, `^GSPC`, `^IXIC`, `^NDX`, and `^SOX`.

## Universe

The sidebar `Seed universe` selector chooses one active preset at a time. `SMH Universe` is the default. Selecting another preset automatically replaces the prior one, and `Manual only` starts with an empty ticker box.

Built-in presets:

- `SMH Universe`: semiconductor production, equipment, EDA, memory, and AI infrastructure.
- `Healthcare Universe`: healthcare services, managed care, medical devices, diagnostics, genomics, life science tools, and pharma.
- `GLP-1 and Obesity Universe`: GLP-1 leaders, obesity therapeutics, diabetes technology, metabolic diagnostics, and obesity care platforms.
- `Defence and Space Universe`: aerospace, defense primes, defense electronics, drones, launch, satellite communications, and space data.
- `Data Center Infrastructure Universe`: power and cooling, electrical infrastructure, data center networking, servers, REITs, storage, and AI infrastructure.
- `Power Grid and Electrification Universe`: power management, grid equipment, electrical services, backup power, grid automation, and power generation.
- `Utilities Universe`: electric utilities, multi-utilities, independent power producers, and regulated power.
- `Clean Energy Universe`: solar, wind, hydrogen, storage, renewable power, geothermal, and clean-energy finance.
- `Nuclear Energy Universe`: uranium, nuclear fuel, nuclear components, advanced reactors, and nuclear-exposed utilities.
- `Rare Earths and Strategic Materials Universe`: rare earths, lithium, copper, diversified mining, uranium, gold/copper, and steel.
- `Crypto Universe`: public crypto and blockchain equities, not crypto assets like `BTC-USD`.
- `Quantum Universe`: quantum pure plays, quantum security, quantum platforms, and enabling semiconductor/software suppliers.
- `Cybersecurity Universe`: security platforms, endpoint, identity, zero trust, vulnerability management, data resilience, and cyber services.
- `Magnificent Seven Universe`: Apple, Microsoft, Alphabet, Amazon, NVIDIA, Meta, and Tesla.
- `Space Exploration Universe`: listed launch, lunar-services, space-infrastructure, Earth-observation, satellite-communications, and mission-systems companies. SpaceX is not included because it is private.

The ticker text area is prefilled from the selected preset and is the active watchlist. Removing a ticker removes it from the current run; adding an unknown ticker marks it as `Manual` while preserving seed metadata for preset tickers.

Tickers can be separated by commas, spaces, or newlines.

The CSV includes:

- `fundamental_score`
- `catalyst_score`
- `notes`

In v1, fundamental and catalyst weights are `0`. They are placeholders for a later scoring pass and do not affect rankings yet.

## Larsson Line Regime Approximation

This dashboard uses an inspectable Larsson Line proxy. It is not an official CTO Larsson formula.

For each stock and the benchmark:

```text
larsson_line_proxy = max(SMA_200D, forward_filled_SMA_200W)
```

Regime states:

- `Bull`: latest price is above the proxy and the proxy's 20-trading-day slope is non-negative.
- `Bear`: latest price is below the proxy and the proxy's 20-trading-day slope is negative.
- `Transition`: every other case, including insufficient data.

`QQQ` is the default benchmark. The benchmark ticker is editable in the sidebar.

Changing the benchmark updates the benchmark metrics, benchmark tab, raw signal table, and relative-strength columns. It changes the ranking order only when `Use relative strength in ranking` is enabled.

## Larsson Regime History

The rankings table includes a `Regime History` histogram next to the current Larsson regime.

Each bar is one weekly observation from the same Larsson-proxy approximation:

- High bar: `Bull`
- Mid bar: `Transition`
- Low bar: `Bear`

The default window is 26 weeks. Change `Larsson history weeks` in the sidebar to view 8 to 104 weeks. The raw comma-separated regime labels are retained in the data audit table for inspection.

## Multi-Timeframe Trend State

Each symbol gets daily, weekly, and monthly trend states:

- Daily: latest price vs `SMA_50D` plus 20-trading-day `SMA_50D` slope.
- Weekly: latest weekly close vs `SMA_30W` plus 10-week `SMA_30W` slope.
- Monthly: latest monthly close vs `SMA_10M` plus 4-month `SMA_10M` slope.

States are:

- `Up`: price above the average and average slope is non-negative.
- `Down`: price below the average and average slope is negative.
- `Mixed`: everything else or insufficient data.

## 200D And 200W Filters

The dashboard computes:

- `pass_200d`: latest adjusted close is above `SMA_200D`.
- `pass_200w`: latest weekly close is above `SMA_200W`.

If a symbol has less than 200 weeks of history, `pass_200w` displays as `n/a` and receives neutral half-credit in the ranking score. Insufficient 200D history is displayed in the data warnings and receives no 200D filter credit.

Sidebar toggles can require a 200D pass and/or 200W pass in the displayed ranking table.

## Drawdown And Extension Metrics

The signal table includes:

- 52-week drawdown from the highest close in the last 252 trading days.
- Fetched-history drawdown from the highest close in the cached history.
- Extension vs `SMA_50D`.
- Extension vs `SMA_200D`.
- Extension vs `SMA_30W`.

Extensions are calculated as:

```text
latest_price / moving_average - 1
```

Drawdown is calculated as:

```text
latest_price / lookback_peak - 1
```

## Ranking Model

The base v1 score is 0-100:

- 15 points: Larsson-proxy regime.
- 15 points: daily trend.
- 20 points: weekly trend.
- 15 points: monthly trend.
- 15 points: 200D filter.
- 10 points: 200W filter.
- 10 points: drawdown/extension discipline.
- 0 points: fundamental score placeholder.
- 0 points: catalyst score placeholder.

Credit mapping:

- Larsson: `Bull` = full, `Transition` = half, `Bear` = zero.
- Trend: `Up` = full, `Mixed` = half, `Down` = zero.
- 200D: pass = full, fail or unavailable = zero.
- 200W: pass = full, fail = zero, unavailable = half.
- Drawdown/extension: starts at full credit and deducts for severe drawdowns, excessive extension, or trading below the 200D trend.

## Benchmark Relative Strength

The dashboard compares each stock with the selected benchmark over aligned trading dates.

Relative-strength spreads are:

```text
stock total return - benchmark total return
```

The table shows:

- `RS 3M`
- `RS 6M`
- `RS 12M`
- `RS Score`
- `RS Label`

The `RS Score` is a 0-100 blend of 3M, 6M, and 12M return spreads. Missing history receives neutral credit.

By default, ranking uses the base `score`. If `Use relative strength in ranking` is enabled, the table sorts by:

```text
benchmark_adjusted_score = 0.85 x base score + 0.15 x RS Score
```

The base score remains visible for auditability.

## Relative Rotation Graph

The `RRG` tab is an inspectable Relative Rotation Graph-style approximation. It is not a proprietary Strike or JdK implementation.

The chart compares selected stocks against the current benchmark using weekly relative movement:

```text
relative_line = stock weekly close / benchmark weekly close
rs_ratio = 100 x relative_line / SMA(relative_line)
rs_momentum = 100 x rs_ratio / SMA(rs_ratio)
```

How to read it:

- Right = stronger relative trend versus the benchmark.
- Up = improving relative momentum.
- Tail direction = recent rotation path.
- Top-right `Leading`: stronger and still improving.
- Bottom-right `Weakening`: stronger but losing momentum.
- Bottom-left `Lagging`: weaker and still weakening.
- Top-left `Improving`: weaker but momentum is recovering.

How to use stock-level RRG:

- Pick a benchmark such as `QQQ`, `SMH`, `SPY`, or `^SOX`.
- Open `RRG`.
- Keep `RRG view` set to `Stocks`.
- Select built-in seed themes such as `AI accelerators`, `Semiconductors`, or `Cloud and AI platforms`.
- Start with the default top 20 ranked symbols.
- Look for stocks rotating into `Improving` or `Leading`.

How to use seed-universe RRG:

- Open `RRG` and set `RRG view` to `Seed universes`.
- Select multiple built-in seed universes to compare on one chart.
- Each seed dot is an equal-weight normalized basket of its seed tickers.
- The seed basket is compared against the current benchmark with the same RRG-style formula used for individual stocks.
- Hover a seed dot to see the quadrant, RS Ratio, RS Momentum, symbols used, and missing symbols.

RRG is visual context only in v1. It does not change the ranking score.

## Portfolio Allocator

The `Portfolio` tab builds an informational model portfolio from selected seed universes. It does not place trades and does not connect to a broker.

The allocator is deterministic and inspectable:

- It builds candidates from selected seed universes.
- It prefers stocks passing the 200D filter and excludes `Bear` Larsson-proxy regimes by default.
- It blends the ranking score with seed-level RRG pressure.
- It sizes concentrated winners with caps for single positions and seed-universe exposure.
- It can reserve cash based on benchmark regime: `Bull` = 0%, `Transition` = 15%, `Bear` = 35%.

Sensitivity controls decide how reactive the system is:

- `Conservative`: lower RRG influence, larger change threshold, lower turnover.
- `Balanced`: default middle setting.
- `Aggressive`: higher RRG influence, smaller change threshold, higher turnover.

The tab shows:

- applied allocation after sensitivity rules
- raw target allocation
- changes since the last saved baseline
- suppressed changes
- seed-level RRG strategy signals
- candidate pool audit
- an LLM prompt preview

Portfolio seed-universe price data is loaded on demand from the `Portfolio` tab. This keeps normal dashboard startup focused on the active rankings universe and benchmark.

The LLM prompt is generated locally for review or future integration. The app does not call any LLM API and does not require API keys.

Portfolio snapshots are saved locally in DuckDB when `Save current allocation as baseline` is clicked. Baselines are matched by benchmark, selected seed universes, sensitivity, and allocator config hash before they are used for change tracking. The saved baseline is used to track added, removed, increased, reduced, and suppressed model-allocation changes.

## Benchmark Seed Overlay

The `Benchmark` tab overlays the active seed universe against the selected benchmark on the daily, weekly, and drawdown charts.

The yellow dashed `Seed EW Basket` line is an equal-weighted average of the active ticker text area. For daily and weekly price charts, each ticker is normalized at the common start date before averaging, then scaled to the benchmark price so the comparison is visually fair. The yellow dashed drawdown line is calculated from the equal-weighted basket index.

This overlay is visual context only. It does not change scores, rankings, RRG, benchmark selection, or cache behavior.

## Dashboard Tabs

- `Rankings`: ranked table with current regime, regime-history histogram, trend states, filters, drawdown, extension, and data warnings.
- `Detail`: selected ticker charts for daily trend, weekly trend, and drawdown.
- `Benchmark`: selected benchmark metrics, regime history, price trend, weekly trend, and drawdown.
- `RRG`: relative rotation view filtered by built-in seed themes.
- `Portfolio`: model portfolio allocation, sensitivity controls, change tracking, and LLM prompt preview.
- `Data Audit`: fetch log and raw signal table for inspection.

Table headers in the rankings, RRG, and portfolio tables include hover explanations for the score, trend, relative-strength, filter, drawdown, extension, allocation, and warning columns.

## Constraints

- No brokerage execution.
- No API keys.
- No hidden paid data.
- Simple, readable Python is preferred over complex abstractions.
- All scoring math is local and inspectable.
