# TradingTransformer — How It Actually Works

A bottom-up walkthrough of every file, following the exact order data moves through the system. By the end you should be able to trace any prediction back to the raw CSV line it came from.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Stage 1 — Feature Engineering](#2-stage-1--feature-engineering)
   - [pipeline.py — the bar-by-bar adapter](#pipelinepy)
   - [technicalIndicators.py — every indicator](#technicalindicatorspy)
3. [Stage 2 — Building the Transformer](#3-stage-2--building-the-transformer)
   - [Embedding.py — turning numbers into vectors](#embeddingpy)
   - [Mask.py — hiding the future](#maskpy)
   - [Attention.py — the core mechanism](#attentionpy)
   - [EncodingnDecoding.py — stacking layers](#encodingdecodingpy)
   - [Model.py — wiring it all together](#modelpy)
4. [Stage 3 — Training](#4-stage-3--training)
   - [DataFrame.py — making sliding windows](#dataframepy)
   - [Tools.py — early stopping](#toolspy)
   - [Interface.py — the training loop](#interfacepy)
   - [Train.py — the entry point](#trainpy)
5. [Stage 4 — Exporting to C++](#5-stage-4--exporting-to-c)
   - [exportModel.py — TorchScript + scalers](#exportmodelpy)
   - [convert_scalers.py — merging the scaler files](#convert_scalerspy)
6. [Stage 5 — The C++ Backtester](#6-stage-5--the-c-backtester)
   - [The event system](#the-event-system)
   - [BacktestConfig.hpp — loading the YAML](#backtestconfighpp)
   - [FeatureCSVDataHandler — reading the data](#featurecsvdatahandler)
   - [MultiAssetDataHandler — multiple symbols](#multiassetdatahandler)
   - [ScalerParams.hpp — normalising in C++](#scalerparamshpp)
   - [MLStrategy — inference in C++](#mlstrategy)
   - [BacktestEngine — the event loop](#backtestengine)
   - [Portfolio — sizing and tracking](#portfolio)
   - [RiskManager — the gate](#riskmanager)
   - [SimulatedExecution — realistic fills](#simulatedexecution)
   - [PerformanceMetrics — final numbers](#performancemetrics)
   - [ml_main.cpp — wiring everything](#ml_maincpp)
7. [run_pipeline.py — the orchestrator](#7-run_pipelinepy--the-orchestrator)
8. [Data Flow Cheat Sheet](#8-data-flow-cheat-sheet)

---

## 1. The Big Picture

The system does one thing: take raw daily stock prices, train a transformer to predict next-day close, then simulate trading with that model and measure how it performed.

It is split into two completely independent layers:

- **Python** — everything to do with data, features, training, and exporting the model
- **C++** — everything to do with simulating trades, running the model, and computing results

The handoff between them is three files that get written by Python and read by C++:

```
transformer.pt          ← the trained model as a portable binary
feature_scaler.csv      ← the normalisation parameters for the 33 input features
target_scaler.csv       ← the normalisation parameters for the close price target
```

These live in `models/`. Once they exist, the C++ engine runs completely independently of Python.

---

## 2. Stage 1 — Feature Engineering

### `pipeline.py`

**Location:** `research/features/pipeline.py`

This file takes a raw OHLCV CSV and produces a feature CSV with 34 columns. You run it like:

```bash
python pipeline.py data/AAPL.csv -o features/
```

The raw CSV has columns `timestamp, open, high, low, close, volume`. The output CSV adds 33 computed features and ends with `close` as the prediction target.

**The key design decision: bar-by-bar processing with an adapter.**

The obvious way to compute features is with pandas vectorised operations — fast and clean. The problem is the indicator functions in `technicalIndicators.py` were written for backtrader's API, which accesses data like `self.data.close[0]` (current bar) and `self.data.close[-1]` (previous bar). If you rewrote them in pandas, you'd have two separate implementations that could silently drift apart. The model would be trained on slightly different numbers than the ones it receives during inference.

Instead, `pipeline.py` wraps the DataFrame in three adapter classes that make it look like backtrader's data structure, then calls `technicalIndicators.py` unchanged:

```python
class _Line:
    """Wraps a numpy array and exposes backtrader-style indexing."""
    def __init__(self, values, idx):
        self._v = values
        self._i = idx

    def __getitem__(self, offset):
        return float(self._v[self._i + offset])   # [0] = current, [-1] = prev bar

    def get(self, size):
        start = max(0, self._i - size + 1)
        return list(self._v[start: self._i + 1])  # last N values


class _DataView:
    """Bundles all the _Lines for a single bar's context."""
    def __init__(self, df, idx):
        self.close  = _Line(df["close"].values,  idx)
        self.open   = _Line(df["open"].values,   idx)
        self.high   = _Line(df["high"].values,   idx)
        self.low    = _Line(df["low"].values,    idx)
        self.volume = _Line(df["volume"].values, idx)


class _BarCtx:
    """The stateful context advanced bar by bar.
    Indicators that need running state (EMA, OBV) store it here via setattr."""
    def __init__(self, df):
        self._df = df
        self.data = None

    def advance(self, idx):
        self.data = _DataView(self._df, idx)
```

Then the core loop is simply:

```python
ctx = _BarCtx(df)
rows = []
for i in range(len(df)):
    ctx.advance(i)                          # move to bar i
    rsi  = calculateRsi(ctx)               # same function used in training
    macd = calculateMacd(ctx)
    # ... all 33 features ...
    rows.append({...})
```

The same `_BarCtx` instance is reused across every bar, which is what allows stateful indicators (EMA, OBV) to accumulate their running values correctly — they store state on `self` via `setattr`, and since `ctx` is the same object, that state persists bar to bar.

**Pivot points** are the one exception — they are computed inline in `pipeline.py` rather than in `technicalIndicators.py`, because they are deterministic (no state needed):

```python
def _pivot_points(high, low, close):
    P = (high + low + close) / 3.0
    return {
        "P": P,
        "R1": 2.0 * P - low,
        "R2": P + (high - low),
        "R3": high + 2.0 * (P - low),
        "S1": 2.0 * P - high,
        "S2": P - (high - low),
        "S3": low - 2.0 * (high - P),
    }
```

**Output column order matters.** The final output DataFrame's columns are defined explicitly in the `row = {...}` dictionary to guarantee they always come out in the same order: `timestamp, high, low, volume, adj close, P, R1, R2, R3, S1, S2, S3, obv, ...close`. The C++ engine reads them by position, so the order must match exactly what `exportModel.py::load_args()` expects.

---

### `technicalIndicators.py`

**Location:** `research/features/technicalIndicators.py`

This is the single source of all indicator logic. Both training (via the `_BarCtx` adapter) and inference use these exact functions. Here's how each one works:

**`calculateRsi(ctx)`** — Relative Strength Index (14-period in the original, 15 called from pipeline)

RSI measures how fast price is moving. It compares average gains to average losses over a lookback window:
```
RS = average_gain / average_loss
RSI = 100 - (100 / (1 + RS))
```
A value above 70 is overbought, below 30 is oversold. The function uses `ctx.data.close.get(size)` to retrieve the last N close prices.

**`calculateMacd(ctx)`** — MACD (12/26 EMA crossover)

MACD = EMA(12) - EMA(26). When the shorter EMA crosses above the longer, momentum is building. The function is stateful — it stores the running EMA values on `ctx` using `getattr/setattr`, which is why the same `ctx` instance must persist across bars.

**`calculateEMA(ctx, period)`** — Exponential Moving Average

```
EMA[0] = close
EMA[t] = close * k + EMA[t-1] * (1 - k)    where k = 2 / (period + 1)
```

Unlike SMA which weights all bars equally, EMA gives more weight to recent prices. The multiplier `k` shrinks older values exponentially. The first call uses close as the seed; subsequent calls update the running value stored on `ctx`.

**`calculateOBV(ctx)`** — On-Balance Volume

OBV accumulates volume directionally: add today's volume if price went up, subtract if it went down. It gives a sense of whether institutional money is flowing in or out without being visible in the price alone.

**`calculateVolatility(ctx)`** — 20-bar rolling standard deviation of returns

```
returns[i] = (close[i] - close[i-1]) / close[i-1]
volatility = std(returns[-20:])
```

**`calculateStochastic(ctx)`** — Stochastic Oscillator

```
K = (close - low_14) / (high_14 - low_14) * 100
D = SMA(K, 3)
```

Where `high_14` and `low_14` are the highest high and lowest low over the last 14 bars. It measures where the current close is relative to the recent range — if close is at the top of the range, K is near 100.

**`calculateStochRSI(ctx, rsi)`** — Stochastic applied to RSI

Same stochastic formula but applied to the RSI series instead of price. This gives a faster oscillator that oscillates more often.

**`calculateATR(ctx)`** — Average True Range

True Range = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = EMA(True Range, 14)

This is a pure volatility measure — how much the price is actually moving per bar, including gaps. Used for position sizing in many real strategies.

---

## 3. Stage 2 — Building the Transformer

The transformer is built up from scratch across five files. Each one adds one layer of abstraction. Here's how they stack.

### `Embedding.py`

**Location:** `research/transformer/Embedding.py`

The model can't work with raw numbers — it needs fixed-length vectors. This file turns each bar's 34 features into a 256-dimensional embedding. Three components are summed:

**1. TokenEmbedding — the feature values**

```python
self.tokenConv = nn.Conv1d(in_channels=cIn, out_channels=dModel,
                           kernel_size=3, padding=1, padding_mode='circular')
```

This is a 1D convolution with kernel size 3. It takes the 34 features at each position and mixes them into a 256-dimensional vector. The `circular` padding means the edges wrap around — this prevents edge artefacts without introducing zeros. The weights are initialised with Kaiming (He) initialisation, which is designed for activations that follow LeakyReLU — it keeps gradients in a healthy range at the start of training.

The `.permute(0, 2, 1)` before the conv switches from shape `(batch, time, features)` to `(batch, features, time)` because Conv1d expects channels first. `.transpose(1, 2)` switches back after.

**2. PositionalEmbedding — where in time**

```python
pe[:, 0::2] = torch.sin(position * divTerm)   # even dims
pe[:, 1::2] = torch.cos(position * divTerm)   # odd dims
```

The transformer has no inherent notion of order — attention treats the sequence as a set. Without positional encoding, shuffling bar 1 and bar 15 would produce the same output. The sinusoidal encoding injects position information by encoding each position with a unique pattern of sins and cosines at different frequencies.

This is stored as a buffer (`self.register_buffer('pe', pe)`) not a parameter — it's fixed and never updated by backprop. The `.require_grad = False` on line 9 explicitly prevents gradient computation.

**3. TimeFeatureEmbedding — what time of year**

```python
self.embed = nn.Linear(3, dModel, bias=False)
```

A learned linear projection from the 3 time features (month, day, weekday) into 256 dimensions. Unlike the sinusoidal encoding which is fixed, these weights are learned — the model can learn that "January is different from July" during training.

**DataEmbedding** combines all three:

```python
def forward(self, x, xMark):
    x = self.valueEmbedding(x) + self.temporalEmbedding(xMark) + self.positionEmbedding(x)
    return self.dropout(x)
```

Shape going in: `(batch, 30, 34)`. Shape coming out: `(batch, 30, 256)`.

---

### `Mask.py`

**Location:** `research/transformer/Mask.py`

```python
self._mask = torch.triu(torch.ones(maskShape, dtype=torch.bool), diagonal=1)
```

This creates an upper-triangular matrix of `True` values (everything above the diagonal). It's used in the decoder's self-attention to prevent position `t` from attending to positions `t+1, t+2, ...` (the future). When this mask is applied, those attention scores are set to `-inf`, which softmax turns to 0 — so those positions are completely ignored.

The encoder does **not** use this mask — it can see the full 30-bar context in both directions.

---

### `Attention.py`

**Location:** `research/transformer/Attention.py`

Two classes here: `FullAttention` (the core computation) and `AttentionLayer` (the multi-head wrapper).

**FullAttention:**

```python
scores = torch.einsum("blhe,bshe->bhls", queries, keys)
```

This computes the dot product between every query and every key. The einsum notation: `b`=batch, `l`=query sequence length, `h`=head, `e`=head dimension, `s`=key sequence length. The output `bhls` is a `(batch, heads, query_len, key_len)` matrix of scores.

```python
if self.maskFlag:
    scores = scores.masked_fill(mask, float("-inf"))
```

The causal mask is applied here. Positions marked `True` in the mask get `-inf` before softmax, so their attention weight becomes 0.

```python
scale = 1. / sqrt(E)
A = self.dropout(torch.softmax(scale * scores, dim=-1))
V = torch.einsum("bhls,bshd->blhd", A, values)
```

`scale = 1/sqrt(d_k)` prevents the dot products from growing too large as dimension increases (which would saturate softmax). The final einsum takes the weighted sum of values according to the attention weights.

**AttentionLayer:**

```python
self.queryProjection  = nn.Linear(dModel, dKeys * nHeads)
self.keyProjection    = nn.Linear(dModel, dKeys * nHeads)
self.valueProjection  = nn.Linear(dModel, dValues * nHeads)
self.outProjection    = nn.Linear(dValues * nHeads, dModel)
```

With `dModel=256` and `nHeads=8`: `dKeys = dValues = 256/8 = 32`. So four linear projections:
- Input `(batch, T, 256)` → Q/K/V `(batch, T, 256)` → reshape to `(batch, T, 8, 32)` 
- After attention: `(batch, T, 8, 32)` → flatten to `(batch, T, 256)` → output projection back to `(batch, T, 256)`

The 8 heads each learn to attend to different things simultaneously — one head might focus on short-term momentum, another on volume spikes, etc.

---

### `EncodingnDecoding.py`

**Location:** `research/transformer/EncodingnDecoding.py`

**EncoderLayer:**

Each encoder layer does two things:

```python
def forward(self, x, attnMask=None):
    # 1. Self-attention sub-layer
    newX, attn = self.attention(x, x, x, attnMask=attnMask)  # Q=K=V=x
    x = x + self.dropout(newX)                                 # residual connection

    # 2. Feed-forward sub-layer
    y = x = self.norm1(x)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # expand to 512
    y = self.dropout(self.conv2(y).transpose(-1, 1))                   # compress back to 256
    return self.norm2(x + y), attn                             # residual connection
```

Notice that in `self.attention(x, x, x, ...)`, queries, keys, and values are all the same — this is **self-attention**. Each bar attends to all other bars in the sequence.

The feed-forward network uses `Conv1d` with `kernel_size=1` instead of `nn.Linear`. These are mathematically identical (a 1D conv with kernel 1 is just a linear transformation applied independently to each time step) but Conv1d is often faster in practice due to memory layout.

**GELU activation:** `F.gelu` is used instead of ReLU. GELU is smooth at 0 (unlike ReLU's hard kink), which helps gradients flow more cleanly.

**Residual connections** (`x = x + ...`) are the reason deep transformers can be trained at all. Without them, gradients vanish through deep layers. The residual path gives gradients a direct route back to early layers.

**LayerNorm** normalises each position's embedding independently. Unlike BatchNorm (which normalises across the batch), LayerNorm works per-sample — important for variable-length sequences and small batch sizes.

**Encoder** — just a list of layers:
```python
for attnLayer in self.attnLayers:
    x, attn = attnLayer(x, attnMask=attnMask)
if self.norm is not None:
    x = self.norm(x)   # final LayerNorm after all layers
```

**DecoderLayer** — three sub-layers instead of two:

```python
def forward(self, x, cross, xMask=None, crossMask=None):
    # 1. Masked self-attention (can't look at future positions)
    x = x + self.dropout(self.selfAttention(x, x, x, attnMask=xMask)[0])
    x = self.norm1(x)

    # 2. Cross-attention (Q from decoder, K/V from encoder output)
    x = x + self.dropout(self.crossAttention(x, cross, cross, attnMask=crossMask)[0])

    # 3. Feed-forward
    y = x = self.norm2(x)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))
    return self.norm3(x + y)
```

The critical step is **cross-attention**: `self.crossAttention(x, cross, cross, ...)`. Here `x` is the decoder's current state (queries) and `cross` is the encoder's output (keys and values). This is how the decoder reads from the encoder — it asks "which parts of the encoded historical context are relevant to predicting this future position?"

---

### `Model.py`

**Location:** `research/transformer/Model.py`

This is where all the pieces connect. The constructor assembles the full model from config:

```python
self.encEmbedding = DataEmbedding(configs.encIn, configs.dModel, configs.dropout)
self.decEmbedding = DataEmbedding(configs.decIn, configs.dModel, configs.dropout)

self.encoder = Encoder([
    EncoderLayer(
        AttentionLayer(
            FullAttention(False, ...),   # maskFlag=False — encoder sees full context
            configs.dModel, configs.nHeads
        ),
        configs.dModel, configs.dFf, dropout=configs.dropout,
    ) for _ in range(configs.eLayers)   # 3 layers
], normLayer=torch.nn.LayerNorm(configs.dModel))

self.decoder = Decoder([
    DecoderLayer(
        AttentionLayer(FullAttention(True, ...),  ...),   # self-attn: maskFlag=True (causal)
        AttentionLayer(FullAttention(False, ...), ...),   # cross-attn: maskFlag=False
        configs.dModel, configs.dFf, dropout=configs.dropout,
    ) for _ in range(configs.dLayers)   # 2 layers
], normLayer=torch.nn.LayerNorm(configs.dModel),
   projection=nn.Linear(configs.dModel, configs.cOut, bias=True))   # 256 → 1
```

The `False`/`True` `maskFlag` arguments are the key difference between encoder and decoder attention. Encoder layers use `False` — no mask, attend freely. Decoder self-attention uses `True` — causal mask applied.

The **forward pass:**

```python
def forward(self, xEnc, xMarkEnc, xDec, xMarkDec, ...):
    encOut = self.encEmbedding(xEnc, xMarkEnc)    # (1, 30, 34) → (1, 30, 256)
    encOut, attns = self.encoder(encOut)            # (1, 30, 256) → (1, 30, 256)

    decOut = self.decEmbedding(xDec, xMarkDec)    # (1, 11, 34) → (1, 11, 256)
    decOut = self.decoder(decOut, encOut)           # → (1, 11, 1) after projection

    return decOut[:, -self.predLen:, :], attns     # take only the last predLen positions
```

`xDec` is constructed during training as the last `labelLen` known bars concatenated with `predLen` zeros:
```python
decInp = torch.zeros_like(batchY[:, -predLen:, :])
decInp = torch.cat([batchY[:, :labelLen, :], decInp], dim=1)
```

The decoder sees the 10 most recent real bars, then has `predLen=5` zero-filled slots it needs to fill in. It uses those real bars as context and the encoder's compressed memory of the 30-bar history to predict the future positions. Only the last `predLen` positions of the output are used.

---

## 4. Stage 3 — Training

### `DataFrame.py`

**Location:** `research/transformer/DataFrame.py`

`DataFrameDataset` is a PyTorch `Dataset` — it tells the `DataLoader` how many samples exist and how to fetch one.

**Data split** happens in `Interface.py::splitData()`:
```python
totalLen = len(df)
trainEnd = int(totalLen * 0.7)
valEnd   = trainEnd + int(totalLen * 0.15)
trainDf  = df.iloc[:trainEnd]
valDf    = df.iloc[trainEnd:valEnd]
testDf   = df.iloc[valEnd:]
```

**Scaler fitting:** The `DataFrameDataset` for `flag='train'` creates and fits the scalers itself:
```python
self.featureScaler = StandardScaler()
self.targetScaler  = StandardScaler()
self.dataXFeatures = self.featureScaler.fit_transform(df[auxilFeatures].values)
self.dataXTarget   = self.targetScaler.fit_transform(df[[target]].values)
```

For `flag='val'` or `'test'`, the scalers must be passed in — they are not refitted. This is the critical data hygiene rule: the scaler learns the mean and std from training data only. Applying training statistics to validation/test data is correct — you can only know the training statistics in a real deployment anyway.

**Window generation** (`__getitem__`):
```python
sBegin = index
sEnd   = sBegin + self.seqLen          # encoder input: bars [sBegin, sEnd)

rBegin = sEnd - self.labelLen
rEnd   = rBegin + self.labelLen + self.predLen  # decoder input

seqX     = self.dataX[sBegin:sEnd]       # (30, 34) — encoder input
seqY     = self.dataY[rBegin:rEnd]       # (11, 34) — decoder input (labelLen + predLen)
seqXMark = self.dataStamp[sBegin:sEnd]   # (30, 3)  — time features for encoder
seqYMark = self.dataStamp[rBegin:rEnd]   # (11, 3)  — time features for decoder
```

**Time features** (`_processTimeFeatures`):
```python
timeSteps['month']   = timeSteps.index.month    # 1–12
timeSteps['day']     = timeSteps.index.day      # 1–31
timeSteps['weekday'] = timeSteps.index.weekday  # 0=Monday, 4=Friday
```

These are the raw integer values. A more sophisticated version (`_processTimeFeaturesLarge`, which exists but is not used) would use cyclical encoding (`sin/cos`) so December wraps smoothly back to January.

---

### `Tools.py`

**Location:** `research/transformer/Tools.py`

**EarlyStopping:**

```python
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.earlyStop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss    # lower loss = higher score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
```

Whenever validation loss improves, the checkpoint is saved and the counter resets. Whenever it doesn't improve, the counter increments. After `patience` consecutive non-improvements, `earlyStop` is set to `True` and training stops. The best weights (not the final weights) are loaded at the end.

**adjustLearningRate:**

```python
def adjustLearningRate(optimizer, epoch, args):
    lr_adjust = {epoch: args.learningRate * (0.5 ** ((epoch - 1) // 1))}
```

This halves the learning rate every epoch. In practice with `patience=10`, training almost always stops before the LR gets very small — but the decay prevents large steps late in training from overshooting a minimum.

---

### `Interface.py`

**Location:** `research/transformer/Interface.py`

`Model_Interface` wraps the training lifecycle. The train loop is at line 160:

```python
for epoch in range(self.args.trainEpochs):
    self.model.train()    # enables dropout

    for i, (batchX, batchY, batchXMark, batchYMark) in enumerate(trainLoader):
        modelOptim.zero_grad()

        # Build decoder input: real labelLen bars + zeros for predLen slots
        decInp = torch.zeros_like(batchY[:, -self.args.predLen:, :])
        decInp = torch.cat([batchY[:, :self.args.labelLen, :], decInp], dim=1)

        # Forward pass
        outputs = self.model(batchX, batchXMark, decInp, batchYMark)[0]

        # Only evaluate the predLen output positions, last feature (close)
        fDim = -1
        outputs = outputs[:, -self.args.predLen:, fDim:]   # (batch, 5, 1)
        batchY  = batchY[:, -self.args.predLen:, fDim:]    # (batch, 5, 1)

        loss = criterion(outputs, batchY)   # MSE
        loss.backward()
        modelOptim.step()
```

`fDim = -1` selects the last feature dimension — which is `close` since it's always the last column. The model outputs `(batch, predLen, 1)` — one predicted close per future step. Even though `predLen=5`, only the first predicted step is used during inference (`output[0][0][0]` in `MLStrategy.cpp`).

After each epoch, `self.vali()` is called on validation and test sets with `model.eval()` and `torch.no_grad()` — disabling dropout and gradient computation. The validation MSE is passed to `EarlyStopping`. If it doesn't improve for 10 epochs, training stops.

Scalers are saved immediately after fitting the training dataset:
```python
joblib.dump(trainData.featureScaler, os.path.join(path, 'featureScaler.pkl'))
joblib.dump(trainData.targetScaler,  os.path.join(path, 'targetScaler.pkl'))
```

---

### `Train.py`

**Location:** `research/training/Train.py`

This is the entry point that configures all the hyperparameters and calls `Model_Interface`. It:
1. Loads feature CSVs from `BacktestingData/`
2. Merges all symbols into one DataFrame with a `ticker` column
3. Creates `Model_Interface(args)` and calls `.train(data)`

The hyperparameters are hardcoded here and passed as a `Namespace` object — `seqLen=30`, `labelLen=10`, `predLen=5`, `dModel=256`, `nHeads=8`, `eLayers=3`, `dLayers=2`, `dFf=512`, `dropout=0.1`, `batchSize=128`, `trainEpochs=100`, `patience=10`, `learningRate=0.0005`.

---

## 5. Stage 4 — Exporting to C++

### `exportModel.py`

**Location:** `research/exportModel.py`

After training, the checkpoint at `models/Model3.pth` (or `checkpoint.pth`) is a Python object — C++ can't load it directly. This file converts it to a format C++ can use: TorchScript.

**TransformerInferenceWrapper:**

The original `Model.forward()` takes four arguments: `xEnc, xMarkEnc, xDec, xMarkDec`. The C++ side would need to construct `xDec` and `xMarkDec` (zeros) on every call. This wrapper hides that:

```python
class TransformerInferenceWrapper(nn.Module):
    def forward(self, xEnc: torch.Tensor, xMarkEnc: torch.Tensor) -> torch.Tensor:
        batch    = xEnc.shape[0]
        features = xEnc.shape[2]

        # Build the zero-filled decoder inputs internally
        xDec     = torch.zeros(batch, self.labelLen + self.predLen, features)
        xMarkDec = torch.zeros(batch, self.labelLen + self.predLen, 3)

        output, _ = self.model(xEnc, xMarkEnc, xDec, xMarkDec)
        return output
```

Now C++ only needs to call `forward(xEnc, xMarkEnc)` — two tensors, not four.

**TorchScript tracing:**

```python
dummy_xEnc     = torch.zeros(1, args.seqLen, num_features)      # (1, 30, 34)
dummy_xMarkEnc = torch.zeros(1, args.seqLen, 3)                  # (1, 30, 3)

with torch.no_grad():
    traced_model = torch.jit.trace(wrapper, (dummy_xEnc, dummy_xMarkEnc))

traced_model.save("models/transformer.pt")
```

`torch.jit.trace` runs the model once with the dummy inputs and records every operation. The result is a self-contained graph that can be loaded in C++ with no Python runtime. The shapes used during tracing are baked in — the C++ model always expects `(1, 30, 34)` and `(1, 30, 3)`.

**Scaler export (`save_scaler_params`):**

```python
with open(feature_path, "w") as f:
    writer.writerow(["feature", "mean", "scale"])
    for name, mean, scale in zip(
        dataset.auxilFeatures,
        dataset.featureScaler.mean_,
        dataset.featureScaler.scale_,
    ):
        writer.writerow([name, mean, scale])
```

The `StandardScaler` formula is `x_scaled = (x - mean_) / scale_`. Exporting `mean_` and `scale_` as CSV lets the C++ `ScalerParams` struct apply the identical transform without importing sklearn.

---

### `convert_scalers.py`

**Location:** `scripts/convert_scalers.py`

Training saves two separate pickle files: `featureScaler.pkl` (33 features) and `targetScaler.pkl` (1 feature: close). The C++ `MLStrategy` needs them merged into one file so it can normalise the full 34-element feature vector in one pass.

This script loads both pickles, extracts `mean_` and `scale_` from each, and writes them in the exact column order that the C++ `MODEL_FEATURE_COLUMNS` list expects.

---

## 6. Stage 5 — The C++ Backtester

### The Event System

**Location:** `backtester/include/events/`

Everything in the C++ engine communicates through events. Rather than components calling each other's methods directly, they push events into a shared queue and the engine dispatches them.

The base type is defined in `Events.hpp`:
```cpp
enum class EventType { MARKET, SIGNAL, ORDER, FILL };

struct Event {
    virtual ~Event() = default;
    virtual EventType getType() const = 0;
};
```

Then four concrete types:

**`MarketEvent`** — new price data for a symbol:
```cpp
struct MarketEvent : Event {
    std::string symbol;
    double      price;
    std::string timestamp;
};
```

**`FeatureMarketEvent`** — inherits MarketEvent, adds the feature vector:
```cpp
struct FeatureMarketEvent : public MarketEvent {
    std::vector<double> features;   // 34 values
    std::vector<double> timeMark;   // [month, day, weekday]
};
```

This inheritance means the engine can handle all MARKET events uniformly. `MLStrategy` recovers the features with a `dynamic_cast` — if it gets a plain `MarketEvent` (not a `FeatureMarketEvent`), the cast returns `nullptr` and it does nothing. Safe.

**`SignalEvent`** — buy or sell decision:
```cpp
struct SignalEvent : Event {
    std::string symbol;
    SignalType  signalType;   // LONG, EXIT, SHORT, BUY, SELL
};
```

**`OrderEvent`** — sized trade instruction:
```cpp
struct OrderEvent : Event {
    std::string symbol;
    OrderType   orderType;   // BUY, SELL, HOLD
    int         quantity;
    double      price;
};
```

**`FillEvent`** — confirmed execution with actual price:
```cpp
struct FillEvent : Event {
    std::string symbol;
    int         quantity;   // negative for sells
    double      price;      // actual fill price (after slippage)
    double      commission;
};
```

`EventQueue` is a simple FIFO wrapper around `std::queue<std::shared_ptr<Event>>`.

---

### `BacktestConfig.hpp`

**Location:** `backtester/include/config/BacktestConfig.hpp`

Header-only YAML parser. Reads `backtest_config.yaml` and exposes all parameters as a plain struct:

```cpp
struct BacktestConfig {
    std::string symbol;
    std::string featureCsv;
    std::string modelPt;
    std::string featureScalerCsv;
    std::string targetScalerCsv;
    double initialCash;
    double riskFraction;
    double maxSymbolExposure;
    double maxTotalExposure;
    int    maxPositionSize;
    double halfSpread;
    double slippageFraction;
    double marketImpact;
    double commission;
    // ... etc
};
```

Multi-symbol support is handled by up to 20 `symbol_N` / `feature_csv_N` key pairs in the YAML.

---

### `FeatureCSVDataHandler`

**Location:** `backtester/src/market/FeatureCSVDataHandler.cpp`

Reads a feature CSV produced by `pipeline.py` and emits one `FeatureMarketEvent` per row on each `streamNext()` call.

The constructor opens the file and reads the header to determine column positions. On each `streamNext()`:
1. Read the next line
2. Parse all 34 feature values into a `std::vector<double>`
3. Extract the `close` column as the price
4. Extract `timestamp`
5. Compute time mark `[month, day, weekday]` from the timestamp string
6. Push a `FeatureMarketEvent` onto the queue

---

### `MultiAssetDataHandler`

**Location:** `backtester/src/market/MultiAssetDataHandler.cpp`

Wraps N `FeatureCSVDataHandler` instances and synchronises them. The algorithm:

- In the constructor, call `streamNext()` once on each handler to pre-fetch one bar from each (the "peek" buffer)
- On each `streamNext()` call:
  1. Find the earliest timestamp across all peeked events
  2. Push all handlers whose peeked timestamp matches the earliest onto the queue
  3. For each emitted handler, call its `streamNext()` to refill the peek buffer
  4. If all handlers are exhausted, push nothing — the engine will break

This guarantees the portfolio always sees all symbols at a given date together, even if their CSVs have different bar counts or occasional missing dates.

---

### `ScalerParams.hpp`

**Location:** `backtester/include/strategy/ScalerParams.hpp`

Header-only struct that reads `feature_scaler.csv` and applies the StandardScaler transform:

```cpp
struct ScalerParams {
    std::vector<double> mean;
    std::vector<double> scale;

    static ScalerParams loadFromCSV(const std::string& path) {
        // Read feature,mean,scale rows
        // Populate mean and scale vectors
    }

    std::vector<double> transform(const std::vector<double>& features) const {
        std::vector<double> out(features.size());
        for (size_t i = 0; i < features.size(); ++i)
            out[i] = (features[i] - mean[i]) / scale[i];
        return out;
    }

    double inverseTransform(double scaled) const {
        return scaled * scale[0] + mean[0];   // for target scaler (single column)
    }
};
```

If `features.size() != mean.size()`, it throws at transform time. This is how a mismatch between Python's 34 columns and C++'s expected column count is detected.

---

### `MLStrategy`

**Location:** `backtester/src/strategy/MLStrategy.cpp`

This is the bridge between the feature data and the model. Per-symbol state:

```cpp
std::deque<std::vector<double>> featureBuffer_;   // rolling 30-bar window
std::deque<std::vector<double>> timeMarkBuffer_;  // rolling 30-bar time marks
bool                            hasPosition_;     // are we currently long?
torch::jit::Module              model_;           // the TorchScript model
ScalerParams                    featureScaler_;   // to normalise input
ScalerParams                    targetScaler_;    // to denormalise output
```

On each `onMarketEvent()` call:

```cpp
// 1. dynamic_cast to get the features — returns nullptr for plain MarketEvent
const auto* featEvent = dynamic_cast<const FeatureMarketEvent*>(&event);
if (!featEvent) return;

// 2. Normalise features
auto scaledFeatures = featureScaler_.transform(featEvent->features);

// 3. Push to rolling buffer, drop oldest if over capacity
featureBuffer_.push_back(std::move(scaledFeatures));
if (featureBuffer_.size() > seqLen_) featureBuffer_.pop_front();

// 4. Wait until 30 bars are buffered
if (!bufferFull()) return;

// 5. Run the model
double predictedClose = runInference();
```

`runInference()` builds the tensor from the deque, calls the TorchScript model, then inverse-scales the result:

```cpp
auto xEnc  = torch::zeros({1, seqLen_, nFeatures_}, opts);
auto xMark = torch::zeros({1, seqLen_, 3}, opts);

for (int t = 0; t < seqLen_; ++t) {
    for (int f = 0; f < nFeatures_; ++f)
        xEnc[0][t][f] = featureBuffer_[t][f];
    for (int m = 0; m < mark.size(); ++m)
        xMark[0][t][m] = timeMarkBuffer_[t][m];
}

auto output   = model_.forward({xEnc, xMark}).toTensor();
float scaledPred = output[0][0][0].item<float>();   // first step of predLen=5
return targetScaler_.inverseTransform(scaledPred);  // back to dollar price
```

Then the signal logic:
```cpp
// LONG: predicted meaningfully above current price
if (!hasPosition_ && predictedClose > currentClose * (1.0 + buyThreshold_)) {
    queue.push(std::make_shared<SignalEvent>(symbol, SignalType::LONG));
    hasPosition_ = true;
}
// EXIT: predicted below current price
if (hasPosition_ && predictedClose < currentClose * (1.0 - exitThreshold_)) {
    queue.push(std::make_shared<SignalEvent>(symbol, SignalType::EXIT));
    hasPosition_ = false;
}
```

`buyThreshold_=0.005` (0.5%) means the model must predict at least a 0.5% gain before entering. `exitThreshold_=0.0` means any predicted decline (even 0.001%) triggers an exit. This asymmetry reflects the logic that entry requires conviction but exit should be fast.

The entire inference block is wrapped in `#ifdef ML_STRATEGY_ENABLED`. If the project is built without LibTorch, this block compiles out and `runInference()` always returns `-1.0` — the strategy generates no signals, but everything else still compiles and the tests still run.

---

### `BacktestEngine`

**Location:** `backtester/src/engine/BacktestEngine.cpp`

The engine loop is 20 lines but they are the most important 20 lines in the codebase:

```cpp
void BacktestEngine::run() {
    while (true) {
        if (queue.empty())
            dataHandler.streamNext(queue);   // fetch new bar ONLY when queue is drained

        if (queue.empty())
            break;                           // no more data — we're done

        auto event = queue.pop();

        switch (event->getType()) {
        case EventType::MARKET:
            portfolio.updateMarket(*marketEvent);
            strategy.onMarketEvent(*marketEvent, queue);   // may produce SignalEvent
            break;
        case EventType::SIGNAL:
            auto order = portfolio.generateOrder(*signal); // produces OrderEvent
            if (order.orderType != OrderType::HOLD)
                queue.push(order);
            break;
        case EventType::ORDER:
            if (riskManager.approveOrder(*order))
                auto fill = execution.executeOrder(*order); // produces FillEvent
                queue.push(fill);
            break;
        case EventType::FILL:
            portfolio.updateFill(*fill);                   // updates cash + positions
            break;
        }
    }
}
```

The key invariant: `streamNext()` is only called when `queue.empty()`. This means the full `MARKET → SIGNAL → ORDER → FILL` chain for bar *t* completes before bar *t+1* is fetched. If a buy FILL from bar *t* hasn't been applied yet when bar *t+1* arrives, an EXIT signal on bar *t+1* would see zero position and do nothing — the position would never close. The empty-check prevents this.

---

### `Portfolio`

**Location:** `backtester/src/portfolio/Portfolio.cpp`

State tracked per instance:
```cpp
double cash_;
std::unordered_map<std::string, int>    positions_;       // shares held
std::unordered_map<std::string, double> latestPrices_;
std::unordered_map<std::string, double> prevPrice_;
std::unordered_map<std::string, std::deque<double>> returnHistory_;  // for correlation
std::unordered_map<std::string, double> benchmarkUnits_;
std::vector<EquityPoint> equityCurve_;
std::vector<Trade>       trades_;
```

**`updateMarket()`** — called on every bar before the strategy runs:
- Updates `returnHistory_` for the symbol (for correlation discount calculation)
- Updates benchmark equity (`benchmarkUnits_[sym] * currentPrice`)
- Pushes a new `EquityPoint` to the equity curve

**`generateOrder()`** — sizing logic for LONG signals:

```cpp
const double equity    = getTotalEquity();           // cash + all positions at market
const double symValue  = getSymbolPositionValue(sym);
const double totalPos  = getTotalPositionValue();

// Reject if already at cap
if (symValue / equity >= maxSymbolExposure_) return HOLD;
if (totalPos / equity >= maxTotalExposure_)  return HOLD;

// Base size: risk fraction of total equity
int baseQty = floor(equity * riskFraction_ / price);

// Cap by remaining symbol headroom
int symCap   = floor(equity * (maxSymbolExposure_ - symValue/equity) / price);

// Cap by remaining total headroom
int totalCap = floor(equity * (maxTotalExposure_  - totalPos/equity) / price);

int qty = max(1, min({baseQty, symCap, totalCap}));

// Apply correlation discount
double discount = correlationDiscount(sym);
qty = max(1, floor(qty * discount));
```

`correlationDiscount()` computes the rolling Pearson correlation between the new symbol's return history and each currently-held symbol's return history. If `|ρ| > threshold`, the discount scales from 0% (at threshold) to 50% (at ρ=1.0):
```cpp
double excess   = std::abs(rho) - correlationThreshold_;
double maxRange = 1.0 - correlationThreshold_;
double discount = 1.0 - (excess / maxRange) * 0.5;
```

**`updateFill()`** — bookkeeping after a trade executes:
```cpp
positions_[fill.symbol] += fill.quantity;   // negative quantity for sells
cash_ -= fill.quantity * fill.price;        // pay for buys, receive for sells
cash_ -= fill.commission;
```

---

### `RiskManager`

**Location:** `backtester/src/portfolio/RiskManager.cpp`

One job: reject orders where `quantity > maxPositionSize`:

```cpp
bool RiskManager::approveOrder(const OrderEvent& order) const {
    return std::abs(order.quantity) <= maxPositionSize_;
}
```

It's a simple gate here, but the design is extensible — drawdown limits, sector exposure checks, VaR constraints could all be added without touching Portfolio or the engine.

---

### `SimulatedExecution`

**Location:** `backtester/src/execution/SimulatedExecution.cpp`

Converts an approved `OrderEvent` to a `FillEvent` with realistic pricing:

```cpp
if (order.orderType == OrderType::BUY) {
    fillPrice = rawPrice * (1.0 + halfSpread_ + slippageFraction_)
                + marketImpact_ * quantity;
}
else if (order.orderType == OrderType::SELL) {
    fillPrice = rawPrice * (1.0 - halfSpread_ - slippageFraction_)
                - marketImpact_ * quantity;
    quantity = -quantity;   // negative → Portfolio reduces position
}

return FillEvent(order.symbol, quantity, fillPrice, commission_);
```

Three cost components:
- `halfSpread`: you buy at the ask (above mid) and sell at the bid (below mid). The full bid-ask spread is `2 × halfSpread`, but each trade only pays one side.
- `slippageFraction`: additional cost from market orders moving the market slightly against you before your fill.
- `marketImpact * qty`: larger orders push the price more. Currently disabled by default (`marketImpact=0.0`).

The `quantity` sign flip for sells is important: `Portfolio::updateFill` does `positions_[sym] += fill.quantity` — so a negative quantity reduces the position.

---

### `PerformanceMetrics`

**Location:** `backtester/include/portfolio/PerformanceMetrics.hpp`

Header-only, takes the completed equity curve and computes all final metrics:

```cpp
// Daily returns
for (size_t i = 1; i < n; ++i)
    dailyReturns[i-1] = curve[i].equity / curve[i-1].equity - 1.0;

// Sharpe: annualised excess return / volatility
double excessMean = mean(dailyReturns) - riskFreeRate/252.0;
double sharpe     = (excessMean / stddev_bessel(dailyReturns)) * sqrt(252.0);

// Max drawdown
double peak = curve[0].equity;
double maxDD = 0.0;
for (const auto& p : curve) {
    peak = max(peak, p.equity);
    maxDD = max(maxDD, (peak - p.equity) / peak);
}

// Information Ratio: active return (vs benchmark) per unit of active risk
for (size_t i = 1; i < n; ++i)
    activeReturns[i-1] = dailyReturns[i-1] - benchmarkDailyReturns[i-1];
double IR = (mean(activeReturns) / stddev_bessel(activeReturns)) * sqrt(252.0);
```

The `sqrt(252)` multiplier annualises daily statistics to yearly. 252 is the number of US trading days per year. `stddev_bessel` uses `n-1` in the denominator — this corrects for the statistical bias of estimating population variance from a sample.

---

### `ml_main.cpp`

**Location:** `backtester/ml_main.cpp`

The entry point that wires everything together:

1. Load config from `argv[1]` (the YAML file path)
2. For each symbol in the config, create a `FeatureCSVDataHandler`
3. Create one `MLStrategy` per symbol (loads the model, scalers)
4. Wrap all handlers in `MultiAssetDataHandler`
5. Wrap all strategies in `MultiSymbolStrategy`
6. Construct `BacktestEngine` and call `.run()`
7. After `.run()` returns, call `PerformanceMetrics::compute()` on the equity curve
8. Write `ml_equity.csv`, `ml_trades.csv`, `ml_metrics.csv`

---

## 7. `run_pipeline.py` — the orchestrator

**Location:** `run_pipeline.py`

Runs all three Python stages in sequence:

```python
# Stage 1: feature engineering
subprocess.run(["python", "research/features/pipeline.py", data_dir, "-o", features_dir])

# Stage 2: training (unless --skip-train)
model_interface = Model_Interface(args)
model_interface.train(data)

# Stage 3: export
export_model()
save_scaler_params(train_dataset)
```

Pass `--skip-train` to reuse an existing checkpoint and just re-export — useful when iterating on the export format without retraining.

---

## 8. Data Flow Cheat Sheet

```
data/AAPL.csv
  │   [timestamp, open, high, low, close, volume]
  │
  ▼ pipeline.py
  │   bar-by-bar: _BarCtx advances, technicalIndicators.py called at each step
  │
features/AAPL_features.csv
  │   [timestamp, high, low, volume, adj_close, P, R1..R3, S1..S3,
  │    obv, volume_zscore, rsi, macd, macds, macdh, sma, lma, sema, lema,
  │    overnight_gap, return_lag_1, return_lag_3, return_lag_5, volatility,
  │    SR_K, SR_D, SR_RSI_K, SR_RSI_D, ATR, HL_PCT, PCT_CHG, close]
  │
  ▼ DataFrame.py
  │   sliding windows of size 30 → (seqX, seqY, seqXMark, seqYMark)
  │   scaler fitted on train split only
  │
  ▼ Interface.py (training loop)
  │   forward: Embedding → Encoder (3 layers) → Decoder (2 layers) → projection
  │   loss: MSE on predicted close vs actual close
  │   early stopping on val MSE, patience=10
  │
models/checkpoint.pth         ← best weights (Python format)
models/featureScaler.pkl      ← sklearn StandardScaler for 33 features
models/targetScaler.pkl       ← sklearn StandardScaler for close
  │
  ▼ exportModel.py
  │   TransformerInferenceWrapper: fuses encoder+decoder into forward(xEnc, xMarkEnc)
  │   torch.jit.trace with dummy (1, 30, 34) and (1, 30, 3) inputs
  │   save_scaler_params: writes mean_/scale_ to CSV
  │
models/transformer.pt         ← portable TorchScript binary (C++ reads this)
models/feature_scaler.csv     ← 33 rows: feature, mean, scale
models/target_scaler.csv      ← 1 row: close, mean, scale
  │
  ▼ C++ backtester
  │
  ├── FeatureCSVDataHandler   reads features/AAPL_features.csv row by row
  │                           emits FeatureMarketEvent(symbol, price, features[34], timeMark[3])
  │
  ├── MultiAssetDataHandler   synchronises N handlers by timestamp
  │
  ├── MLStrategy              on each FeatureMarketEvent:
  │                             1. ScalerParams::transform(features)
  │                             2. push to featureBuffer_ (deque, max 30)
  │                             3. if full: build tensors, model_.forward()
  │                             4. ScalerParams::inverseTransform(output[0][0][0])
  │                             5. emit LONG or EXIT SignalEvent
  │
  ├── BacktestEngine          MARKET → SIGNAL → ORDER → FILL loop
  │
  ├── Portfolio               generateOrder: size = floor(equity × 0.10 / price)
  │                                           capped by exposure limits + correlation
  │                           updateFill: positions += qty, cash -= qty × fillPrice
  │
  ├── RiskManager             approveOrder: qty ≤ maxPositionSize
  │
  └── SimulatedExecution      fillPrice = price × (1 + spread + slippage) + impact×qty
  │
output/ml_equity.csv          [timestamp, equity, price, benchmarkEquity]
output/ml_trades.csv          [timestamp, symbol, price, qty, direction, profit]
output/ml_metrics.csv         [sharpe, IR, maxDrawdown, alpha, totalReturn]