"""
Signal Lab — BTC/ETH 무기한선물 롱/숏/관망 신호 엔진 (4시간봉 특징, 매일 00:00 UTC 결정, 1~3일 보유)

python signal_lab.py --data DIR --state state.json --out OUT --mode daily|improve [--eth-new eth_new.json] [--fund-new f.json ...] [--add-candidates cands.json]

DIR 에 필요한 파일
  hist.csv.gz, latest.csv   : Bitstamp BTC/USD 1분봉 (없으면 GitHub에서 받음)
  okx_eth_4h.csv            : OKX ETH-USDT-SWAP 4시간봉 (ts,open,high,low,close,...)
  funding.csv               : symbol,fundingTime,fundingRate (BTCUSDT/ETHUSDT, 8시간마다)
--eth-new  : OKX /api/v5/market/candles 응답 JSON (새 4시간봉 이어붙이기, 검증 후 병합)
--fund-new : OKX /api/v5/public/funding-rate-history 응답 JSON (여러 개 가능)
--add-candidates : 후보 전략 설정 JSON 리스트 (개선 세션이 제안). 문법은 CANDIDATE GRAMMAR 참조.

비용 가정: 테이커 수수료 0.055%/편도 + 슬리피지 0.03%/편도, 실제 펀딩비(롱 지급, 숏 수령).
선택 규칙: 개발구간(최근 180일 제외) Deflated Sharpe >= 0.90, 개발 Sharpe >= 0.5, 거래 40회 이상, MDD > -45%,
          홀드아웃(최근 180일) Sharpe > 0. 통과 전략이 없으면 '관망'.
교체 규칙(improve): 도전자가 통과하고 (개발+홀드아웃) Sharpe 가 챔피언보다 0.15 이상 높을 때, 또는 챔피언이 탈락했을 때만 교체.
"""
import os, sys, json, gzip, argparse, urllib.request, datetime as dt, math, hashlib
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis

FEE = 0.00055 + 0.0003          # per side
START = pd.Timestamp('2020-01-01', tz='UTC')
HOLDOUT_DAYS = 180
RISK_PER_TRADE = 0.01
LEV_CAP = 3.0

# ---------------------------------------------------------------- data
def load_btc(D):
    B = 'https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data/'
    for f, u in [('hist.csv.gz', 'historical/btcusd_bitstamp_1min_2012-2025.csv.gz'), ('latest.csv', 'updates/btcusd_bitstamp_1min_latest.csv')]:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(B + u, p)
    m = pd.concat([pd.read_csv(os.path.join(D, 'hist.csv.gz')), pd.read_csv(os.path.join(D, 'latest.csv'))])
    m = m.drop_duplicates('timestamp').sort_values('timestamp')
    m = m[m.timestamp >= START.timestamp() - 86400 * 400]
    m.index = pd.to_datetime(m.timestamp, unit='s', utc=True)
    b = m.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    last_min = m.index[-1]
    # drop a 4h bar that is not complete yet
    if b.index[-1] + pd.Timedelta(hours=4) > last_min + pd.Timedelta(minutes=1):
        b = b.iloc[:-1]
    return b

def okx_rows_to_df(rows):
    df = pd.DataFrame([r[:6] + [r[8] if len(r) > 8 else '1'] for r in rows], columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'confirm'])
    df = df.astype({'ts': 'int64', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df = df[df.confirm.astype(str) == '1']
    df.index = pd.to_datetime(df.ts, unit='ms', utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']].sort_index()

def load_eth(D, new_json, log):
    raw = pd.read_csv(os.path.join(D, 'okx_eth_4h.csv'))
    raw = raw[raw.confirm.astype(str) == '1'] if 'confirm' in raw else raw
    e = pd.DataFrame({'open': raw.open.values, 'high': raw.high.values, 'low': raw.low.values, 'close': raw.close.values,
                      'volume': raw.vol.values if 'vol' in raw else raw.volume.values},
                     index=pd.to_datetime(raw.ts, unit='ms', utc=True)).sort_index()
    e = e[~e.index.duplicated(keep='last')]
    if new_json:
        try:
            js = json.load(open(new_json))
            n = okx_rows_to_df(js['data'] if isinstance(js, dict) else js)
            ok = ((n.high >= n[['open', 'close']].max(1)) & (n.low <= n[['open', 'close']].min(1)) & (n.low > 0)).all()
            ov = e.index.intersection(n.index)
            dev = float((n.loc[ov, 'close'] / e.loc[ov, 'close'] - 1).abs().max()) if len(ov) else 0.0
            gap = (n.index.min() - e.index.max()) if not len(ov) else pd.Timedelta(0)
            src = js.get('source', 'okx') if isinstance(js, dict) else 'okx'
            tol = 0.002 if src == 'okx' else 0.006
            if not ok or dev > tol or gap > pd.Timedelta(hours=4):
                log.append(f'ETH 새 봉 검증 실패(OHLC={ok}, 겹침오차={dev:.4f}, 공백={gap}) — 병합 안 함')
            else:
                before = len(e)
                e = pd.concat([e[~e.index.isin(n.index)], n]).sort_index()
                log.append(f'ETH 4시간봉 {len(e) - before}개 추가 ({src}, 마지막 {e.index[-1]:%m-%d %H:%M} UTC)')
        except Exception as ex:
            log.append(f'ETH 새 봉 읽기 실패: {ex}')
    return e

def load_funding(D, new_jsons, log):
    f = pd.read_csv(os.path.join(D, 'funding.csv'))
    f['t'] = pd.to_datetime((f.fundingTime // 1000) * 1000, unit='ms', utc=True).dt.floor('h')
    f = f[['symbol', 't', 'fundingRate']]
    add = []
    for p in new_jsons or []:
        try:
            js = json.load(open(p))
            for r in (js['data'] if isinstance(js, dict) else js):
                sym = 'BTCUSDT' if r['instId'].startswith('BTC') else 'ETHUSDT'
                add.append([sym, pd.to_datetime(int(r['fundingTime']), unit='ms', utc=True).floor('h'), float(r.get('realizedRate') or r['fundingRate'])])
        except Exception as ex:
            log.append(f'펀딩비 새 데이터 읽기 실패: {ex}')
    if add:
        a = pd.DataFrame(add, columns=['symbol', 't', 'fundingRate'])
        a = a[a.fundingRate.abs() < 0.03]
        n0 = len(f)
        f = pd.concat([f, a]).drop_duplicates(['symbol', 't'], keep='first')
        log.append(f'펀딩비 {len(f) - n0}건 추가')
    return f.sort_values(['symbol', 't']).reset_index(drop=True)

# ---------------------------------------------------------------- features
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def prep(b, fund_sym):
    b = b[b.index >= START - pd.Timedelta(days=400)].copy()
    c, h, l = b.close, b.high, b.low
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(1)
    b['atr'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    b['atrp'] = b.atr / c
    b['atrp_rank'] = b.atrp.rolling(6 * 365, min_periods=600).rank(pct=True)
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    b['rsi'] = 100 - 100 / (1 + up / dn)
    for n in [50, 100, 200, 400, 1200]:
        b[f'ema{n}'] = ema(c, n)
    fr = fund_sym.set_index('t').fundingRate
    fr = fr[~fr.index.duplicated()]
    b['fund_rate_at'] = fr.reindex(b.index)              # charged at bar open (00/08/16 UTC)
    b['fund_last'] = fr.reindex(b.index.union(fr.index)).ffill().reindex(b.index)
    b['fund_3d'] = b.fund_last.rolling(18, min_periods=1).mean()
    b['hour'] = b.index.hour
    b['rv'] = np.log(c).diff().rolling(180, min_periods=60).std() * math.sqrt(6 * 365)
    return b

# ---------------------------------------------------------------- CANDIDATE GRAMMAR
# {"rule": "ema", "n": 50|100|200|400|1200}                 c>EMA → 롱, c<EMA → 숏
# {"rule": "tsmom", "L": 정수(봉)}                           L봉 수익률 부호
# {"rule": "donch", "N": 정수(봉)}                           N봉 고가 돌파 롱 / 저가 이탈 숏, 반대 신호까지 유지
# {"rule": "rsi_mr", "lo": 20~40, "hi": 60~80}             RSI 역추세
# {"rule": "ml", "th": 0.52~0.65}                           LightGBM 다음 24h 방향 확률 (연 1회 재학습)
# 공통: "k": 손절 ATR 배수(1~6), "dir": "both"|"long"|"short",
#       "vt": 목표 연변동성(0.2~1.0, 넣으면 변동성 맞춤 비중; 비중<=1),
#       "filters": {"fund_max": 펀딩 상한(예 0.0003), "vol_q": ATR% 백분위 상한(0~1), "regime": EMA기간(롱은 위, 숏은 아래에서만)}
DESC = {'ema': 'EMA({n}) 추세', 'tsmom': '{L}봉 모멘텀', 'donch': '{N}봉 돌파', 'rsi_mr': 'RSI {lo}/{hi} 역추세', 'ml': 'LightGBM 확률>{th}'}

def cid(cfg):
    s = json.dumps({k: cfg[k] for k in sorted(cfg) if k != 'id'}, sort_keys=True)
    return cfg['rule'] + '_' + hashlib.md5(s.encode()).hexdigest()[:6]

def describe(cfg):
    d = DESC[cfg['rule']].format(**cfg)
    d += f" · 손절 {cfg.get('k', 2.5)}ATR"
    if cfg.get('vt'): d += f" · 변동성 {cfg['vt'] * 100:.0f}% 맞춤 비중"
    d += {'both': ' · 양방향', 'long': ' · 롱만', 'short': ' · 숏만'}[cfg.get('dir', 'both')]
    f = cfg.get('filters') or {}
    if 'fund_max' in f: d += f" · 펀딩>{f['fund_max'] * 100:.3f}%면 추격 금지"
    if 'vol_q' in f: d += f" · 변동성 상위{(1 - f['vol_q']) * 100:.0f}% 회피"
    if 'regime' in f: d += f" · EMA{f['regime']} 국면 필터"
    return d

def default_grid():
    g = []
    for k in [2.0, 3.5]:
        for dr in ['both', 'long']:
            for n in [50, 100, 200, 400]: g.append({'rule': 'ema', 'n': n, 'k': k, 'dir': dr})
            for L in [18, 42, 90, 180]: g.append({'rule': 'tsmom', 'L': L, 'k': k, 'dir': dr})
            for N in [30, 60, 120]: g.append({'rule': 'donch', 'N': N, 'k': k, 'dir': dr})
            for lo, hi in [(25, 75), (30, 70)]: g.append({'rule': 'rsi_mr', 'lo': lo, 'hi': hi, 'k': k, 'dir': dr, 'filters': {'regime': 1200}})
    for n in [100, 200]:
        g.append({'rule': 'ema', 'n': n, 'k': 3.5, 'dir': 'both', 'filters': {'fund_max': 0.0003}})
        g.append({'rule': 'ema', 'n': n, 'k': 3.5, 'dir': 'both', 'filters': {'vol_q': 0.9}})
    for vt in [0.4]:
        for dr in ['both', 'long']:
            for n in [100, 200]: g.append({'rule': 'ema', 'n': n, 'k': 3.5, 'dir': dr, 'vt': vt})
            for L in [90, 180]: g.append({'rule': 'tsmom', 'L': L, 'k': 3.5, 'dir': dr, 'vt': vt})
            g.append({'rule': 'donch', 'N': 120, 'k': 3.5, 'dir': dr, 'vt': vt})
    for th in [0.55, 0.6]:
        for dr in ['both', 'long']: g.append({'rule': 'ml', 'th': th, 'k': 3.0, 'dir': dr})
    for c in g: c['id'] = cid(c)
    return g

def valid_cfg(c):
    try:
        assert c['rule'] in DESC
        if c['rule'] == 'ema': assert int(c['n']) in (50, 100, 200, 400, 1200)
        if c['rule'] == 'tsmom': assert 3 <= int(c['L']) <= 1000
        if c['rule'] == 'donch': assert 6 <= int(c['N']) <= 1000
        if c['rule'] == 'rsi_mr': assert 10 <= c['lo'] < 50 < c['hi'] <= 90
        if c['rule'] == 'ml': assert 0.5 < c['th'] < 0.8
        assert c.get('vt') is None or 0.1 <= float(c['vt']) <= 1.5
        assert 0.5 <= float(c.get('k', 2.5)) <= 8 and c.get('dir', 'both') in ('both', 'long', 'short')
        f = c.get('filters') or {}
        assert set(f) <= {'fund_max', 'vol_q', 'regime'}
        if 'regime' in f: assert int(f['regime']) in (50, 100, 200, 400, 1200)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------- signals (raw, at bar close)
ML_CACHE = {}
def ml_prob(b, key):
    if key in ML_CACHE: return ML_CACHE[key]
    import lightgbm as lgb
    c = b.close
    X = pd.DataFrame(index=b.index)
    for L in [6, 18, 42, 90, 180]: X[f'r{L}'] = np.log(c / c.shift(L))
    for n in [50, 200, 400, 1200]: X[f'e{n}'] = c / b[f'ema{n}'] - 1
    X['atrp'] = b.atrp; X['rsi'] = b.rsi; X['f'] = b.fund_last; X['f3'] = b.fund_3d
    X['vr'] = b.atrp / b.atrp.rolling(180).mean()
    y = (np.log(c.shift(-6) / c) > 0).astype(float)
    y[c.shift(-6).isna()] = np.nan
    dec = X.index[X.index.hour == 20]                       # last bar before 00:00 decision
    p = pd.Series(np.nan, index=b.index)
    years = sorted(set(dec.year))
    for yr in years:
        tr = dec[(dec < pd.Timestamp(f'{yr}-01-01', tz='UTC') - pd.Timedelta(days=2))]
        tr = tr[X.loc[tr].notna().all(axis=1) & y.loc[tr].notna()]
        te = dec[(dec.year == yr)]
        te = te[X.loc[te].notna().all(axis=1)]
        if len(tr) < 500 or not len(te): continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(X.loc[tr], y.loc[tr])
        p.loc[te] = m.predict_proba(X.loc[te])[:, 1]
    ML_CACHE[key] = p
    return p

def raw_signal(b, cfg, key):
    c = b.close; r = cfg['rule']
    if r == 'ema':
        s = np.sign(c - b[f"ema{cfg['n']}"])
    elif r == 'tsmom':
        s = np.sign(c / c.shift(int(cfg['L'])) - 1)
    elif r == 'donch':
        N = int(cfg['N'])
        hi = b.high.shift(1).rolling(N).max(); lo = b.low.shift(1).rolling(N).min()
        ev = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, -1.0, np.nan)), index=b.index)
        s = ev.ffill().fillna(0)
    elif r == 'rsi_mr':
        s = pd.Series(np.where(b.rsi < cfg['lo'], 1.0, np.where(b.rsi > cfg['hi'], -1.0, 0.0)), index=b.index)
    elif r == 'ml':
        p = ml_prob(b, key)
        s = pd.Series(np.where(p > cfg['th'], 1.0, np.where(p < 1 - cfg['th'], -1.0, 0.0)), index=b.index)
        s[p.isna()] = 0.0
        s = s.where(b.index.hour == 20).ffill().fillna(0)
    s = pd.Series(s, index=b.index).fillna(0)
    d = cfg.get('dir', 'both')
    if d == 'long': s = s.clip(lower=0)
    if d == 'short': s = s.clip(upper=0)
    f = cfg.get('filters') or {}
    if 'fund_max' in f:
        s[(s > 0) & (b.fund_last > f['fund_max'])] = 0
        s[(s < 0) & (b.fund_last < -f['fund_max'])] = 0
    if 'vol_q' in f:
        s[b.atrp_rank > f['vol_q']] = 0
    if 'regime' in f:
        e = b[f"ema{int(f['regime'])}"]
        s[(s > 0) & (c < e)] = 0
        s[(s < 0) & (c > e)] = 0
    return s

# ---------------------------------------------------------------- backtest (daily decision at 00:00 UTC bar open)
def backtest(b, sig, k, vt=None):
    o, h, l, c = b.open.values, b.high.values, b.low.values, b.close.values
    atr = b.atr.values; fr = b.fund_rate_at.fillna(0).values
    rv = b.rv.values
    hrs = b.index.hour.values; n = len(b)
    sg = sig.values
    pos = 0; sz = 0.0; stop = np.nan; ref = np.nan; entry = np.nan
    bar_ret = np.zeros(n); trades = []; posarr = np.zeros(n)
    for i in range(1, n):
        r = 0.0
        if hrs[i] == 0:                                      # decision
            want = sg[i - 1]
            if want != pos:
                if pos != 0:
                    r += sz * (pos * (o[i] / ref - 1) - FEE)
                    trades.append((i, sz * (pos * (o[i] / entry - 1) - 2 * FEE)))
                pos = int(want)
                if pos != 0:
                    sz = 1.0 if not vt else float(min(1.0, vt / rv[i - 1])) if rv[i - 1] > 0 else 1.0
                    r -= sz * FEE; ref = o[i]; entry = o[i]
                    stop = o[i] - pos * k * atr[i - 1]
            elif pos != 0:
                r += sz * pos * (o[i] / ref - 1); ref = o[i]
                ns = o[i] - pos * k * atr[i - 1]           # daily trailing stop (only tightens)
                stop = max(stop, ns) if pos == 1 else min(stop, ns)
        elif pos != 0:
            r += sz * pos * (o[i] / ref - 1); ref = o[i]
        if pos != 0 and hrs[i] in (0, 8, 16):
            r -= sz * pos * fr[i]
        if pos == 1 and l[i] <= stop:
            px = min(o[i], stop); r += sz * ((px / ref - 1) - FEE); trades.append((i, sz * (px / entry - 1 - 2 * FEE))); pos = 0
        elif pos == -1 and h[i] >= stop:
            px = max(o[i], stop); r += sz * (-(px / ref - 1) - FEE); trades.append((i, sz * (-(px / entry - 1) - 2 * FEE))); pos = 0
        elif pos != 0:
            r += sz * pos * (c[i] / ref - 1); ref = c[i]
        bar_ret[i] = r; posarr[i] = pos * sz
    trades = [(b.index[i], p) for i, p in trades]
    br = pd.Series(bar_ret, index=b.index)
    day = br.groupby(br.index.floor('D')).sum()
    return day, trades, pd.Series(posarr, index=b.index), (pos, stop, entry, sz)

def stats(day, trades, nyears_min=0.3):
    day = day.dropna()
    if len(day) < 60 or day.std() == 0: return None
    sr = day.mean() / day.std() * math.sqrt(365)
    eq = (1 + day).cumprod(); dd = (eq / eq.cummax() - 1).min()
    yrs = len(day) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    t = np.array(trades) if len(trades) else np.array([0.0])
    pf = t[t > 0].sum() / max(1e-9, -t[t < 0].sum())
    return {'sharpe': round(float(sr), 3), 'cagr': round(float(cagr), 4), 'maxdd': round(float(dd), 4), 'trades': int(len(trades)),
            'win': round(float((t > 0).mean()), 3), 'pf': round(float(pf), 3), 'exposure': None,
            'skew': float(skew(day)), 'kurt': float(kurtosis(day, fisher=False)), 'srd': float(day.mean() / day.std()), 'n': int(len(day))}

def dsr(st, sr_trials):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado), per-day SR."""
    N = max(2, len(sr_trials)); v = np.var(sr_trials) if len(sr_trials) > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    srd = st['srd']; T = st['n']
    den = math.sqrt(max(1e-9, 1 - st['skew'] * srd + (st['kurt'] - 1) / 4 * srd ** 2))
    return float(norm.cdf((srd - sr0) * math.sqrt(T - 1) / den))

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--state', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='daily', choices=['daily', 'improve'])
    ap.add_argument('--eth-new'); ap.add_argument('--fund-new', nargs='*'); ap.add_argument('--add-candidates')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    now = pd.Timestamp.now('UTC').floor('min')
    log = []
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    state.setdefault('candidates', []); state.setdefault('champion', {}); state.setdefault('ledger', []); state.setdefault('log', [])
    if not state['candidates']:
        state['candidates'] = default_grid(); log.append(f'기본 후보 전략 {len(state["candidates"])}개 등록')
    if a.add_candidates:
        added = 0; known = {c['id'] for c in state['candidates']}
        for c in json.load(open(a.add_candidates)):
            c = {k: v for k, v in c.items() if k != 'id'}
            if valid_cfg(c):
                c['id'] = cid(c)
                if c['id'] not in known: state['candidates'].append(c); known.add(c['id']); added += 1
            else:
                log.append(f'형식이 틀린 후보 거부: {json.dumps(c, ensure_ascii=False)[:120]}')
        log.append(f'새 후보 전략 {added}개 추가 (총 {len(state["candidates"])}개)')

    fund = load_funding(a.data, a.fund_new, log)
    eth = load_eth(a.data, a.eth_new, log)
    btc = load_btc(a.data)
    # persist merged inputs for next run
    eth_out = pd.DataFrame({'ts': ((eth.index - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(milliseconds=1)).astype('int64'), 'open': eth.open, 'high': eth.high, 'low': eth.low, 'close': eth.close, 'vol': eth.volume, 'confirm': 1})
    eth_out.to_csv(os.path.join(a.out, 'okx_eth_4h.csv'), index=False)
    fo = fund.copy(); fo['fundingTime'] = ((fo.t - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(milliseconds=1)).astype('int64')
    fo[['symbol', 'fundingTime', 'fundingRate']].to_csv(os.path.join(a.out, 'funding.csv'), index=False)

    coins = {'BTC': prep(btc, fund[fund.symbol == 'BTCUSDT']), 'ETH': prep(eth, fund[fund.symbol == 'ETHUSDT'])}
    out = {'signals': [], 'equity': [], 'board': [], 'paper': [], 'log': []}
    for coin, b in coins.items():
        last_bar = b.index[-1]
        decision_t = (last_bar + pd.Timedelta(hours=4)).floor('D') if last_bar.hour == 20 else None
        stale = decision_t is None or (now - (last_bar + pd.Timedelta(hours=4))) > pd.Timedelta(hours=30)
        b = b[b.index >= START - pd.Timedelta(days=5)]
        # decision bar placeholder so backtest decides on the newest signal
        if decision_t is not None:
            ph = b.iloc[[-1]].copy(); ph.index = [decision_t]; ph[['open', 'high', 'low', 'close']] = b.close.iloc[-1]
            ph['fund_rate_at'] = 0.0
            bx = pd.concat([b, ph])
        else:
            bx = b
        cut = bx.index[-1] - pd.Timedelta(days=HOLDOUT_DAYS)
        res = []
        for cfg in state['candidates']:
            try:
                s = raw_signal(bx, cfg, coin)
                day, trades, posr, last = backtest(bx, s, float(cfg.get('k', 2.5)), cfg.get('vt'))
                dday = day[(day.index >= START) & (day.index < cut.floor('D'))]
                hday = day[day.index >= cut.floor('D')]
                c0 = cut.floor('D')
                st_dev = stats(dday, [p for t, p in trades if START <= t < c0])
                st_ho = stats(hday, [p for t, p in trades if t >= c0])
                st_all = stats(day[day.index >= START], [p for t, p in trades if t >= START])
                if not st_dev or not st_all: continue
                res.append({'cfg': cfg, 'day': day, 'dev': st_dev, 'ho': st_ho, 'all': st_all, 'last': last, 'sig_now': float(s.iloc[-2]) if decision_t is not None else float(s.iloc[-1])})
            except Exception as ex:
                log.append(f'{coin} {cfg.get("id")} 계산 오류: {ex}')
        trials = [r['dev']['srd'] for r in res]
        for r in res:
            r['dsr'] = dsr(r['dev'], trials)
            ho_sr = r['ho']['sharpe'] if r['ho'] else -9
            r['eligible'] = bool(r['dsr'] >= 0.90 and r['dev']['sharpe'] >= 0.5 and r['dev']['trades'] >= 40 and r['dev']['maxdd'] > -0.45 and ho_sr > 0)
            r['score'] = r['dev']['sharpe'] * 0.6 + (r['ho']['sharpe'] if r['ho'] else 0) * 0.4
        res.sort(key=lambda r: (-r['eligible'], -r['score']))
        champ_id = (state['champion'].get(coin) or {}).get('id')
        champ = next((r for r in res if r['cfg']['id'] == champ_id), None)
        best = next((r for r in res if r['eligible']), None)
        if a.mode == 'improve' or champ is None:
            if champ is None or not champ['eligible']:
                new = best
            elif best and best['cfg']['id'] != champ_id and best['score'] >= champ['score'] + 0.15:
                new = best
            else:
                new = champ
            if (new and new['cfg']['id']) != champ_id:
                if new:
                    log.append(f"{coin} 챔피언 교체: {describe(champ['cfg']) if champ else '없음'} → {describe(new['cfg'])} (DSR {new['dsr']:.2f}, 개발 Sharpe {new['dev']['sharpe']}, 홀드아웃 {new['ho']['sharpe'] if new['ho'] else '-'})")
                else:
                    log.append(f'{coin} 검증을 통과한 전략 없음 → 관망')
            champ = new
            state['champion'][coin] = {'id': champ['cfg']['id'], 'since': str(now.date())} if champ else {}
        elif champ is not None and not champ['eligible']:
            log.append(f'{coin} 챔피언이 오늘 검증 기준에서 탈락 — 다음 개선 때 교체 검토, 그 전까진 관망')
        # ------------- today's signal (champion if validated, else top candidate as reference only)
        price = float(b.close.iloc[-1]); atr = float(b.atr.iloc[-1])
        ok = bool(champ and champ['eligible'])
        use = champ if ok else (res[0] if res else None)
        sgn = 0; stop = None; stop_pct = None; lev = None; sz = 1.0
        if use is not None and not stale:
            sgn = int(use['sig_now']); k = float(use['cfg'].get('k', 2.5))
            pos_l, stop_l, entry_l, sz_l = use['last']
            vt = use['cfg'].get('vt'); rv = float(b.rv.iloc[-1])
            sz = float(sz_l) if (sgn != 0 and pos_l == sgn) else (min(1.0, vt / rv) if vt and rv > 0 else 1.0)
            if sgn:
                stop = price - sgn * k * atr
                if pos_l == sgn and not np.isnan(stop_l):  # holding: trailing stop only tightens
                    stop = max(stop, float(stop_l)) if sgn == 1 else min(stop, float(stop_l))
            stop_pct = abs(price - stop) / price if stop else None
            lev = min(LEV_CAP, sz * RISK_PER_TRADE / stop_pct) if stop_pct else None
        word = {1: '롱', -1: '숏', 0: '관망'}[sgn]
        st = use['all'] if use else {}
        sig_row = {'coin': coin, 'decision_utc': str(decision_t), 'signal': word if ok and not stale else '관망', 'ref_signal': word,
                   'kind': '검증 통과' if ok else '참고(검증 미통과)', 'price': round(price, 2),
                   'stop': round(stop, 2) if stop else None, 'stop_pct': round(stop_pct * 100, 2) if stop_pct else None,
                   'take_profit': round(price + sgn * 2 * abs(price - stop), 2) if stop else None,
                   'notional_pct': round(lev * 100, 1) if lev else None, 'size_frac': round(sz, 3),
                   'strategy': describe(use['cfg']) if use else '없음', 'strategy_id': use['cfg']['id'] if use else None,
                   'dsr': round(use['dsr'], 3) if use else None, 'dev_sharpe': use['dev']['sharpe'] if use else None,
                   'ho_sharpe': use['ho']['sharpe'] if use and use['ho'] else None,
                   'cagr_1x': st.get('cagr'), 'maxdd_1x': st.get('maxdd'), 'trades': st.get('trades'), 'win': st.get('win'), 'pf': st.get('pf'),
                   'funding_now_pct': round(float(b.fund_last.iloc[-1]) * 100, 4),
                   'status': '데이터 지연' if stale else ('검증 통과' if ok else '검증 통과 전략 없음 → 관망'), 'n_candidates': len(res),
                   'next_decision_utc': str((decision_t or b.index[-1].floor('D')) + pd.Timedelta(days=1))}
        out['signals'].append(sig_row)
        # ------------- equity (shown strategy vs buy&hold), last 730 days
        if use:
            d = use['day'][use['day'].index >= now.floor('D') - pd.Timedelta(days=730)]
            eq = (1 + d).cumprod()
            px = b.close.groupby(b.close.index.floor('D')).last().reindex(eq.index).ffill()
            hold = px / px.iloc[0]
            for t in eq.index:
                out['equity'].append({'date': str(t.date()), 'coin': coin, 'strategy': round(float(eq[t]), 4), 'hold': round(float(hold[t]), 4)})
        for r in res[:10]:
            out['board'].append({'coin': coin, 'id': r['cfg']['id'], 'strategy': describe(r['cfg']), 'eligible': r['eligible'],
                                 'champion': bool(champ and r['cfg']['id'] == champ['cfg']['id']), 'dsr': round(r['dsr'], 3),
                                 'dev_sharpe': r['dev']['sharpe'], 'ho_sharpe': r['ho']['sharpe'] if r['ho'] else None,
                                 'cagr_1x': r['all']['cagr'], 'maxdd_1x': r['all']['maxdd'], 'trades': r['all']['trades']})
        # ------------- paper ledger: score open entries for this coin
        for e in state['ledger']:
            if e['coin'] != coin or e.get('closed'): continue
            t0 = pd.Timestamp(e['decision_utc']); t1 = t0 + pd.Timedelta(days=1)
            seg = b[(b.index >= t0) & (b.index < t1)]
            if len(seg) < 6: continue
            sg = {'롱': 1, '숏': -1, '관망': 0}[e['signal']]
            if sg == 0:
                e.update({'closed': True, 'pnl_pct': 0.0, 'exit': None}); continue
            entry = float(seg.open.iloc[0]); px = float(seg.close.iloc[-1]); hit = False
            for _, rw in seg.iterrows():
                if sg == 1 and e['stop'] and rw.low <= e['stop']: px = min(rw.open, e['stop']); hit = True; break
                if sg == -1 and e['stop'] and rw.high >= e['stop']: px = max(rw.open, e['stop']); hit = True; break
            fund_paid = float(seg.fund_rate_at.fillna(0)[seg.index.hour.isin([0, 8, 16])].sum()) * sg
            pnl = e.get('size', 1.0) * (sg * (px / entry - 1) - 2 * FEE - fund_paid)
            e.update({'closed': True, 'entry': round(entry, 2), 'exit': round(px, 2), 'stopped': hit, 'pnl_pct': round(pnl * 100, 3)})
        if decision_t is not None and not any(x['coin'] == coin and x['decision_utc'] == str(decision_t) for x in state['ledger']):
            state['ledger'].append({'coin': coin, 'decision_utc': str(decision_t), 'signal': sig_row['ref_signal'], 'kind': sig_row['kind'], 'size': sig_row['size_frac'],
                                    'stop': sig_row['stop'], 'strategy_id': sig_row['strategy_id']})
    # ------------- paper output
    cum = {}
    for e in state['ledger']:
        if not e.get('closed'): continue
        cum[e['coin']] = cum.get(e['coin'], 1.0) * (1 + e['pnl_pct'] / 100)
        out['paper'].append({'date': e['decision_utc'][:10], 'coin': e['coin'], 'signal': e['signal'], 'kind': e.get('kind', ''), 'entry': e.get('entry'), 'exit': e.get('exit'),
                             'stopped': e.get('stopped', False), 'pnl_pct': e['pnl_pct'], 'cum_1x': round(cum[e['coin']], 4)})
    stamp = now.strftime('%Y-%m-%d %H:%M UTC')
    for m in log: state['log'].append({'at': stamp, 'mode': a.mode, 'text': m})
    state['log'] = state['log'][-200:]
    state['last_run'] = {'at': stamp, 'mode': a.mode}
    out['log'] = [{'at': x['at'], 'mode': x['mode'], 'text': x['text']} for x in reversed(state['log'][-40:])]
    json.dump(state, open(a.state, 'w'), ensure_ascii=False, indent=0, default=str)
    code = open(os.path.abspath(__file__), encoding='utf-8').read()
    meta = [{'label': 'signal_lab.py', 'value': code, 'type': 'code', 'lang': 'python'},
            {'label': 'Input', 'value': f'BTC: Bitstamp 1분봉(GitHub ff137) → 4시간봉 / ETH: OKX ETH-USDT-SWAP 4시간봉 / 펀딩비: 바이낸스 기록 + OKX 최신 · 실행 {stamp} ({a.mode})'}]
    for k, rows in out.items():
        json.dump({'rows': rows, 'meta': meta}, open(os.path.join(a.out, f'{k}.json'), 'w'), ensure_ascii=False, default=str)
    print(json.dumps({'signals': out['signals'], 'log': log}, ensure_ascii=False, indent=1, default=str))

if __name__ == '__main__':
    main()
