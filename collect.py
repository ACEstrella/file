"""
collect.py — GitHub Actions에서 매일 ETH 4시간봉과 BTC·ETH 펀딩비 최신분을 받아 OKX 형식 JSON으로 저장.
1순위 OKX, 실패하면 ETH 봉은 Kraken(현물), 펀딩비는 Hyperliquid(1시간 펀딩을 8시간 단위로 합산)로 대체.
출력: data/new/eth.json  {"source": ..., "data": [[ts,o,h,l,c,vol,volccy,volquote,confirm], ...]}
      data/new/fund_btc.json, data/new/fund_eth.json  {"source": ..., "data": [{"instId","fundingTime","fundingRate"}, ...]}
"""
import json, os, time, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'new')
os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'coin-signal-lab/1.0'}

def get(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={**UA, **({'Content-Type': 'application/json'} if data else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), 'w'))
    print('saved', name, obj['source'], len(obj['data']))

# ---------------- ETH 4h candles
eth = None
try:
    j = get('https://www.okx.com/api/v5/market/candles?instId=ETH-USDT-SWAP&bar=4H&limit=60')
    if j.get('code') == '0' and j['data']:
        eth = {'source': 'okx', 'data': j['data']}
except Exception as ex:
    print('okx candles failed:', ex)
if eth is None:
    try:
        j = get('https://api.kraken.com/0/public/OHLC?pair=ETHUSD&interval=240')
        key = [k for k in j['result'] if k != 'last'][0]
        rows = j['result'][key][-60:]
        data = []
        for i, r in enumerate(rows):
            confirm = '0' if i == len(rows) - 1 else '1'      # last bar is still forming
            data.append([str(int(r[0]) * 1000), r[1], r[2], r[3], r[4], r[6], r[6], '0', confirm])
        eth = {'source': 'kraken-spot', 'data': list(reversed(data))}
    except Exception as ex:
        print('kraken candles failed:', ex)
if eth: save('eth.json', eth)

# ---------------- funding
def okx_fund(inst):
    j = get(f'https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=30')
    if j.get('code') != '0' or not j['data']:
        raise RuntimeError(j.get('msg'))
    return [{'instId': r['instId'], 'fundingTime': r['fundingTime'], 'fundingRate': r.get('realizedRate') or r['fundingRate']} for r in j['data']]

def hl_fund(coin):
    start = int((time.time() - 10 * 86400) * 1000)
    rows = get('https://api.hyperliquid.xyz/info', {'type': 'fundingHistory', 'coin': coin, 'startTime': start})
    buckets = {}
    for r in rows:
        t = int(r['time']) // 1000
        b = ((t // 3600) // 8 + 1) * 8 * 3600                 # 8h boundary that closes this hour
        buckets.setdefault(b, []).append(float(r['fundingRate']))
    now = time.time()
    out = [{'instId': f'{coin}-USDT-SWAP', 'fundingTime': str(b * 1000), 'fundingRate': str(sum(v))}
           for b, v in sorted(buckets.items()) if len(v) >= 7 and b <= now]
    return out

for coin in ['BTC', 'ETH']:
    rows, src = None, None
    try:
        rows, src = okx_fund(f'{coin}-USDT-SWAP'), 'okx'
    except Exception as ex:
        print('okx funding failed:', coin, ex)
        try:
            rows, src = hl_fund(coin), 'hyperliquid'
        except Exception as ex2:
            print('hyperliquid funding failed:', coin, ex2)
    if rows: save(f'fund_{coin.lower()}.json', {'source': src, 'data': rows})
