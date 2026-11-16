"""
BTC 가격 예측 모델 (변동성 GARCH-t + LightGBM 분위수 회귀)
설치:  pip install pandas numpy scipy arch lightgbm
실행:  python btc_forecast.py      -> ./btc_data 폴더에 데이터 받고 result.json 출력
데이터: Bitstamp BTC/USD 1분봉 (github.com/ff137/bitstamp-btcusd-minute-data, 매일 자동 업데이트)
"""
import os, urllib.request, pandas as pd, sys
D=os.path.abspath(sys.argv[1] if len(sys.argv)>1 else 'btc_data'); os.makedirs(D,exist_ok=True)
os.makedirs(os.path.join(D,'..','work'),exist_ok=True)
B='https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data/'
for f,u in [('hist.csv.gz','historical/btcusd_bitstamp_1min_2012-2025.csv.gz'),('latest.csv','updates/btcusd_bitstamp_1min_latest.csv')]:
    p=os.path.join(D,f)
    if f=='latest.csv' or not os.path.exists(p):
        print('download',f); urllib.request.urlretrieve(B+u,p)
m=pd.concat([pd.read_csv(os.path.join(D,'hist.csv.gz')),pd.read_csv(os.path.join(D,'latest.csv'))]).drop_duplicates('timestamp').sort_values('timestamp')
m.index=pd.to_datetime(m.timestamp,unit='s',utc=True)
m.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().to_csv(os.path.join(D,'daily.csv'))
LASTTS=m.index[-1]; del m
sys.argv=[sys.argv[0],D]
import numpy as np, json, warnings
warnings.filterwarnings('ignore')
from arch import arch_model
import lightgbm as lgb
from scipy.stats import norm
D=sys.argv[1]
df=pd.read_csv(D+'/daily.csv',index_col=0,parse_dates=True)
df=df[df.index>='2014-01-01']
LIVE=df.iloc[-1]          # partial 11/15 bar
df=df.iloc[:-1]           # full days thru 11/14
c,h,l,v=df.close,df.high,df.low,df.volume
lc=np.log(c); r=lc.diff()
F=pd.DataFrame(index=df.index)
for k in [1,3,7,14,30,90]: F[f'ret{k}']=lc-lc.shift(k)
for k in [20,50,200]: F[f'dma{k}']=lc-np.log(c.rolling(k).mean())
F['dma1400']=lc-np.log(c.rolling(1400,min_periods=700).mean())  # ~200-week
for k in [7,30,90]: F[f'vol{k}']=r.rolling(k).std()
F['volratio']=F.vol7/F.vol30
tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
F['atr14']=(tr.rolling(14).mean()/c)
d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/14).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/14).mean()
F['rsi14']=100-100/(1+up/dn)
F['dd_ath']=lc-np.log(c.cummax())
F['dd_90hi']=lc-np.log(h.rolling(90).max())
F['up_90lo']=lc-np.log(l.rolling(90).min())
F['vz']=(np.log(v+1)-np.log(v+1).rolling(30).mean())/np.log(v+1).rolling(30).std()
F['range1']=np.log(h/l)
F['dow']=df.index.dayofweek
F=F.replace([np.inf,-np.inf],np.nan)
H=[1,3,5,7]
T={}
for k in H:
    T[f'cl{k}']=lc.shift(-k)-lc
    T[f'mn{k}']=np.log(l[::-1].rolling(k).min()[::-1].shift(-1))-lc
    T[f'mx{k}']=np.log(h[::-1].rolling(k).max()[::-1].shift(-1))-lc
T=pd.DataFrame(T)

def lgbq(Xtr,ytr,Xte,q):
    m=lgb.LGBMRegressor(objective='quantile',alpha=q,n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    m.fit(Xtr,ytr); return m.predict(Xte)

def garch_sim(ret_pct,hz,n=20000,seed=0):
    am=arch_model(ret_pct,vol='GARCH',p=1,q=1,dist='t',mean='Zero')
    res=am.fit(disp='off')
    f=res.forecast(horizon=hz,method='simulation',simulations=n,reindex=False)
    sims=f.simulations.values[-1]/100   # n x hz log returns
    sig=np.sqrt(f.simulations.residual_variances[-1])/100
    return sims,sig,res

def touch_prob(sims,sig,barrier_lr,intraday=True):
    # path log level relative to start; Brownian-bridge intraday touch between closes
    lvl=np.cumsum(sims,1); prev=np.hstack([np.zeros((len(lvl),1)),lvl[:,:-1]])
    b=barrier_lr
    a=prev-b; e=lvl-b
    hit=(lvl<=b).any(1) if b<0 else (lvl>=b).any(1)
    if intraday:
        sgn=1 if b<0 else -1
        A=sgn*a; E=sgn*e
        pb=np.where((A>0)&(E>0),np.exp(-2*A*E/np.maximum(sig**2,1e-12)),1.0)
        notouch=np.prod(1-pb,1)
        return float(np.mean(np.where(hit,1.0,1-notouch)))
    return float(hit.mean())

out={}
# ---------- 1) CASE STUDY: origin 11/10 close (user's question time 11/11 03:34 KST)
O=pd.Timestamp('2026-11-10',tz='UTC'); S0=c[O]; B=62000; hz=5
blr=np.log(B/S0)
cs={'origin':str(O.date()),'S0':round(S0,2),'barrier':B,'drop_needed_pct':round((B/S0-1)*100,2),
    'actual_min':round(l[O+pd.Timedelta(days=1):O+pd.Timedelta(days=5)].min(),2)}
# a) random walk, 30d realized vol, reflection principle
s30=r[:O].tail(30).std(); cs['rw_vol30_daily_pct']=round(s30*100,2)
cs['p_rw']=round(2*norm.cdf(blr/(s30*np.sqrt(hz))),4)
# b) GARCH-t
sims,sig,res=garch_sim(r[:O].dropna().tail(1500)*100,hz)
cs['p_garch']=round(touch_prob(sims,sig,blr),4)
# c) LGBM quantiles of 5d min (trained only on data known before O)
X=F.dropna(); Y=T.loc[X.index]
trn=X.index[(X.index<=O-pd.Timedelta(days=hz))]
qs=[0.01,0.02,0.05,0.1,0.25,0.5]
qmn={q:float(lgbq(X.loc[trn],Y.loc[trn,'mn5'],X.loc[[O]],q)[0]) for q in qs}
cs['lgbm_min5_quantiles']={str(q):round(S0*np.exp(v),0) for q,v in qmn.items()}
# interpolate implied prob
qv=np.array([qmn[q] for q in qs]); cs['p_lgbm_interp']=round(float(np.interp(blr,qv,qs)),4)
# d) regime analog: price<200DMA, 30d ret<0, below 200w -> empirical P(5d min <= -15.9%)
reg=(F.dma200<0)&(F.ret30<0)
hist=pd.concat([F,T],axis=1).loc[:O-pd.Timedelta(days=hz)].dropna(subset=['mn5'])
cs['regime_now']={'below_200dma':bool(F.dma200[O]<0),'ret30_neg':bool(F.ret30[O]<0),'below_200w':bool(F.dma1400[O]<0),'rsi14':round(F.rsi14[O],1)}
for name,mask in [('all',pd.Series(True,index=hist.index)),('bear_regime',reg.loc[hist.index]),('bear+below200w',(reg&(F.dma1400<0)).loc[hist.index])]:
    sub=hist[mask]; cs[f'p_emp_{name}']=round(float((sub.mn5<=blr).mean()),4); cs[f'n_{name}']=int(len(sub))
out['case']=cs

# ---------- 2) WALK-FORWARD BACKTEST 2019-01 .. 2026-11 (yearly retrain) for 5d min q05 and 7d close quantiles
bt=[]
for yr in range(2019,2027):
    te=X.index[(X.index.year==yr)]
    te=te[te.isin(Y.dropna(subset=['mn5','cl7']).index)]
    if len(te)==0: continue
    tr_=X.index[X.index<=pd.Timestamp(f'{yr}-01-01',tz='UTC')-pd.Timedelta(days=8)]
    p={}
    for q in [0.05,0.5,0.95]: p[f'cl7_q{q}']=lgbq(X.loc[tr_],Y.loc[tr_,'cl7'],X.loc[te],q)
    p['mn5_q05']=lgbq(X.loc[tr_],Y.loc[tr_,'mn5'],X.loc[te],0.05)
    bt.append(pd.DataFrame(p,index=te))
bt=pd.concat(bt).join(Y[['mn5','cl7']])
# baseline: vol-scaled normal using vol30
z=norm.ppf(0.05); s=F.vol30.loc[bt.index]
bt['base_mn5_q05']=z*s*np.sqrt(5)*1.0
bt['base_cl7_q05']=z*s*np.sqrt(7); bt['base_cl7_q95']=-z*s*np.sqrt(7)
def pin(y,qh,q): u=y-qh; return float(np.mean(np.maximum(q*u,(q-1)*u)))
out['backtest']={
 'n_days':int(len(bt)),'period':f'{bt.index[0].date()}~{bt.index[-1].date()}',
 'mn5_q05_breach_rate_lgbm':round(float((bt.mn5<bt.mn5_q05).mean()),4),
 'mn5_q05_breach_rate_base':round(float((bt.mn5<bt.base_mn5_q05).mean()),4),
 'mn5_pinball_lgbm':round(pin(bt.mn5,bt.mn5_q05,0.05)*1e4,2),'mn5_pinball_base':round(pin(bt.mn5,bt.base_mn5_q05,0.05)*1e4,2),
 'cl7_90band_cover_lgbm':round(float(((bt.cl7>bt['cl7_q0.05'])&(bt.cl7<bt['cl7_q0.95'])).mean()),4),
 'cl7_90band_cover_base':round(float(((bt.cl7>bt.base_cl7_q05)&(bt.cl7<bt.base_cl7_q95)).mean()),4),
 'cl7_dir_acc_lgbm_median':round(float((np.sign(bt['cl7_q0.5'])==np.sign(bt.cl7)).mean()),4),
 'cl7_up_share':round(float((bt.cl7>0).mean()),4),
}
# ---------- 3) LIVE FORECAST from 11/14 close
O2=X.index[-1]; S1=c[O2]
trn2=X.index[X.index<=O2-pd.Timedelta(days=8)]
fc={'origin':str(O2.date()),'S0':round(S1,2),'live_price':round(LIVE.close,2),'live_low':round(LIVE.low,2),
    'regime':{'below_200dma':bool(F.dma200[O2]<0),'below_200w':bool(F.dma1400[O2]<0),'rsi14':round(F.rsi14[O2],1),'dd_ath_pct':round((np.exp(F.dd_ath[O2])-1)*100,1),'vol30_ann_pct':round(F.vol30[O2]*np.sqrt(365)*100,1)}}
bands={}
for k in [1,3,7]:
    qd={}
    for q in [0.05,0.25,0.5,0.75,0.95]:
        qd[str(q)]=round(S1*np.exp(lgbq(X.loc[trn2],Y.loc[trn2,f'cl{k}'],X.loc[[O2]],q)[0]),0)
    qd['min_q05']=round(S1*np.exp(lgbq(X.loc[trn2],Y.loc[trn2,f'mn{k}'],X.loc[[O2]],0.05)[0]),0)
    qd['max_q95']=round(S1*np.exp(lgbq(X.loc[trn2],Y.loc[trn2,f'mx{k}'],X.loc[[O2]],0.95)[0]),0)
    bands[k]=qd
fc['lgbm_bands']=bands
sims,sig,res=garch_sim(r.dropna().tail(1500)*100,7)
fc['garch_params']={k:round(float(v),4) for k,v in res.params.items()}
tp={}
for lv in [55000,58000,60000,62000,68000,70000,72000,75000]:
    tp[lv]=round(touch_prob(sims,sig,np.log(lv/S1)),3)
fc['garch_touch_7d']=tp
lvl=S1*np.exp(np.cumsum(sims,1))
fc['garch_close_quant']={d+1:[round(float(x),0) for x in np.percentile(lvl[:,d],[5,25,50,75,95])] for d in range(7)}
# feature importance (median 7d)
m=lgb.LGBMRegressor(objective='quantile',alpha=0.05,n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=60,verbose=-1).fit(X.loc[trn2],Y.loc[trn2,'mn5'])
fc['importance_mn5_q05']=dict(sorted({k:int(v) for k,v in zip(X.columns,m.booster_.feature_importance('gain'))}.items(),key=lambda x:-x[1])[:8])
out['forecast']=fc
# chart data: last 120d closes + case quantile + backtest series
out['chart']={'dates':[str(x.date()) for x in df.index[-90:]],'close':[round(x,0) for x in c[-90:]],'low':[round(x,0) for x in l[-90:]],
  'dma200':[round(x,0) for x in c.rolling(200).mean()[-90:]],'dma1400':[round(x,0) for x in c.rolling(1400,min_periods=700).mean()[-90:]]}
json.dump(out,open(os.path.join(D,'result.json'),'w'),indent=1,default=str)
print(json.dumps({k:v for k,v in out.items() if k!='chart'},indent=1,default=str))

# ---------- 4) DASHBOARD EXPORT (JSON {rows, meta}) ----------
import datetime as _dt
P0=float(LIVE.close)
CODE=open(os.path.abspath(__file__),encoding='utf-8').read()
META=[{'label':'btc_forecast.py','value':CODE,'type':'code','lang':'python'},
      {'label':'Input','value':'Bitstamp BTC/USD 1분봉 (github.com/ff137/bitstamp-btcusd-minute-data), 마지막 봉 '+str(LASTTS)}]
OD=os.path.join(D,'dash'); os.makedirs(OD,exist_ok=True)
def save(name,rows): json.dump({'rows':rows,'meta':META},open(os.path.join(OD,name+'.json'),'w'),ensure_ascii=False,default=str)
# GARCH from live price
simsL,sigL,_=garch_sim(r.dropna().tail(1500)*100,7,seed=1)
lvlL=P0*np.exp(np.cumsum(simsL,1))
ma200=c.rolling(200).mean(); ma200w=c.rolling(1400,min_periods=700).mean()
rows=[]
for t in df.index[-120:]:
    rows.append({'date':str(t.date()),'close':round(float(c[t]),2),'low':round(float(l[t]),2),'ma200':round(float(ma200[t]),2),'ma200w':round(float(ma200w[t]),2),
                 'q05':None,'q25':None,'q50':None,'q75':None,'q95':None})
rows.append({'date':str(LASTTS.date()),'close':round(P0,2),'low':round(float(LIVE.low),2),'ma200':None,'ma200w':None,'q05':round(P0,0),'q25':round(P0,0),'q50':round(P0,0),'q75':round(P0,0),'q95':round(P0,0)})
for d_ in range(7):
    qq=np.percentile(lvlL[:,d_],[5,25,50,75,95])
    rows.append({'date':str((LASTTS+pd.Timedelta(days=d_+1)).date()),'close':None,'low':None,'ma200':None,'ma200w':None,
                 'q05':round(qq[0],0),'q25':round(qq[1],0),'q50':round(qq[2],0),'q75':round(qq[3],0),'q95':round(qq[4],0)})
save('price',rows)
step=1000
lv=sorted({int(round(P0*(1+k)/step)*step) for k in [-0.15,-0.10,-0.07,-0.04,0.04,0.07,0.10,0.15]})
save('touch',[{'level':x,'side':'down' if x<P0 else 'up','pct_from_now':round((x/P0-1)*100,1),'prob':round(touch_prob(simsL,sigL,np.log(x/P0)),3)} for x in lv])
b7=fc['lgbm_bands'][7]; bt_=out['backtest']; g7=np.percentile(lvlL[:,6],[5,50,95])
save('summary',[{'asof':LASTTS.strftime('%Y-%m-%d %H:%M UTC'),'price':round(P0,2),'base_close':round(float(S1),2),
  'rsi14':round(float(F.rsi14[O2]),1),'dd_ath_pct':round(float((np.exp(F.dd_ath[O2])-1)*100),1),'vol30_ann_pct':round(float(F.vol30[O2]*np.sqrt(365)*100),1),
  'ma200':round(float(ma200[O2]),0),'ma200w':round(float(ma200w[O2]),0),'below_200dma':bool(F.dma200[O2]<0),'below_200w':bool(F.dma1400[O2]<0),
  'g7_q05':round(g7[0],0),'g7_q50':round(g7[1],0),'g7_q95':round(g7[2],0),
  'l7_q05':b7['0.05'],'l7_q50':b7['0.5'],'l7_q95':b7['0.95'],'l7_min_q05':b7['min_q05'],'l7_max_q95':b7['max_q95'],
  'bt_dir_acc':bt_['cl7_dir_acc_lgbm_median'],'bt_up_share':bt_['cl7_up_share'],'bt_min_breach':bt_['mn5_q05_breach_rate_lgbm'],
  'bt_base_breach':bt_['mn5_q05_breach_rate_base'],'bt_cover90':bt_['cl7_90band_cover_lgbm'],'bt_n':bt_['n_days'],'bt_period':bt_['period']}])
cs_=out['case']
save('case62k',[
 {'model':'단순 변동성(30일)','prob':cs_['p_rw'],'note':'직전 일변동성 '+str(cs_['rw_vol30_daily_pct'])+'%로 조용한 장'},
 {'model':'GARCH-t','prob':cs_['p_garch'],'note':'변동성 군집·두꺼운 꼬리 반영'},
 {'model':'LightGBM 분위수','prob':cs_['p_lgbm_interp'],'note':'62k가 하위 2~5% 구간'},
 {'model':'과거 유사 국면','prob':cs_['p_emp_bear_regime'],'note':'200일선 아래·30일 하락 국면의 과거 빈도'}])
print('dashboard files ->',OD)
