# coin-signal-lab

BTC·ETH 무기한선물 롱/숏/관망 신호 실험실. 결과는 Claude 대시보드 "BTC·ETH 선물 레이더"에 매일 올라갑니다.

- `collect.py` — ETH 4시간봉, BTC·ETH 펀딩비 최신분 수집 (OKX → 실패 시 Kraken/Hyperliquid)
- `btc_forecast.py` — BTC 7일 가격 범위(GARCH-t, LightGBM)
- `signal_lab.py` — 후보 전략 백테스트(수수료·슬리피지·펀딩 반영), Deflated Sharpe 검증, 챔피언 선택, 모의매매 채점
- `data/` — ETH 4시간봉, 펀딩비, 상태(state.json: 후보 전략·챔피언·모의매매 기록)
- `out/` — 매 실행 결과 JSON
- 실행: `tick.txt` 또는 `candidates_add.json`이 바뀐 커밋이 이 브랜치에 푸시되면 GitHub Actions가 실행. 화·금(UTC)과 새 후보가 있을 때는 개선 모드.

투자 조언이 아닌 연구용 도구입니다.
