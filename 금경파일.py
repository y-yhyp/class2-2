import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import platform

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================================
# 0. Matplotlib 한글 폰트 설정 (깨짐 방지)
# ============================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

# 시스템별 폰트 자동 설정
if platform.system() == 'Linux':
    # 스트림릿 서버(리눅스)에 설치된 나눔고딕 파
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 웹 페이지 기본 레이아웃 설정
# ============================================
st.set_page_config(page_title="AI 주가 예측 대시보드", layout="wide")

st.title("금융과 경제생활 주가 예측 대시보드")
st.write("머신러닝(XGBoost)을 활용하여 개별 종목의 과거 차트 패턴과 국가별 대표 지수, 반도체지수를 종합 분석해 **다음 거래일 주가 향방**을 예측해보았습니다.")

# ============================================
# 2. 사이드바 - 종목 선택 및 설명
# ============================================
st.sidebar.header("예측 종목 선택")

stock_options = {
    "🍎 애플 (AAPL)": {
        "symbol": "AAPL", 
        "market": "^GSPC", 
        "market_name": "S&P 500"
    },
    "🟦 삼성전자 (005930.KS)": {
        "symbol": "005930.KS", 
        "market": "^KS11", 
        "market_name": "KOSPI"
    }
}

selected_name = st.sidebar.selectbox("분석할 종목을 선택하세요", list(stock_options.keys()))
selected_info = stock_options[selected_name]

st.sidebar.markdown("---")
st.sidebar.header("모델 민감도 설정")
threshold = st.sidebar.slider(
    "상승 판단 임계값 (Threshold)", 
    min_value=0.40, max_value=0.65, value=0.50, step=0.01,
    help="AI가 계산한 상승 확률이 이 값 이상일 때만 '상승'으로 예측합니다."
)
st.sidebar.caption("**임계값이 높을수록** 더 확실한 상승 신호에만 '상승' 예측을 내립니다.")

# ============================================
# 3. 데이터 로드 및 전처리
# ============================================
@st.cache_data
def load_data(stock_symbol, market_symbol):
    stock = yf.Ticker(stock_symbol).history(start="2015-01-01")
    market = yf.Ticker(market_symbol).history(start="2015-01-01")
    sox = yf.Ticker("^SOX").history(start="2015-01-01")

    stock.index = stock.index.tz_localize(None)
    market.index = market.index.tz_localize(None)
    sox.index = sox.index.tz_localize(None)

    df = pd.DataFrame({
        "Close": stock["Close"],
        "Volume": stock["Volume"],
        "Market_Close": market["Close"],
        "SOX_Close": sox["Close"]
    }).sort_index(ascending=True)

    df = df.ffill().bfill()

    # 파생 변수
    df["Return"] = df["Close"].pct_change()
    df["Market_Return"] = df["Market_Close"].pct_change()
    df["SOX_Return"] = df["SOX_Close"].pct_change()

    df["MA5_Ratio"] = df["Close"] / df["Close"].rolling(5).mean()
    df["MA20_Ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["SOX_MA20_Ratio"] = df["SOX_Close"] / df["SOX_Close"].rolling(20).mean()

    rsi = RSIIndicator(close=df["Close"])
    df["RSI"] = rsi.rsi()

    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()

    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["Volume_Change"] = df["Volume"].pct_change()

    # Target: 다음날 종가가 더 높으면 1, 아니면 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

with st.spinner(f"'{selected_name}' 최신 데이터 수집 및 AI 모델 학습 진행 중..."):
    df = load_data(selected_info["symbol"], selected_info["market"])

# ============================================
# 4. 모델 학습 및 예측
# ============================================
features = [
    "Return", "Market_Return", "SOX_Return",
    "MA5_Ratio", "MA20_Ratio", "SOX_MA20_Ratio",
    "RSI", "MACD", "BB_Width", "Volume_Change"
]

X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

pred_prob = model.predict_proba(X_test)[:, 1]
pred_custom = (pred_prob >= threshold).astype(int)
acc_custom = accuracy_score(y_test, pred_custom)

# ============================================
# 5. 웹 대시보드 메인 화면 구성
# ============================================
st.subheader(f"{selected_name} AI 종합 예측 리포트")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 다음 거래일 AI 예측 신호")
    latest_data = X.iloc[-1:]
    latest_prob = model.predict_proba(latest_data)[0][1]

    if latest_prob >= threshold:
        st.success(f"**상승 예상** (AI 산출 상승 확률: **{latest_prob*100:.1f}%**)")
    else:
        st.error(f"**하락 예상** (AI 산출 상승 확률: **{latest_prob*100:.1f}%**)")

    st.caption(f"* 설정된 임계값(**{threshold:.2f}**)을 기준으로 예측 결과가 출력됩니다.")

with col2:
    st.markdown("### 모델 검증 정확도 (Accuracy)")
    st.metric(label="과거 20% 데이터 검증 정확도", value=f"{acc_custom*100:.2f}%")
    st.caption("* 하루 단위 주가 방향 예측의 시장 평균 정확도는 약 50~53% 수준입니다.")

# 결과 해석 가이드 박스
st.info(f"""
💡 **예측 결과 보는 법**:
* **상승 확률**: 모델이 최근 기술적 지표와 시장 지수를 고려했을 때 내일 종가가 상승할 것이라고 확신하는 정도입니다.
* **검증 정확도**: 과거 데이터 중 학습에 쓰이지 않은 최근 20% 기간 동안 AI가 실제 상승/하락 방향을 맞춘 비율입니다.
""")

st.markdown("---")

#차트 해설 하는거
col3, col4 = st.columns(2)

with col3:
    st.subheader(" 지표별 영향력 (Feature Importance)")
    
    feature_names = {
        "Return": "개별종목 당일 수익률",
        "Market_Return": f"{selected_info['market_name']} 당일 수익률",
        "SOX_Return": "필라델피아 반도체 수익률",
        "MA5_Ratio": "5일 이동평균선 대비 위치",
        "MA20_Ratio": "20일 이동평균선 대비 위치",
        "SOX_MA20_Ratio": "반도체지수 MA20 위치",
        "RSI": "RSI (매수/매도 과열지표)",
        "MACD": "MACD (추세 방향성)",
        "BB_Width": "볼린저밴드 폭 (변동성)",
        "Volume_Change": "거래량 변화율"
    }
    
    importance_series = pd.Series(
        model.feature_importances_, 
        index=[feature_names[f] for f in features]
    ).sort_values()
    
    fig, ax = plt.subplots(figsize=(7, 4.8))
    importance_series.plot(kind="barh", ax=ax, color="#2b5c8f")
    ax.set_title("각 변수가 AI 판단에 미친 비중", fontsize=11)
    ax.set_xlabel("중요도 비율")
    st.pyplot(fig)

    st.caption("막대가 길수록 AI가 내일 주가를 결정할 때 해당 지표의 신호를 많이 참고했음을 의미합니다.")

with col4:
    st.subheader("최근 100거래일 종가 흐름")
    st.line_chart(df[["Close"]].tail(100))
    st.caption(f"최근 100일간 {selected_name}의 종가(Close) 추이 그래프입니다.")

st.markdown("---")
# 설명하는거
with st.expander("**대시보드에 사용된 핵심 지표와 용어 해설 보기**"):
    st.markdown(f"""
    #### 1. 시장 지수 (Market Indices)
    * **{selected_info['market_name']}**: 해당 종목이 속한 국가 대표 주가지수로, 시장 전체의 투자 심리 흐름을 나타냅니다.
    * **필라델피아 반도체 지수 (^SOX)**: 글로벌 기술주 및 반도체 업황을 대변하는 지수로, 애플과 삼성전자 같은 빅테크 종목에 매우 유의미한 신호를 줍니다.

    #### 2. 주요 기술적 지표 (Technical Indicators)
    * **이동평균 비율 (MA5 / MA20)**: 주가가 최근 5일 또는 20일간의 평균 가격보다 위에 있는지 아래에 있는지를 나타내며, 추세 전환을 판단하는 데 쓰입니다.
    * **RSI (상대강도지수)**: 주가의 과매수(70 이상) 또는 과매도(30 이하) 상태를 판단하는 지표입니다.
    * **MACD**: 단기 이동평균선과 장기 이동평균선의 차이를 이용해 주가의 매수/매도 타이밍을 판단합니다.
    * **볼린저밴드 폭 (BB_Width)**: 주가의 변동성 크기를 의미하며, 밴드 폭이 좁아진 후에는 주가가 한 방향으로 폭발적으로 움직이는 경향이 있습니다.
    """)
