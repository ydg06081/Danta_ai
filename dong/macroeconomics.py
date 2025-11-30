import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path
from datetime import datetime

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "pro_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 기간 설정
START_DATE = "2024-10-01"
END_DATE = "2025-10-01"


def get_bitcoin_price(start: str, end: str) -> pd.DataFrame:
    """비트코인 가격 데이터를 가져옵니다."""
    print("[1/3] 비트코인 가격 데이터 수집 중...")
    btc = yf.download("BTC-USD", start=start, end=end, progress=False)
    # MultiIndex 컬럼 처리
    if isinstance(btc.columns, pd.MultiIndex):
        btc = btc["Close"]["BTC-USD"].to_frame(name="비트코인가격(USD)")
    else:
        btc = btc[["Close"]].rename(columns={"Close": "비트코인가격(USD)"})
    btc.index.name = "일자"
    print(f"  ✅ 비트코인 데이터 {len(btc)}일치 수집 완료")
    return btc


def get_fed_funds_rate(start: str, end: str) -> pd.DataFrame:
    """미국 연방기금금리(기준금리) 데이터를 가져옵니다."""
    print("[2/3] 미국 기준금리 데이터 수집 중...")
    # DFF: Daily Federal Funds Effective Rate (일별 실효 연방기금금리)
    fed_rate = pdr.DataReader("DFF", "fred", start, end)
    fed_rate = fed_rate.rename(columns={"DFF": "미국기준금리(%)"})
    fed_rate.index.name = "일자"
    print(f"  ✅ 기준금리 데이터 {len(fed_rate)}일치 수집 완료")
    return fed_rate


def get_us_gdp(start: str, end: str) -> pd.DataFrame:
    """미국 GDP 데이터를 가져옵니다. (분기별 데이터를 일별로 확장)"""
    print("[3/3] 미국 GDP 데이터 수집 중...")
    # GDP: Gross Domestic Product (분기별, 십억 달러)
    # 분기 데이터이므로 더 넓은 범위에서 가져와서 reindex
    gdp = pdr.DataReader("GDP", "fred", "2024-01-01", end)
    gdp = gdp.rename(columns={"GDP": "미국GDP(십억달러)"})
    
    # 일별 인덱스 생성 후 forward fill
    date_range = pd.date_range(start=start, end=end, freq="D")
    gdp = gdp.reindex(date_range, method="ffill")
    gdp.index.name = "일자"
    print(f"  ✅ GDP 데이터 수집 완료 (분기별 → 일별 확장)")
    return gdp


def main():
    print(f"\n{'='*50}")
    print(f"📊 미국 경제 데이터 수집")
    print(f"📅 기간: {START_DATE} ~ {END_DATE}")
    print(f"{'='*50}\n")

    # 데이터 수집
    btc_df = get_bitcoin_price(START_DATE, END_DATE)
    fed_df = get_fed_funds_rate(START_DATE, END_DATE)
    gdp_df = get_us_gdp(START_DATE, END_DATE)

    # 데이터 병합 (일자 기준)
    print("\n[병합] 데이터 통합 중...")
    merged_df = btc_df.join(fed_df, how="outer").join(gdp_df, how="outer")
    
    # 결측치 처리 (주말/공휴일은 forward fill)
    merged_df = merged_df.ffill()
    
    # 인덱스를 컬럼으로 변환
    merged_df = merged_df.reset_index()
    merged_df["일자"] = merged_df["일자"].dt.strftime("%Y-%m-%d")

    # CSV 저장
    output_path = RESULTS_DIR / "us_economic_data.csv"
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*50}")
    print(f"✅ 저장 완료: {output_path}")
    print(f"📊 총 {len(merged_df)}일 데이터")
    print(f"{'='*50}")
    
    # 미리보기
    print("\n📋 데이터 미리보기:")
    print(merged_df.head(10).to_string(index=False))
    print("\n...")
    print(merged_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

