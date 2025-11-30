import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "pro_data" / "csv"
RESULTS_DIR = BASE_DIR / "pro_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 기간 설정
START_DATE = "2024-10-01"
END_DATE = "2025-10-01"

# 회사명 → 티커 매핑
COMPANY_TICKERS = {
    "마이크로소프트": "MSFT",
    "메타": "META",
    "버크셔 해서웨이": "BRK-B",
    "브로드컴": "AVGO",
    "아마존": "AMZN",
    "알파벳": "GOOGL",
    "애플": "AAPL",
    "엔비디아": "NVDA",
    "월마트": "WMT",
    "테슬라": "TSLA",
}


def get_daily_stock_data(ticker: yf.Ticker, ticker_symbol: str) -> pd.DataFrame:
    """일별 주가 데이터를 가져옵니다."""
    print("  [1/5] 일별 주가 데이터...")
    
    # 약간 더 넓은 범위로 가져오기 (시작일 데이터 확보)
    hist = yf.download(ticker_symbol, start="2024-09-25", end=END_DATE, progress=False)
    
    if hist.empty:
        return pd.DataFrame()
    
    # MultiIndex 처리
    if isinstance(hist.columns, pd.MultiIndex):
        close_prices = hist["Close"][ticker_symbol]
    else:
        close_prices = hist["Close"]
    
    df = pd.DataFrame({"주가(USD)": close_prices})
    df.index.name = "일자"
    
    # 기간 필터링
    df = df.loc[START_DATE:END_DATE]
    
    print(f"      ✅ {len(df)}일 주가 데이터 수집")
    return df


def get_quarterly_financials(ticker: yf.Ticker, company_name: str) -> pd.DataFrame:
    """분기별 재무 데이터를 수집합니다."""
    print("  [2/5] 분기별 손익계산서...")
    results = []
    
    try:
        income_stmt = ticker.quarterly_income_stmt
        
        if income_stmt.empty:
            print(f"      ⚠️ 손익계산서 데이터 없음")
            return pd.DataFrame()
        
        for col in income_stmt.columns:
            quarter_date = pd.Timestamp(col)
            quarter_data = income_stmt[col]
            
            revenue = quarter_data.get("Total Revenue", None)
            operating_income = quarter_data.get("Operating Income", None)
            net_income = quarter_data.get("Net Income", None)
            
            operating_margin = None
            if revenue and operating_income and revenue != 0:
                operating_margin = (operating_income / revenue) * 100
            
            results.append({
                "분기일자": quarter_date,
                "매출(USD)": revenue,
                "영업이익(USD)": operating_income,
                "영업이익률(%)": operating_margin,
                "순이익(USD)": net_income,
            })
        
        print(f"      ✅ {len(results)}개 분기 데이터 수집")
    
    except Exception as e:
        print(f"      ❌ 손익계산서 오류: {e}")
    
    return pd.DataFrame(results)


def get_eps_data(ticker: yf.Ticker, company_name: str) -> pd.DataFrame:
    """EPS 및 컨센서스 데이터를 수집합니다."""
    print("  [3/5] EPS 및 컨센서스...")
    results = []
    
    try:
        earnings = ticker.earnings_dates
        
        if earnings is None or earnings.empty:
            print(f"      ⚠️ EPS 데이터 없음")
            return pd.DataFrame()
        
        for idx, row in earnings.iterrows():
            try:
                quarter_date = pd.Timestamp(idx)
                if quarter_date.tzinfo is not None:
                    quarter_date = quarter_date.tz_localize(None)
                
                eps_estimate = row.get("EPS Estimate", None)
                eps_actual = row.get("Reported EPS", None)
                
                beat_consensus = None
                if pd.notna(eps_estimate) and pd.notna(eps_actual):
                    if eps_actual > eps_estimate:
                        beat_consensus = "상회"
                    elif eps_actual < eps_estimate:
                        beat_consensus = "하회"
                    else:
                        beat_consensus = "일치"
                
                surprise_pct = row.get("Surprise(%)", None)
                
                results.append({
                    "EPS발표일": quarter_date,
                    "EPS추정치": eps_estimate,
                    "EPS실적": eps_actual,
                    "컨센서스": beat_consensus,
                    "서프라이즈(%)": surprise_pct,
                })
            except Exception:
                continue
        
        print(f"      ✅ {len(results)}개 EPS 데이터 수집")
    
    except Exception as e:
        print(f"      ❌ EPS 오류: {e}")
    
    return pd.DataFrame(results)


def get_buyback_data(ticker: yf.Ticker, company_name: str) -> pd.DataFrame:
    """자사주 매입 데이터를 수집합니다."""
    print("  [4/5] 자사주 매입 데이터...")
    results = []
    
    try:
        cashflow = ticker.quarterly_cashflow
        
        if cashflow.empty:
            return pd.DataFrame()
        
        for col in cashflow.columns:
            quarter_date = pd.Timestamp(col)
            quarter_data = cashflow[col]
            
            buyback = quarter_data.get("Repurchase Of Capital Stock", None)
            dividends = quarter_data.get("Cash Dividends Paid", None)
            
            results.append({
                "분기일자": quarter_date,
                "자사주매입(USD)": buyback,
                "배당금지급(USD)": dividends,
            })
        
        print(f"      ✅ {len(results)}개 현금흐름 데이터 수집")
    
    except Exception as e:
        print(f"      ❌ 현금흐름 오류: {e}")
    
    return pd.DataFrame(results)


def calculate_trailing_eps(ticker: yf.Ticker) -> pd.DataFrame:
    """TTM EPS를 계산합니다 (최근 4분기 합계)."""
    try:
        income_stmt = ticker.quarterly_income_stmt
        info = ticker.info
        shares = info.get("sharesOutstanding", None)
        
        if income_stmt.empty or shares is None:
            return pd.DataFrame()
        
        results = []
        columns = sorted(income_stmt.columns, reverse=False)  # 오래된 순
        
        for i in range(len(columns)):
            current_date = pd.Timestamp(columns[i])
            
            # 최근 4분기 순이익 합계 (가능한 만큼)
            start_idx = max(0, i - 3)
            ttm_net_income = 0
            count = 0
            
            for j in range(start_idx, i + 1):
                net_income = income_stmt[columns[j]].get("Net Income", 0)
                if pd.notna(net_income):
                    ttm_net_income += net_income
                    count += 1
            
            # 최소 1분기 데이터가 있으면 계산 (4분기 미만이면 연환산)
            if count > 0 and shares > 0:
                # 4분기 미만이면 연환산 (annualize)
                annualized_net_income = ttm_net_income * (4 / count)
                ttm_eps = annualized_net_income / shares
                results.append({
                    "분기일자": current_date,
                    "TTM_EPS": ttm_eps,
                })
        
        return pd.DataFrame(results)
    
    except Exception:
        return pd.DataFrame()


def expand_to_daily(quarterly_df: pd.DataFrame, date_col: str, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """분기별 데이터를 일별로 확장합니다 (forward fill + backward fill)."""
    if quarterly_df.empty:
        return pd.DataFrame(index=daily_index)
    
    quarterly_df = quarterly_df.copy()
    quarterly_df[date_col] = pd.to_datetime(quarterly_df[date_col])
    quarterly_df = quarterly_df.sort_values(date_col)
    quarterly_df = quarterly_df.set_index(date_col)
    
    # 분기 데이터와 일별 인덱스 결합
    combined_index = quarterly_df.index.union(daily_index).sort_values()
    expanded = quarterly_df.reindex(combined_index)
    
    # forward fill 후 backward fill (시작일 이전 데이터가 없는 경우 대비)
    expanded = expanded.ffill().bfill()
    
    # 최종적으로 daily_index만 선택
    expanded = expanded.reindex(daily_index)
    
    return expanded


def process_company(company_name: str, ticker_symbol: str) -> None:
    """회사별 재무 데이터를 일별로 수집하고 저장합니다."""
    print(f"\n{'='*50}")
    print(f"📊 {company_name} ({ticker_symbol}) 데이터 수집")
    print(f"{'='*50}")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. 일별 주가 데이터 (기준 인덱스)
    stock_df = get_daily_stock_data(ticker, ticker_symbol)
    
    if stock_df.empty:
        print(f"  ❌ 주가 데이터 없음 - 건너뜀")
        return
    
    daily_index = stock_df.index
    
    # 2. 분기별 재무제표 → 일별 확장
    financials_df = get_quarterly_financials(ticker, company_name)
    financials_daily = expand_to_daily(financials_df, "분기일자", daily_index)
    
    # 3. EPS 데이터 → 일별 확장
    eps_df = get_eps_data(ticker, company_name)
    eps_daily = expand_to_daily(eps_df, "EPS발표일", daily_index)
    
    # 4. 자사주 매입 → 일별 확장
    buyback_df = get_buyback_data(ticker, company_name)
    buyback_daily = expand_to_daily(buyback_df, "분기일자", daily_index)
    
    # 5. TTM EPS 계산 → 일별 PER 계산
    print("  [5/5] 일별 PER 계산...")
    ttm_eps_df = calculate_trailing_eps(ticker)
    ttm_eps_daily = expand_to_daily(ttm_eps_df, "분기일자", daily_index)
    
    # 데이터 병합
    result_df = stock_df.copy()
    
    # 재무 데이터 병합
    if not financials_daily.empty:
        for col in financials_daily.columns:
            result_df[col] = financials_daily[col]
    
    # EPS 데이터 병합
    if not eps_daily.empty:
        for col in eps_daily.columns:
            result_df[col] = eps_daily[col]
    
    # 자사주 매입 데이터 병합
    if not buyback_daily.empty:
        for col in buyback_daily.columns:
            result_df[col] = buyback_daily[col]
    
    # 일별 PER 계산 (주가 / TTM EPS)
    if not ttm_eps_daily.empty and "TTM_EPS" in ttm_eps_daily.columns:
        result_df["TTM_EPS"] = ttm_eps_daily["TTM_EPS"]
        result_df["PER(일별)"] = result_df["주가(USD)"] / result_df["TTM_EPS"]
        # 음수 또는 비정상적인 PER 제거
        result_df.loc[result_df["PER(일별)"] < 0, "PER(일별)"] = np.nan
        result_df.loc[result_df["PER(일별)"] > 1000, "PER(일별)"] = np.nan
    
    # YoY 성장률 계산 (분기별로 계산된 값을 일별로 확장)
    print("  [추가] YoY 성장률 계산...")
    if not financials_df.empty:
        financials_df = financials_df.sort_values("분기일자").reset_index(drop=True)
        financials_df["매출성장률YoY(%)"] = None
        financials_df["순이익성장률YoY(%)"] = None
        
        for idx, row in financials_df.iterrows():
            if idx >= 4:  # 4분기 이전 데이터가 있어야 YoY 계산 가능
                prev_row = financials_df.iloc[idx - 4]
                
                curr_rev = row["매출(USD)"]
                prev_rev = prev_row["매출(USD)"]
                if pd.notna(curr_rev) and pd.notna(prev_rev) and prev_rev != 0:
                    try:
                        growth = ((float(curr_rev) - float(prev_rev)) / abs(float(prev_rev))) * 100
                        financials_df.at[idx, "매출성장률YoY(%)"] = growth
                    except (TypeError, ValueError):
                        pass
                
                curr_net = row["순이익(USD)"]
                prev_net = prev_row["순이익(USD)"]
                if pd.notna(curr_net) and pd.notna(prev_net) and prev_net != 0:
                    try:
                        growth = ((float(curr_net) - float(prev_net)) / abs(float(prev_net))) * 100
                        financials_df.at[idx, "순이익성장률YoY(%)"] = growth
                    except (TypeError, ValueError):
                        pass
        
        yoy_df = financials_df[["분기일자", "매출성장률YoY(%)", "순이익성장률YoY(%)"]].copy()
        yoy_daily = expand_to_daily(yoy_df, "분기일자", daily_index)
        
        if not yoy_daily.empty:
            result_df["매출성장률YoY(%)"] = yoy_daily["매출성장률YoY(%)"]
            result_df["순이익성장률YoY(%)"] = yoy_daily["순이익성장률YoY(%)"]
    
    # 인덱스를 컬럼으로
    result_df = result_df.reset_index()
    result_df["일자"] = result_df["일자"].dt.strftime("%Y-%m-%d")
    
    # 컬럼 정렬
    cols_order = [
        "일자", "주가(USD)", "PER(일별)", 
        "매출(USD)", "영업이익(USD)", "영업이익률(%)", "순이익(USD)",
        "매출성장률YoY(%)", "순이익성장률YoY(%)",
        "EPS추정치", "EPS실적", "컨센서스", "서프라이즈(%)",
        "자사주매입(USD)", "배당금지급(USD)"
    ]
    existing_cols = [c for c in cols_order if c in result_df.columns]
    other_cols = [c for c in result_df.columns if c not in cols_order]
    result_df = result_df[existing_cols + other_cols]
    
    # TTM_EPS 컬럼 제거 (중간 계산용)
    if "TTM_EPS" in result_df.columns:
        result_df = result_df.drop(columns=["TTM_EPS"])
    
    # CSV 저장
    output_path = RESULTS_DIR / f"{company_name}_재무데이터.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n  ✅ 저장 완료: {output_path}")
    print(f"  📊 {len(result_df)}일 데이터")
    print(f"\n  📋 미리보기 (처음 5일):")
    print(result_df.head().to_string(index=False))
    print(f"\n  📋 미리보기 (마지막 5일):")
    print(result_df.tail().to_string(index=False))


def main():
    print(f"\n{'#'*60}")
    print(f"#  기업 재무 데이터 수집 (일별)")
    print(f"#  기간: {START_DATE} ~ {END_DATE}")
    print(f"#  대상: {len(COMPANY_TICKERS)}개 기업")
    print(f"{'#'*60}")
    
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    companies = [f.stem for f in csv_files if not f.stem.endswith("_gemini_results")]
    
    print(f"\n📁 발견된 회사: {companies}")
    
    for company_name in companies:
        if company_name in COMPANY_TICKERS:
            ticker_symbol = COMPANY_TICKERS[company_name]
            process_company(company_name, ticker_symbol)
        else:
            print(f"\n⚠️ {company_name}: 티커 매핑 없음 (건너뜀)")
    
    print(f"\n{'#'*60}")
    print(f"#  ✅ 전체 처리 완료!")
    print(f"#  📁 저장 위치: {RESULTS_DIR}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
