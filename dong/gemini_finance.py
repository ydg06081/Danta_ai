import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import time

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "pro_data" / "results"
OUTPUT_DIR = RESULTS_DIR / "finance_gemini"

# 환경 변수 로드
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# 배치 처리 설정
BATCH_SIZE = 5
BATCH_DELAY = 3

# 기간 설정
START_DATE = "2024-10-01"
END_DATE = "2025-10-01"


def call_gemini(input_text: str, company_name: str) -> str:
    """Gemini API를 호출하여 재무 데이터 분석을 받아옵니다."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    prompt = f"""
역할: 당신은 월스트리트의 노련한 '주식 애널리스트'입니다.

목표: 아래 제공된 {company_name}의 재무 데이터를 분석하여, 투자 판단에 도움이 되는 인사이트를 도출해 주세요.

데이터 설명:
- 주가(USD): 해당일 종가
- PER(일별): 주가수익비율 (주가 / TTM EPS)
- 매출(USD): 분기 매출액
- 영업이익(USD): 분기 영업이익
- 영업이익률(%): 영업이익 / 매출
- 순이익(USD): 분기 순이익
- 매출성장률YoY(%): 전년 동기 대비 매출 성장률
- 순이익성장률YoY(%): 전년 동기 대비 순이익 성장률
- EPS추정치: 애널리스트 컨센서스
- EPS실적: 실제 발표 EPS
- 컨센서스: 상회/하회/일치
- 서프라이즈(%): 실적 서프라이즈 비율
- 자사주매입(USD): 자사주 매입 금액 (음수는 매입)
- 배당금지급(USD): 배당금 지급액

요청 사항:
1. [밸류에이션 분석] PER 수준이 적정한지, 고평가/저평가 여부를 판단해 주세요.
2. [실적 분석] 매출/영업이익/순이익 성장률과 컨센서스 상회 여부를 분석해 주세요.
3. [수익성 분석] 영업이익률 수준과 추세를 분석해 주세요.
4. [주주환원 분석] 자사주 매입과 배당금 지급 현황을 분석해 주세요.
5. [투자 의견] 위 분석을 종합하여 간단한 투자 의견을 제시해 주세요.

추측은 추측이라고 명확히 표시하고, 팩트와 의견을 분리해서 작성해 주세요.
과장된 낙관/비관은 피하고 객관적으로 분석해 주세요.

재무 데이터:
{input_text}

투자는 나도 전문가이니 나에게 투자 주의 문구는 하지 마세요.
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )
    return response.text


def process_company(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """회사별 재무 데이터를 일자별로 Gemini에 전달하고 결과를 반환합니다."""
    df = df.sort_values(by="일자").reset_index(drop=True)
    
    # 기간 필터링
    df["일자"] = pd.to_datetime(df["일자"])
    df = df[(df["일자"] >= START_DATE) & (df["일자"] <= END_DATE)]
    df = df.reset_index(drop=True)

    print(f"[{company_name}] 전체 데이터: {len(df)}일")
    print(f"[{company_name}] 기간: {df['일자'].min().strftime('%Y-%m-%d')} ~ {df['일자'].max().strftime('%Y-%m-%d')}")

    def format_value(val, fmt=",.0f", prefix="$", suffix=""):
        """값 포맷팅 헬퍼 함수"""
        if pd.isna(val):
            return "N/A"
        try:
            return f"{prefix}{val:{fmt}}{suffix}"
        except (ValueError, TypeError):
            return str(val)

    tasks = []
    for _, row in df.iterrows():
        date = row["일자"].strftime("%Y-%m-%d")
        data_text = f"""
일자: {date}
주가: {format_value(row['주가(USD)'], ',.2f', '$')}
PER(일별): {format_value(row['PER(일별)'], '.2f', '')}
매출: {format_value(row['매출(USD)'], ',.0f', '$')}
영업이익: {format_value(row['영업이익(USD)'], ',.0f', '$')}
영업이익률: {format_value(row['영업이익률(%)'], '.2f', '', '%')}
순이익: {format_value(row['순이익(USD)'], ',.0f', '$')}
매출성장률YoY: {format_value(row['매출성장률YoY(%)'], '.2f', '', '%')}
순이익성장률YoY: {format_value(row['순이익성장률YoY(%)'], '.2f', '', '%')}
EPS추정치: {row['EPS추정치'] if pd.notna(row['EPS추정치']) else 'N/A'}
EPS실적: {row['EPS실적'] if pd.notna(row['EPS실적']) else 'N/A'}
컨센서스: {row['컨센서스'] if pd.notna(row['컨센서스']) else 'N/A'}
서프라이즈: {format_value(row['서프라이즈(%)'], '.2f', '', '%')}
자사주매입: {format_value(row['자사주매입(USD)'], ',.0f', '$')}
배당금지급: {format_value(row['배당금지급(USD)'], ',.0f', '$')}
"""
        tasks.append((date, data_text))

    print(f"[{company_name}] 처리할 일자 수: {len(tasks)}개\n")

    results = []
    success_count = 0
    error_count = 0
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(tasks), desc=f"{company_name} 진행", unit="일자") as pbar:
        for batch_idx in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[batch_idx:batch_idx + BATCH_SIZE]
            current_batch = batch_idx // BATCH_SIZE + 1
            pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "성공": success_count, "오류": error_count})

            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                future_to_task = {
                    executor.submit(call_gemini, data_text, company_name): (date, data_text)
                    for date, data_text in batch
                }

                for future in as_completed(future_to_task):
                    date, data_text = future_to_task[future]
                    try:
                        gemini_response = future.result()
                        results.append(
                            {
                                "일자": date,
                                "원본내용": data_text,
                                "답변": gemini_response,
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        results.append(
                            {
                                "일자": date,
                                "원본내용": data_text,
                                "답변": f"오류: {str(e)}",
                            }
                        )
                    pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "성공": success_count, "오류": error_count})
                    pbar.update(1)

            if batch_idx + BATCH_SIZE < len(tasks):
                time.sleep(BATCH_DELAY)

    print(f"\n[{company_name}] ✅ 처리 완료! 성공 {success_count}개 / 오류 {error_count}개\n")
    return pd.DataFrame(results)


def main():
    # 결과 저장 폴더 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 재무데이터 파일 찾기
    finance_files = sorted(RESULTS_DIR.glob("*_재무데이터.csv"))
    
    if not finance_files:
        raise FileNotFoundError(f"재무데이터 파일이 없습니다: {RESULTS_DIR}")

    print(f"\n{'#'*60}")
    print(f"#  재무 데이터 Gemini 분석")
    print(f"#  기간: {START_DATE} ~ {END_DATE}")
    print(f"#  대상: {len(finance_files)}개 기업")
    print(f"{'#'*60}\n")

    print(f"📁 발견된 재무데이터 파일:")
    for f in finance_files:
        print(f"   - {f.name}")
    print()

    for finance_file in finance_files:
        company_name = finance_file.stem.replace("_재무데이터", "")
        print(f"\n{'='*60}")
        print(f"📊 {company_name} 재무 데이터 분석 시작")
        print(f"{'='*60}")
        
        df = pd.read_csv(finance_file)
        result_df = process_company(df, company_name)
        
        # 일자순 정렬
        result_df = result_df.sort_values(by="일자").reset_index(drop=True)
        
        output_path = OUTPUT_DIR / f"{company_name}_재무분석_gemini_results.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[완료] 결과 저장 -> {output_path}")

    print(f"\n{'#'*60}")
    print(f"#  ✅ 전체 처리 완료!")
    print(f"#  📁 저장 위치: {OUTPUT_DIR}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
