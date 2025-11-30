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
INPUT_FILE = RESULTS_DIR / "us_economic_data.csv"

# 환경 변수 로드
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# 배치 처리 설정
BATCH_SIZE = 5
BATCH_DELAY = 3


def call_gemini(input_text: str) -> str:
    """Gemini API를 호출하여 거시경제 분석을 받아옵니다."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    prompt = (
        f"""
        역할: 당신은 월스트리트의 노련한 '매크로 투자 전략가'입니다.

목표: 아래 제공된 [일자, 비트코인 가격, 미국 기준금리, 미국 GDP] 데이터를 분석하여, 주식 시장 투자 전략을 위한 핵심 인사이트를 도출해 주세요.

데이터 설명:
- 비트코인 가격: 시장 내 '유동성'과 '위험 자산 선호 심리(Risk Appetite)'의 선행 지표로 해석합니다.
- 미국 기준금리: 주식 밸류에이션(PER) 압박 요인이자 자본 조달 비용으로 해석합니다.
- 미국 GDP: 경기 침체(Recession) 여부와 기업 이익의 기초 체력으로 해석합니다.

요청 사항:
1. [상관관계 분석] 금리 변화와 GDP 추세가 비트코인(위험 자산 심리)에 미친 영향을 분석해 주세요. (예: 금리 인상기에 비트코인 가격 방어 여부 등)
2. [국면 판단] 제공된 데이터를 기반으로 현재(또는 가장 최근 데이터 시점) 경제가 다음 4가지 국면 중 어디에 해당하는지 정의해 주세요.
   - 골디락스 (성장↑, 금리 안정)
   - 스태그플레이션 (성장↓, 물가/금리↑)
   - 경기 침체 (성장↓, 금리↓)
   - 긴축 과열 (성장↑, 금리↑)
3. [주식 투자 전략] 위 국면 판단에 따라 주식 포트폴리오 전략을 제안해 주세요.
   - 비중 확대/축소 여부 (공격적 투자 vs 현금 확보)
   - 유리한 섹터 추천 (예: 금리 하락+성장 둔화 시 → 필수소비재/배당주, 금리 안정+비트코인 상승 시 → 기술주/성장주)

Input_text:
{input_text}
        """
    )

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


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 일자별로 Gemini에 전달하고 결과를 DataFrame으로 반환합니다."""
    df = df.sort_values(by="일자").reset_index(drop=True)

    print(f"[거시경제] 전체 데이터: {len(df)}일")
    print(f"[거시경제] 기간: {df['일자'].min()} ~ {df['일자'].max()}")
    print(f"[거시경제] 상위 5개 미리보기:")
    print(df.head())

    tasks = []
    for _, row in df.iterrows():
        date = row["일자"]
        data_text = (
            f"일자: {row['일자']}\n"
            f"비트코인 가격: ${row['비트코인가격(USD)']:,.2f}\n"
            f"미국 기준금리: {row['미국기준금리(%)']}%\n"
            f"미국 GDP: {row['미국GDP(십억달러)']:,.3f} 십억 달러\n"
        )
        tasks.append((date, data_text))

    print(f"\n[거시경제] 처리할 일자 수: {len(tasks)}개\n")

    results = []
    success_count = 0
    error_count = 0
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(tasks), desc="거시경제 분석 진행", unit="일자") as pbar:
        for batch_idx in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[batch_idx:batch_idx + BATCH_SIZE]
            current_batch = batch_idx // BATCH_SIZE + 1
            pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "성공": success_count, "오류": error_count})

            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                future_to_task = {
                    executor.submit(call_gemini, data_text): (date, data_text)
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

    print(f"\n[거시경제] ✅ 처리 완료! 성공 {success_count}개 / 오류 {error_count}개\n")
    return pd.DataFrame(results)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")

    # 결과 저장 폴더 생성
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📊 거시경제 데이터 Gemini 분석")
    print(f"📁 입력 파일: {INPUT_FILE}")
    print(f"{'='*60}\n")

    df = pd.read_csv(INPUT_FILE)
    result_df = process_dataframe(df)
    
    # 일자순 정렬
    result_df = result_df.sort_values(by="일자").reset_index(drop=True)
    
    output_path = RESULTS_DIR / "거시경제_gemini_results.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"{'='*60}")
    print(f"✅ 저장 완료: {output_path}")
    print(f"📊 총 {len(result_df)}일 데이터 분석 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
