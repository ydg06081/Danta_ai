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
CSV_DIR = BASE_DIR / "pro_data" / "csv"
RESULTS_DIR = BASE_DIR / "pro_data" / "results"

# 환경 변수 로드
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# 배치 처리 설정
BATCH_SIZE = 5
BATCH_DELAY = 3


def call_gemini(input_text: str, company_name: str) -> str:
    """Gemini API를 호출하여 응답을 받아옵니다."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    prompt = (
        f"당신은 주식 리서치 애널리스트입니다. 아래의 {company_name} 관련 오늘 뉴스들을 바탕으로, 투자 판단에 도움이 되는 인사이트를 제공해 주세요. "
        "단, 관련없는 뉴스도 있으니 잘 구분해주세요."
        "추측은 추측이라고 명확히 표시하고, 팩트와 의견을 분리해서 작성해 주세요. 과장된 낙관/비관은 피하고 객관적으로 분석해 주세요."
        f"{input_text}"
        "투자는 나도 전문가이니 나에게 투자조심문자는 하지마세요."
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


def process_dataframe(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """DataFrame을 일자별로 묶어 Gemini에 전달하고 결과를 DataFrame으로 반환합니다."""
    df = df.sort_values(by="일자").reset_index(drop=True)

    print(f"[{company_name}] 전체 데이터: {len(df)}개, 일자 수: {df['일자'].nunique()}개")
    print(f"[{company_name}] 상위 5개 미리보기:")
    print(df[["일자", "제목"]].head())

    grouped = df.groupby("일자")
    date_groups = list(grouped)
    print(f"[{company_name}] 처리할 일자 수: {len(date_groups)}개\n")

    tasks = []
    for date, group in date_groups:
        news_list = []
        for idx, row in enumerate(group.itertuples(), 1):
            news_text = f"뉴스{idx}:\n제목: {row.제목}\n본문: {row.본문}\n"
            news_list.append(news_text)
        combined_news = "\n".join(news_list)
        tasks.append((date, combined_news))

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
                    executor.submit(call_gemini, combined_news, company_name): (date, combined_news)
                    for date, combined_news in batch
                }

                for future in as_completed(future_to_task):
                    date, combined_news = future_to_task[future]
                    try:
                        gemini_response = future.result()
                        results.append(
                            {
                                "일자": date,
                                "원본내용": combined_news,
                                "답변": gemini_response,
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        results.append(
                            {
                                "일자": date,
                                "원본내용": combined_news,
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
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV 디렉터리를 찾을 수 없습니다: {CSV_DIR}")

    # 결과 저장 폴더 생성
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"처리할 CSV 파일이 없습니다: {CSV_DIR}")

    print(f"[시작] 총 {len(csv_files)}개 CSV 파일 처리 예정.")
    for csv_file in csv_files:
        print(f"\n=== {csv_file.name} 처리 시작 ===")
        df = pd.read_csv(csv_file)
        company_name = csv_file.stem
        result_df = process_dataframe(df, company_name)
        output_path = RESULTS_DIR / f"{csv_file.stem}_gemini_results.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[완료] 결과 저장 -> {output_path}")

    print("\n[전체 완료] 모든 CSV 파일 처리 종료.")


if __name__ == "__main__":
    main()
