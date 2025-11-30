from pathlib import Path
import pandas as pd


COLUMNS_TO_SAVE = ["일자", "제목", "키워드", "본문", "URL"]

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "dong" / "raw_data"
OUTPUT_DIR = BASE_DIR / "dong" / "pro_data" / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_excel_file(xlsx_path: Path) -> None:
    """주어진 엑셀 파일에서 필요한 컬럼만 추려 CSV로 저장합니다."""
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"[오류] {xlsx_path.name} 읽기 실패: {e}")
        return

    base_name = xlsx_path.stem
    selected_csv_path = OUTPUT_DIR / f"{base_name}_selected.csv"

    missing_cols = [col for col in COLUMNS_TO_SAVE if col not in df.columns]
    if missing_cols:
        print(f"[경고] {xlsx_path.name}에 필요한 컬럼이 없어 선택 컬럼 CSV를 건너뜁니다: {missing_cols}")
        return

    df[COLUMNS_TO_SAVE].to_csv(selected_csv_path, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] 선택 컬럼 -> {selected_csv_path}")


def main():
    if not RAW_DIR.exists():
        print(f"[오류] RAW 데이터 디렉터리를 찾을 수 없습니다: {RAW_DIR}")
        return

    excel_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not excel_files:
        print(f"[안내] 처리할 엑셀 파일이 없습니다: {RAW_DIR}")
        return

    print(f"[시작] 총 {len(excel_files)}개 엑셀 파일을 처리합니다.")
    for xlsx_path in excel_files:
        print(f"\n[처리 중] {xlsx_path.name}")
        process_excel_file(xlsx_path)
    print("\n[완료] 모든 엑셀 파일 처리를 마쳤습니다.")


if __name__ == "__main__":
    main()