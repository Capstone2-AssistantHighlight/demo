import json
from collections import defaultdict

def parse_timestamp(ts: str) -> int:
    h, m, s = map(int, ts.split('-'))
    return h*3600 + m*60 + s

def format_window(start_sec: int, window_size: int = 5) -> str:
    end_sec = start_sec + window_size - 1
    def fmt(sec):
        hh = sec // 3600
        mm = (sec % 3600) // 60
        ss = sec % 60
        return f"{hh:02d}-{mm:02d}-{ss:02d}"
    return f"{fmt(start_sec)}~{fmt(end_sec)}"

def rank_lines_by_window(records, window_size=5):
    # 1) 초 단위로 라인 개수 세기
    lines_per_sec = defaultdict(int)
    for r in records:
        sec = parse_timestamp(r['capture'])
        lines_per_sec[sec] += 1

    # 2) 전체 타임라인에서 5초 창마다 합산
    all_secs = sorted(lines_per_sec)
    if not all_secs:
        return []

    min_sec, max_sec = all_secs[0], all_secs[-1]
    window_stats = []
    t = min_sec
    while t <= max_sec:
        cnt = sum(lines_per_sec.get(s, 0) for s in range(t, t + window_size))
        window_stats.append({
            "window": format_window(t, window_size),
            "line_count": cnt
        })
        t += window_size

    # 3) 내림차순 정렬 후 랭크 부여
    window_stats.sort(key=lambda x: x["line_count"], reverse=True)
    for rank, w in enumerate(window_stats, start=1):
        w["rank"] = rank

    return window_stats

if __name__ == "__main__":
    # pipeline_results.json 은 중복 제거 후의 최종 레코드 리스트
    with open("pipeline_results.json", encoding="utf-8") as f:
        records = json.load(f)

    rankings = rank_lines_by_window(records, window_size=5)

    # 결과 저장
    with open("line_speed_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=2)

    print("Line-speed rankings saved to line_speed_rankings.json")