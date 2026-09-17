#!/usr/bin/env python3
"""Report mandatory ERE/STA pre-score gates without inventing downstream results."""
import argparse
from collections import Counter
import json
from pathlib import Path

from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v3_validation import verify


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    root = p.parse_args().output
    protocol = verify(root)
    gate = json.loads((root / 'extractor_gate.json').read_text())
    truth = json.loads((root / 'truth_audit.json').read_text())
    counts = Counter(x['domain'] + '/' + x['split'] for x in protocol['inventory'])
    lines = ['# ERE·STA v3.1 개발 자료 재감사', '',
             f"추출기 관문: **{gate['status']}**. 정식 후보 점수 관측 수: **{gate['scores_observed']}**.", '',
             '존재·움직임 채널과 내용 의존 근거 판정을 개발 원본에서 검사했다. 관문 통과는 본 지표 성능·신규성 검증이 아니다.', '',
             '| 대상 | 개발 원본 | 재현율 | 정밀도 | 필수 여부 | 통과 |', '|---|---:|---:|---:|---|---|']
    for name, row in gate['gates'].items():
        recall = f"{row['recall']:.1%}" if row['recall'] is not None else 'null'
        precision = f"{row['precision']:.1%}" if row.get('precision') is not None else '해당 없음'
        lines.append(f"| {name} | {row['sources']} | {recall} | {precision} | {row['required']} | {row['passed']} |")
    lines += ['', f"정답 검증: 전체 {truth['total']}개 중 {truth['accepted']}개 채택, {truth['rejected']}개 제외.", '',
              '오류를 넣어도 사건 열이 바뀌지 않은 사례는 점수 계산 대상에서 제외했다. 정상군도 사건·존재 조건을 검증했다.', '',
              '## 후속 계산 상태', '',
              '- 이 실행은 prepare → truth_audit → extractor_gate의 재감사 요청 범위다. 평가 원본 후보 점수는 0건이다.',
              '- 공개 사건·객체는 사전 선언대로 진단 전용이다. STA/rules 행은 영상 재현율이 아니라 규칙 사례 6건의 일치율이다.',
              '- 필수 추출기 관문 미달이면 ERE 본 평가, OOR/HOR/ODR 비교, STA 모의 복원 귀착 정확도 및 실제 복원 진단을 실행하지 않는다.',
              '- 미실행한 AUC·검출률·귀착 정확도는 null이며 실패율 100% 또는 정확도 0%로 기록하지 않는다.',
              '- STA의 산술·근거 판정 단위 검사는 별도이며, 모의 복원 전체 실험이나 실제 채널 검증으로 해석하지 않는다.',
              '- 공개 영상의 기존 사람 주석은 이번 추가 검수 없이 정답 검증·추출기 감사에만 사용했다.', '',
              '[원본 목록·동결 정의](protocol.json) · [정답 감사](truth_audit.json) · [추출기 관문](extractor_gate.json)']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write(root / 'summary.json', {"status": 'PRE_SCORE_GATE_' + gate['status'], "sources": dict(counts),
        "truth_cases": {k: truth[k] for k in ('total', 'accepted', 'rejected', 'reasons')}, "extractor_gate": gate,
        "ere_formal_results": None, "sta_formal_results": None,
        "novelty_demonstrated": False, "new_human_review": False})


if __name__ == '__main__':
    main()
