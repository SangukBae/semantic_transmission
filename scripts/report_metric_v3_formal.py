#!/usr/bin/env python3
"""Render the completed frozen ERE/STA test results without changing scores."""
import argparse
from pathlib import Path

from semantic_transmission.metric_v3_formal import read, verify


def percent(value):
    return 'null' if value is None else f'{value:.1%}'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output
    verify(root); s = read(root / 'summary.json')
    text = ['# ERE·STA 고정 조건의 평가 자료 검증', '',
        f"ERE 채택 기준: **{s['ere_adoption_gate']}**, 모든 비교 지표에 대한 우위: **{s['ere_superiority_all_comparators']}**.",
        f"STA 채택 기준: **{s['sta']['adoption_gate']}**. 신규성 입증: **아직 아님**.", '',
        '합성은 같은 렌더러의 다른 seed이며 공개 영상은 진단 전용이다. STA는 PNG 패킷과 모의 복호 영상으로 검사했다.', '',
        '| 도메인·오류 | 후보 | AUC | 검출률 | 정상 오탐률 | 오류 측정률 | 12.5% 검출률 | 채택 기준 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for key, group in s['error_detection'].items():
        r = group['metrics'][group['candidate']]
        a = 'null' if r['auc'] is None else f"{r['auc']:.3f}"
        text.append(f"| {key} | {group['candidate']} | {a} | {percent(r['positive']['rate_abstentions_as_no_detection'])} | {percent(r['control']['rate_abstentions_as_no_detection'])} | {percent(r['positive']['coverage'])} | {percent(r['by_severity']['0.125']['rate_abstentions_as_no_detection'])} | {r['passed']} |")
    text += ['', '## STA 영상·표현 연결 검사', '',
        f"4종 원인 구분 정확도(클래스 평균): **{s['sta']['macro_accuracy']:.1%}**, 무훼손 오탐률: **{s['sta']['normal_false_alarm_rate']:.1%}**.", '',
        '| 성분 | 해당 원인 재현율 | 해당 조작 대 정상 AUC | 측정률 |', '|---|---:|---:|---:|']
    for name, r in s['sta']['components'].items():
        a = 'null' if r['target_vs_normal_auc'] is None else f"{r['target_vs_normal_auc']:.3f}"
        text.append(f"| {name} | {percent(s['sta']['class_recall'][name])} | {a} | {percent(r['target']['coverage'])} |")
    text += ['', '| 같은 귀착 과제를 받은 기준선 | 4분류 정확도 |', '|---|---:|']
    for name, r in s['sta']['baselines'].items():
        text.append(f"| {name} | {r['macro_accuracy']:.1%} |")
    text += ['', '우연 수준은 25%다. 기준선 분류기는 개발 자료에서만 학습한 고정 최근접 클래스 중심이다.',
        '개별 비율의 미정의 값은 null이며 0으로 바꾸지 않는다. 평가용 자료로 문턱·수식을 조정하지 않았다.', '',
        '[전체 수치·원본 bootstrap CI](summary.json) · [고정된 개발 문턱](calibration.json) · [사례별 점수](scores.csv)', '',
        '## 해석 범위', '',
        '- 관문 통과와 본 성능 기준·신규성은 별도다.',
        '- 공개 사건 추출의 낮은 정확도로 공개 결과를 주 성능으로 쓰지 않는다.',
        '- STA는 실제 바이트 복호를 실행했지만 LGVSC 생성 디코더나 실제 무선 채널 실험은 아니다.',
        '- 동작을 관측하는 모델의 오차와 정의의 한계가 점수에 포함된다. 사람의 추가 검수는 없다.']
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [k.split('/')[1] for k, g in s['error_detection'].items() if k.startswith('rendered/') and g['candidate'] == 'ere']
    values = [s['error_detection']['rendered/' + k]['metrics']['ere']['positive']['rate_abstentions_as_no_detection'] for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    axes[0].bar(names, values); axes[0].axhline(.8, color='gray', linestyle='--')
    axes[0].set(title='ERE: unseen synthetic seeds', ylabel='Detection rate', ylim=(0, 1.05)); axes[0].tick_params(axis='x', labelrotation=25)
    axes[1].bar(list(s['sta']['class_recall']), list(s['sta']['class_recall'].values()))
    axes[1].axhline(.8, color='gray', linestyle='--'); axes[1].set(title='STA: packet/raster simulator', ylabel='Correct cause classification', ylim=(0, 1.05))
    fig.savefig(root / 'comparison.png', dpi=170); fig.savefig(root / 'comparison.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
