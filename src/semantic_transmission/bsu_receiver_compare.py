"""One command: copy completed WebVid artifacts, compare three RX-only policies.

Remote experiments and generators are never modified or launched. Imported
inputs, local policies, outputs and evaluation receipts have separate identities.
"""
import argparse
import contextlib
import csv
import fcntl
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from . import bsu_compare_transfer as transfer

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_receiver_compare'
ROOT = REPO/'outputs/bsu_webvid_receiver_comparison_20261013'
IDS = ['webvid_001_2255949c', 'webvid_002_1669a7b1', 'webvid_003_1ca01463', 'webvid_004_10d5cd04']
ARMS = ('current_combined', 'cadence_only', 'key_guarded')
CODE = ('bsu_receiver_compare.py', 'bsu_compare_transfer.py', 'receiver_key_guard.py',
        'guarded_trajectory_receiver.py', 'trajectory_receiver.py', 'sparse_trajectory.py',
        'bsu_sequence_refinement.py', 'official_quality.py', 'stutter_metrics.py',
        'research_quality.py', 'wire.py', 'artifacts.py', 'etri_60s_check.py')


def read(path):
    return json.loads(Path(path).read_text())


def event(**values):
    print('COMPARE_PROGRESS '+json.dumps(values, ensure_ascii=False), flush=True)


def tick(label):
    start, last, previous = time.monotonic(), -1., -1

    def update(done, total):
        nonlocal start, last, previous
        now = time.monotonic()
        if done < previous:
            start, last = now, -1.
        previous = done
        if now-last < .25 and done != total:
            return
        last = now
        elapsed = now-start
        eta = ('0초' if done == total else
               f'{elapsed/done*(total-done):.0f}초' if done and elapsed >= 2 else '미확정')
        event(stage=f'{label}: {done}/{total} ({100*done/total:.1f}%) | ETA {eta}')
    return update


@contextlib.contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'execution.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('같은 출력 폴더의 비교 명령이 이미 실행 중입니다') from None
        yield


def runtime_identity(device):
    import clip
    import DISTS_pytorch
    import lpips
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('local CUDA unavailable; --device cpu is supported but slower')
    weights = [Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth',
               Path.home()/'.cache/clip/ViT-B-32.pt',
               Path(lpips.__file__).parent/'weights/v0.1/vgg.pth',
               Path(DISTS_pytorch.__file__).parent/'weights.pt']
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(f'local metric weight required (no remote model transfer): {path}')
    return dict(device=device, python=sys.version, executable=sys.executable,
                packages={n:importlib.metadata.version(n) for n in
                          ('torch', 'torchvision', 'numpy', 'lpips', 'DISTS-pytorch', 'scikit-image')},
                cv2_version=__import__('cv2').__version__,
                clip_code_sha256=sha256(Path(clip.__file__)),
                metric_weights={str(p):sha256(p) for p in weights},
                ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0])


def freeze(root, manifest, device):
    from .receiver_key_guard import ANCHOR
    from .guarded_trajectory_receiver import GUARD
    from .trajectory_receiver import PARAMETERS
    code = {name:sha256(REPO/'src/semantic_transmission'/name) for name in CODE}
    value = dict(policy='bsu_webvid_received_three_way_v1', transfer_signature=manifest['signature'],
                 arms=list(ARMS), code=code, parameters=dict(anchor=ANCHOR, guard=GUARD, shared=PARAMETERS),
                 runtime=runtime_identity(device), primary_boundary='delivered_mp4',
                 secondary_boundary='lossless_frames', receiver_reads_source=False,
                 generator_reused=True, channel_reused=True, independent_semantic_review='PENDING',
                 candidate_scope='joint correspondence gate + RGB key preservation; not a factor-isolated ablation',
                 rate_scope='same received packet budget for primary comparison; cadence avoidable trajectory cost reported separately',
                 selection_scope='first four already inspected development videos; not an unseen evaluation',
                 visual_review_scope='generated review sheets require independent evaluation; no automatic semantic pass')
    value['signature'] = transfer.fingerprint(value)
    path = root/'protocol.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('comparison code/options/runtime changed; use a new --root')
    else:
        write_json(path, value)
        for name in CODE:
            target = root/'frozen_code'/name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(REPO/'src/semantic_transmission'/name, target)
    return value


def verify_receipt(folder, kind, signature):
    path = folder/(kind+'.json')
    if not path.exists():
        return False
    receipt = read(path)
    if receipt.get('signature') != signature or receipt.get('status') != 'PASSED':
        raise ValueError(f'incompatible receipt: {path}')
    for name, digest in receipt['artifacts'].items():
        candidate = transfer.local_file(folder, name)
        if not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError(f'completed artifact changed: {candidate}')
    return True


def receipt(folder, kind, signature, paths, **extra):
    write_json(folder/(kind+'.json'), dict(status='PASSED', signature=signature,
               artifacts={p.relative_to(folder).as_posix():sha256(p) for p in paths}, **extra))


def case_name(case):
    return f"{case['id']}/snr_{case['snr']:g}"


def receiver_inputs(mirror, case):
    """The receiver job deliberately has no source frame/movie path."""
    return dict(frames=case['frames'], fps=24, keys=case['keys'],
                raw_frames=[str(mirror/p) for p in case['raw_frames']],
                key_paths=[str(mirror/p) for p in case['key_paths']],
                base_packet=str(mirror/case['base_packet']),
                trajectory=str(mirror/case['trajectory']) if case['trajectory'] else None)


def pixels(paths, progress=None):
    import numpy as np
    from PIL import Image
    result = np.empty((len(paths), 320, 576, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as image:
            array = np.asarray(image.convert('RGB'))
        if array.shape != (320, 576, 3):
            raise ValueError(f'noncanonical image shape: {p}')
        result[i] = array
        if progress:
            progress(i+1, len(paths))
    return result


def refine(job, output, signature, arm):
    import cv2
    import numpy as np
    from PIL import Image
    from .bsu_sequence_refinement import decode_packet, validate_keys
    from .trajectory_receiver import refine_temporal
    from .guarded_trajectory_receiver import refine_references
    from .receiver_key_guard import preserve_keys
    from .wire import unpack
    from .etri_60s_check import pts_audit
    if arm not in ('cadence_only', 'key_guarded'):
        raise ValueError('current combined must be imported, not regenerated')
    allowed = {'frames', 'fps', 'keys', 'raw_frames', 'key_paths', 'base_packet', 'trajectory'}
    if set(job) != allowed:
        raise ValueError('receiver job must contain received inputs only')
    validate_keys(job['keys'], job['frames'])
    if len(job['raw_frames']) != job['frames'] or len(job['key_paths']) != len(job['keys']):
        raise ValueError('receiver input count mismatch')
    packet = Path(job['base_packet']).read_bytes()
    header, _ = unpack(packet)
    if ([k['index'] for k in header['keyframes']] != job['keys']
            or header['video'] != dict(frames=job['frames'], fps=24, width=576, height=320)):
        raise ValueError('received metadata timeline differs')
    # Cadence-only never reads trajectory bytes or key pixels.
    tracks = (decode_packet(Path(job['trajectory']).read_bytes(), packet)
              if arm == 'key_guarded' and job['trajectory'] else {})
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output/'frames'; frames_dir.mkdir(exist_ok=True)
    audits = []
    for i, (a, b) in enumerate(zip(job['keys'], job['keys'][1:])):
        window_signature = transfer.fingerprint(dict(parent=signature, arm=arm, start=a, end=b))
        window_dir = output/f'windows/{i:04d}'; window_dir.mkdir(parents=True, exist_ok=True)
        record = window_dir/'window.json'
        if record.exists():
            old = read(record)
            if old['signature'] != window_signature:
                raise ValueError('window receipt differs')
            for name, digest in old['artifacts'].items():
                if sha256(transfer.local_file(output, name)) != digest:
                    raise ValueError('completed window pixels changed')
            audits.append(old['audit'])
            event(state=f'구간 {i+1} 완료 PNG 검증·재사용')
            continue
        label = f'구간 {i+1}/{len(job["keys"])-1}'
        base = pixels(job['raw_frames'][a:b+1], tick(label+' PNG 읽기'))
        current = base
        audit = dict(start=a, end=b)
        if arm == 'key_guarded':
            key_images = [np.asarray(Image.open(p).convert('RGB')) for p in job['key_paths'][i:i+2]]
            current, audit['guard'] = refine_references(current, key_images, tracks.get((a, b)), tick(label+' 윤곽 검사'))
        current, audit['cadence'] = refine_temporal(current, tick(label+' 시간 보정'))
        if arm == 'key_guarded':
            current, audit['anchor'] = preserve_keys(current, key_images, tick(label+' 수신 키 보존'))
        elif not np.array_equal(current[[0, -1]], base[[0, -1]]):
            raise ValueError('cadence modified generated endpoints')
        saved = []
        callback = tick(label+' PNG 저장')
        for j, array in enumerate(current):
            if i and j == 0:
                previous = np.asarray(Image.open(frames_dir/f'{a:05d}.png').convert('RGB'))
                if not np.array_equal(previous, array):
                    raise ValueError('shared key boundary differs across segments')
                continue
            path = frames_dir/f'{a+j:05d}.png'
            temporary = path.with_suffix('.tmp.png')
            Image.fromarray(array).save(temporary); temporary.replace(path)
            saved.append(path); callback(j+1, len(current))
        write_json(record, dict(signature=window_signature, audit=audit,
                   artifacts={p.relative_to(output).as_posix():sha256(p) for p in saved}))
        audits.append(audit)
    paths = [frames_dir/f'{i:05d}.png' for i in range(job['frames'])]
    if len(list(frames_dir.glob('*.png'))) != len(paths):
        raise ValueError('unexpected or incomplete generated frames')
    event(stage='MP4 인코딩 (진행률 미확정)', state='실행 중')
    temporary = output/'encoding.tmp.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-nostdin', '-threads', '2', '-framerate', '24',
                    '-i', str(frames_dir/'%05d.png'), '-c:v', 'libx264', '-crf', '18',
                    '-preset', 'fast', '-pix_fmt', 'yuv420p', '-threads', '2', str(temporary)], check=True)
    audit = pts_audit(temporary, job['frames'], 24)
    movie = output/'reconstruction.mp4'; temporary.replace(movie)
    trace = output/'refinement.json'; write_json(trace, dict(arm=arm, windows=audits,
                receiver_reads_source=False, received_keys_exact=(arm == 'key_guarded'), timeline=audit))
    receipt(output, 'generation', signature, [*paths, movie, trace])


def boundary_diagnostics(source, generated, keys):
    import cv2
    import numpy as np
    gray = [cv2.cvtColor(cv2.resize(f, (288, 160)), cv2.COLOR_RGB2GRAY).astype(float) for f in generated]
    original = [cv2.cvtColor(cv2.resize(f, (288, 160)), cv2.COLOR_RGB2GRAY).astype(float) for f in source]
    rows = []
    for k in keys:
        indices = range(max(1, k-4), min(len(gray), k+5))
        errors = [float(np.mean(np.abs((gray[t]-gray[t-1])-(original[t]-original[t-1])))) for t in indices]
        rows.append(dict(key=k, maximum_change_error=max(errors), mean_change_error=float(np.mean(errors))))
    return rows


def evaluate(job, output, signature, device):
    import cv2
    import numpy as np
    from .official_quality import OfficialMetrics, KEYS
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .etri_60s_check import pts_audit
    cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=True)
    source = pixels(job['source_frames'], tick('원본 PNG 평가 입력'))
    key_images = pixels(job['key_paths'])
    if len(source) != job['frames']:
        raise ValueError('evaluation source count mismatch')
    pts_audit(Path(job['movie']), len(source), 24)
    quality = dict(arm=job['arm'], evaluation_profile='lgvsc_official_metrics_v1',
                   source_only_used_for_evaluation=True, frames=len(source),
                   neural_quality_reused=(job['arm'] == 'current_combined'))
    files = []
    for boundary in ('lossless_frames', 'delivered_mp4'):
        generated = (pixels(job['frames_paths'], tick('출력 PNG 읽기')) if boundary == 'lossless_frames'
                     else read_video(job['movie']))
        if generated.shape != source.shape:
            raise ValueError('evaluation timeline/shape mismatch')
        if job['arm'] == 'current_combined':
            summary = job['existing_quality'][boundary]
            if summary['frames'] != len(source) or any(not np.isfinite(summary[k]) for k in KEYS):
                raise ValueError('invalid imported official quality')
        else:
            event(stage=boundary+' 화질 모델 준비 (진행률 미확정)')
            summary, per_frame = OfficialMetrics(device).evaluate(source, generated, tick(boundary+' 화질 지표'))
            path = output/f'quality_{boundary}.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=per_frame[0].keys())
                writer.writeheader(); writer.writerows(per_frame)
            files.append(path)
        movement = temporal(source, generated, tick(boundary+' 시간 지표'))
        key_quality = []
        for k, key in zip(job['keys'], key_images):
            key_quality.append(dict(frame=k, received_vs_source_psnr=float(cv2.PSNR(source[k], key)),
                          output_vs_source_psnr=float(cv2.PSNR(source[k], generated[k])),
                          output_vs_received_psnr=float(cv2.PSNR(key, generated[k])),
                          exact_received_rgb=bool(np.array_equal(key, generated[k]))))
        if job['arm'] == 'key_guarded' and boundary == 'lossless_frames' and not all(k['exact_received_rgb'] for k in key_quality):
            raise ValueError('candidate did not preserve received keys')
        quality[boundary] = dict(quality=summary, temporal=movement, key_fidelity=key_quality,
                                 key_boundary_changes=boundary_diagnostics(source, generated, job['keys']))
        del generated
    path = output/'quality.json'; write_json(path, quality); files.append(path)
    receipt(output, 'evaluation', signature, files)


def arm_paths(root, mirror, case, arm):
    folder = root/'comparison'/case_name(case)/arm
    if arm == 'current_combined':
        return folder, [str(mirror/p) for p in case['current_frames']], str(mirror/case['current_movie'])
    return folder, [str(folder/f'frames/{i:05d}.png') for i in range(case['frames'])], str(folder/'reconstruction.mp4')


def worker(stage, job, output, signature, arm, device):
    command = [sys.executable, '-m', MODULE, '--worker', stage, '--job', str(job),
               '--output', str(output), '--signature', signature, '--arm', arm, '--device', device]
    subprocess.run(command, check=True)


def report(root, manifest, protocol):
    from .official_quality import KEYS
    import statistics
    rows, pending = [], []
    for case in manifest['cases']:
        for arm in ARMS:
            folder = root/'comparison'/case_name(case)/arm
            if not verify_receipt(folder, 'evaluation', protocol['signature']):
                pending.append(dict(id=case['id'], snr=case['snr'], arm=arm)); continue
            q = read(folder/'quality.json')
            item = dict(id=case['id'], snr=case['snr'], arm=arm, frames=case['frames'],
                        matched_channel_uses=case['base_channel_uses']+case['trajectory_channel_uses'],
                        standalone_channel_uses=case['base_channel_uses']+(0 if arm == 'cadence_only' else case['trajectory_channel_uses']),
                        standalone_scope='accounting counterfactual only; channel was not rerun')
            for boundary in ('lossless_frames', 'delivered_mp4'):
                item[boundary] = {k:v for k,v in q[boundary].items() if k != 'temporal'}
                item[boundary]['temporal'] = {k:v for k,v in q[boundary]['temporal'].items() if k != 'per_transition'}
            rows.append(item)
    complete = len(manifest['cases'])*len(ARMS)-len(pending)
    aggregated = []
    for snr in manifest['snrs']:
        complete_ids = [vid for vid in manifest['ids'] if all(any(r['id']==vid and r['snr']==snr and r['arm']==a for r in rows) for a in ARMS)]
        for arm in ARMS:
            matched = [r for r in rows if r['snr']==snr and r['arm']==arm and r['id'] in complete_ids]
            if not matched:
                continue
            group = dict(snr=snr, arm=arm, videos=len(matched), ids=complete_ids)
            for boundary in ('lossless_frames', 'delivered_mp4'):
                group[boundary] = dict(video_mean={k:statistics.mean(r[boundary]['quality'][k] for r in matched) for k in KEYS},
                    frame_mean={k:sum(r[boundary]['quality'][k]*r['frames'] for r in matched)/sum(r['frames'] for r in matched) for k in KEYS})
            aggregated.append(group)
    result = dict(status='COMPLETE_QUANTITATIVE_REVIEW_PENDING' if not pending else 'PARTIAL',
                  signature=protocol['signature'], completed=complete, total=len(manifest['cases'])*len(ARMS),
                  pending=pending, rows=rows, matched_cohort_aggregates=aggregated,
                  independent_semantic_review='PENDING', performance_improvement_claimed=False,
                  source_videos=len(manifest['sources']), original_lgvsc_comparison=False)
    write_json(root/'RESULT.json', result)
    flat = []
    for r in rows:
        q = r['delivered_mp4']
        flat.append(dict(id=r['id'], snr=r['snr'], arm=r['arm'], **{k:q['quality'][k] for k in KEYS},
                         flow_error=q['temporal']['flow_error_mean'], cadence_error=q['temporal']['cadence_error_mean'],
                         false_freeze=q['temporal']['false_freeze_rate'],
                         matched_channel_uses=r['matched_channel_uses'], standalone_channel_uses=r['standalone_channel_uses']))
    if flat:
        with (root/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    headers = ('id', 'snr', 'arm', *KEYS, 'flow_error', 'cadence_error', 'false_freeze')
    lines = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 수신부 3조건 비교</title>',
             '<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}video{width:24%}img{max-width:100%}</style>',
             '<h1>BSU 수신부 3조건 비교</h1>', f'<p>{result["status"]}: {complete}/{result["total"]}</p>',
             '<p>동일 생성 PNG·수신 정보의 현재 결합 / 시간 보정만 / 윤곽 검사+수신 키 보존 비교. ',
             '원본은 평가에만 사용. 생성기·채널 재실행 없음. 독립 의미 검수는 미완료이며 자동 채택하지 않습니다. ',
             'key_guarded는 두 변경의 결합 조건으로 개별 기여를 분리한 대조가 아닙니다. ',
             'PSNR·SSIM·CLIP은 높을수록, LPIPS·DISTS·광류·시간 변화 오차는 낮을수록 좋습니다.</p>',
             '<p>주 비교는 모든 조건에 같은 실제 수신 패킷 비용을 계상합니다. summary.csv의 standalone_channel_uses는 ',
             '시간 보정만 단독 배포할 경우 궤적을 생략할 수 있는 회계상 비용으로, 재전송 실험 결과가 아닙니다.</p>',
             '<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>']
    for row in flat:
        lines.append('<tr>'+''.join('<td>'+html.escape(f'{row[k]:.5f}' if isinstance(row[k], float) else str(row[k]))+'</td>' for k in headers)+'</tr>')
    lines.append('</table><p>평균은 3조건 모두 완료한 같은 영상 집합에서만 집계합니다. 원본 4개를 12개의 독립 표본으로 취급하지 않습니다.</p>')
    mirror = root/'inputs'
    for case in manifest['cases']:
        source = next(s for s in manifest['sources'] if s['id']==case['id'])
        lines.append('<h2>'+html.escape(case_name(case))+'</h2>')
        choices = [('원본', str(mirror/source['movie']))]
        for arm in ARMS:
            folder, _, movie = arm_paths(root, mirror, case, arm)
            if Path(movie).is_file():
                choices.append((arm, movie))
        for label, path in choices:
            rel = Path(path).relative_to(root).as_posix()
            lines.append(f'<a href="{html.escape(rel)}">{label}</a><video controls preload="metadata" src="{html.escape(rel)}"></video>')
        sheet = root/'review'/case_name(case)/'samples.jpg'
        if sheet.exists():
            lines.append(f'<p>균등 6시점과 키 경계 표본. 독립 의미 판정 전의 검토 자료입니다.</p><img src="{sheet.relative_to(root).as_posix()}">')
    lines.append('<p><a href="RESULT.json">전체 수치·누락 조건</a> · <a href="protocol.json">동결 조건</a></p></html>')
    (root/'report.html').write_text('\n'.join(lines))
    return result


def review_sheet(root, mirror, case, source):
    import cv2
    from PIL import Image, ImageDraw
    paths = [str(mirror/source['movie'])]+[arm_paths(root, mirror, case, arm)[2] for arm in ARMS]
    indices = sorted(set([round((case['frames']-1)*i/5) for i in range(6)]+case['keys']))
    sheet = Image.new('RGB', (4*576, len(indices)*344), 'white'); draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(zip(('source', *ARMS), paths)):
        cap = cv2.VideoCapture(path)
        for row, k in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, frame = cap.read()
            if not ok:
                raise ValueError('review frame unavailable')
            draw.text((column*576+5, row*344+3), f'{label} | frame {k} | {k/24:.3f}s', fill='black')
            sheet.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (column*576, row*344+24))
        cap.release()
    path = root/'review'/case_name(case)/'samples.jpg'; path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def request_matches(manifest, args):
    transfer.validate_manifest(manifest)
    if any(manifest[k] != v for k,v in dict(host=args.host, remote_repo=args.remote_repo,
               benchmark=args.remote_root, ids=args.ids, snrs=args.snr).items()):
        raise ValueError('frozen remote selection/options changed; use a new --root')


def run(args):
    root = args.root.resolve(); mirror = root/'inputs'
    if args.action == 'check':
        if (root/'transfer_manifest.json').exists():
            manifest = read(root/'transfer_manifest.json'); request_matches(manifest, args)
        else:
            if args.offline:
                raise ValueError('offline check needs a local transfer manifest')
            event(stage='원격 완료 조건·원본 해시 목록 확인 (진행률 미확정)')
            manifest = transfer.remote_inventory(args.host, args.remote_repo, args.remote_root, args.ids, args.snr)
        size = sum(f['size'] for f in manifest['files'].values())
        print(f'CHECK: {len(manifest["cases"])} video/SNR conditions, {len(manifest["files"])} files, {size/1024**3:.2f} GiB; no downloads', flush=True)
        runtime_identity(args.device)
        return
    with lock(root):
        if args.action in ('status', 'report'):
            manifest = read(root/'transfer_manifest.json'); request_matches(manifest, args)
            if not (root/'protocol.json').exists():
                print(f'STATUS: 비교 실행 전 | {root}', flush=True)
                return
            protocol = read(root/'protocol.json')
            result = report(root, manifest, protocol)
            print(f"STATUS: {result['status']} {result['completed']}/{result['total']} | {root/'report.html'}", flush=True)
            return
        event(overall='준비: 원격 산출물 목록 확인', current='완료된 영상·SNR만 선택', stage='목록 확인 (진행률 미확정)')
        manifest_path = root/'transfer_manifest.json'
        if manifest_path.exists():
            manifest = read(manifest_path); request_matches(manifest, args)
        else:
            if args.offline:
                raise ValueError('offline run needs a local transfer manifest')
            manifest = transfer.remote_inventory(args.host, args.remote_repo, args.remote_root, args.ids, args.snr)
            write_json(manifest_path, manifest)
        event(overall='준비: 필요한 산출물 복사·검증', stage='파일 해시 검사', state='모델·전체 데이터셋 복사 없음')
        audit = transfer.fetch(manifest, mirror, tick('입력 파일 검증'), args.offline)
        write_json(root/'transfer_receipt.json', dict(signature=manifest['signature'], status='PASSED', **audit))
        if args.action == 'fetch':
            event(overall='산출물 가져오기 완료', state='비교 실행 전')
            print(f'STATUS: {len(manifest["files"])} files verified | {root}', flush=True)
            return
        event(stage='로컬 평가 환경·가중치 및 정책 동결 (진행률 미확정)')
        protocol = freeze(root, manifest, args.device); signature = protocol['signature']
        result = report(root, manifest, protocol)
        total = len(manifest['cases'])*len(ARMS); completed = result['completed']
        try:
            for case in manifest['cases']:
                source = next(s for s in manifest['sources'] if s['id']==case['id'])
                for arm in ARMS:
                    folder, paths, movie = arm_paths(root, mirror, case, arm)
                    event(overall=f'{completed}/{total} ({100*completed/total:.1f}%) 영상·SNR·방식 평가 완료',
                          current=f'{case["id"]} / {case["snr"]:g}dB / {arm}', state='실행 중')
                    if arm != 'current_combined' and not verify_receipt(folder, 'generation', signature):
                        path = root/'jobs'/case_name(case)/(arm+'_receiver.json')
                        write_json(path, receiver_inputs(mirror, case))
                        worker('refine', path, folder, signature, arm, args.device)
                    if verify_receipt(folder, 'evaluation', signature):
                        event(state='완료 결과 해시 검증·재사용'); continue
                    imported = read(mirror/case['result_path'])
                    job = dict(arm=arm, source_frames=[str(mirror/p) for p in source['frames_paths']],
                               frames=case['frames'], keys=case['keys'], key_paths=[str(mirror/p) for p in case['key_paths']],
                               frames_paths=paths, movie=movie, existing_quality=imported['refined_quality'] if arm=='current_combined' else None)
                    path = root/'jobs'/case_name(case)/(arm+'_evaluation.json'); write_json(path, job)
                    worker('evaluate', path, folder, signature, arm, args.device)
                    result = report(root, manifest, protocol); completed = result['completed']
                review_sheet(root, mirror, case, source)
            result = report(root, manifest, protocol)
            event(overall=f'{completed}/{total} (100.0%) 영상·SNR·방식 평가 완료',
                  stage='보고서 저장 완료', state='정량 완료 / 독립 의미 검수 미완료')
            print(f'REPORT: {root/"report.html"}', flush=True)
        except BaseException as error:
            write_json(root/'last_failure.json', dict(type=type(error).__name__, error=str(error),
                       signature=signature, completed=completed, total=total, resume='repeat the same command'))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='run', choices=('run', 'fetch', 'check', 'status', 'report'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--host', default='sangukbae@155.230.15.77')
    p.add_argument('--remote-repo', default='/home/sangukbae/ETRI/Semantic/semantic_transmission')
    p.add_argument('--remote-root', default='outputs/bsu_webvid_paper')
    p.add_argument('--ids', nargs='+', default=IDS)
    p.add_argument('--snr', nargs='+', type=int, choices=(6, 8, 10), default=[6, 8, 10])
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--offline', action='store_true', help='require a verified local mirror; never contact remote')
    p.add_argument('--driver', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--worker', choices=('refine', 'evaluate'), help=argparse.SUPPRESS)
    p.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--signature', help=argparse.SUPPRESS)
    p.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.worker:
        if args.worker == 'refine':
            refine(read(args.job), args.output, args.signature, args.arm)
        else:
            evaluate(read(args.job), args.output, args.signature, args.device)
        return
    for executable in ('ssh', 'rsync', 'ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise FileNotFoundError(executable)
    root = args.root.resolve()
    if not args.driver and args.action in ('run', 'fetch', 'check'):
        ready = False
        manifest_path = root/'transfer_manifest.json'
        if manifest_path.exists():
            manifest = read(manifest_path); request_matches(manifest, args)
            if args.action == 'check' or args.offline:
                ready = True  # Checking a frozen inventory does not need SSH.
            else:
                # Password entry must own the foreground TTY, before the worker
                # display captures output. Keep its prerequisite hash scan live.
                live = sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
                callback = None
                if live:
                    def callback(done, total):
                        print(f'\r\033[2K연결 필요 여부 검사: {done}/{total} ({100*done/total:.1f}%)', end='', flush=True)
                else:
                    print('연결 필요 여부 검사: 로컬 산출물 SHA256 확인', flush=True)
                ready = not transfer.inspect_files(manifest, root/'inputs', callback)
                if live:
                    print(flush=True)
        if not ready and not args.offline:
            print('원격 산출물 읽기 연결을 준비합니다. 필요 시 SSH 비밀번호를 입력하세요.', flush=True)
            transfer.authenticate(args.host)
        from .terminal_progress import run as display
        command = [sys.executable, '-m', MODULE, *sys.argv[1:], '--driver']
        raise SystemExit(display(command, REPO/'.local/terminal_logs/bsu_receiver_compare', os.environ.get('BSU_PROGRESS', 'auto')))
    run(args)


if __name__ == '__main__':
    main()
