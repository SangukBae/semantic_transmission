"""Isolated stages of the pinned SKEM+DSA baseline (no research repairs)."""
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .artifacts import sha256, write_json
from .official_release import source_path, read
from .validation_progress import emit


def prepare(request, repo, run):
    import cv2
    from .workers import load_module, csv_write
    from .video_io import probe
    data = run / 'data'
    source = data / 'source'
    source.mkdir(parents=True)
    csv_write(source / 'videos.csv', [{'path': request['input']}], ['path'])
    # Execute the pinned author's first and second pass, including its audio,
    # FPS rounding and <=16-second policy. Never touch the original source.
    movie = load_module(source_path('01_data_prep/edit_movie.py', repo), 'official_edit_movie')
    # MoviePy 1.x creates its temporary audio in cwd. Isolate that path too,
    # so preparing the same source in two output roots cannot collide.
    previous_cwd = Path.cwd()
    try:
        os.chdir(source)
        movie.process_videos(str(source), 'first_pass')
    finally:
        os.chdir(previous_cwd)
    first = source / 'first_pass' / Path(request['input']).name
    if not first.is_file():
        raise RuntimeError('Official MoviePy pass did not produce a video; see stage log')
    folder = data / '16x24'
    folder.mkdir()
    normalized = folder / 'sample.mp4'
    shutil.copyfile(first, normalized)
    subprocess.run(['bash', str(source_path('01_data_prep/resize_videoframe.sh', repo)), str(folder)], check=True)
    info = probe(normalized)
    if (info['width'], info['height'], info['fps']) != (576, 320, 24) or not 5 <= info['frames'] <= 385:
        raise ValueError(f'Unexpected official preprocessing result: {info}')
    (data / 'normalized.mp4').symlink_to('16x24/sample.mp4')
    frames = data / 'frames/sample'
    extractor = load_module(source_path('01_data_prep/get_frame.py', repo), 'official_get_frame')
    extractor.extract_frames(str(normalized), str(frames), interval=.2)
    with (frames / 'frames.csv').open() as stream:
        candidates = [int(Path(row['frame_path']).stem) for row in csv.DictReader(stream)]
    expected = list(range(0, info['frames'], 4))
    if candidates != expected:
        raise ValueError('Author candidate extraction differs from the frozen 0.2s rule')
    # Preserve all decoded source pixels for metrics/flow input lookup. The
    # selector reads ONLY the original frames.csv, so EOF is never appended.
    cap = cv2.VideoCapture(str(normalized))
    count = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if not (frames / f'{count}.png').exists() and not cv2.imwrite(str(frames / f'{count}.png'), image):
                raise RuntimeError('Cannot retain source frame')
            count += 1
    finally:
        cap.release()
    if count != info['frames']:
        raise ValueError('Decoded frame count changed during extraction')
    csv_write(folder / 'videos.csv', [{'path': str(normalized), 'frame_save_dir': str(frames)}],
              ['path', 'frame_save_dir'])
    cfg = dict(request, frames=count, max_frames=count, candidate_indices=candidates)
    write_json(run / 'run_config.json', cfg)
    write_json(run / 'prepare.json', dict(source_sha256=sha256(request['input']),
        first_pass_sha256=sha256(first), normalized_sha256=sha256(normalized), video=info,
        candidates=candidates, last_candidate=candidates[-1], actual_last_frame=count-1,
        frames_after_last_candidate=count-1-candidates[-1], actual_eof_added=False))


def select(cfg, repo, run):
    from .cli import settings
    script = run / 'release/MLM-keyframe-internvl.py'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo/'src'), str(repo/'.local/vendor/InternVL')]))
    command = [settings(repo)['internvl_python'], str(script), '--csv-path', str(run/'data/16x24/videos.csv'),
               '--method', cfg['method'].removeprefix('key_frames'), '--threshold', '.35',
               '--model_path', cfg['models']['internvl']]
    subprocess.run(command, cwd=run, env=env, check=True)
    target = run/'data/frames/sample'/cfg['method']
    keys = sorted(int(p.stem) for p in target.glob('*.png'))
    candidates = cfg['candidate_indices']
    if len(keys) < 2 or keys[0] != 0 or keys[-1] != candidates[-1] or not set(keys) <= set(candidates):
        raise ValueError('Official selector returned invalid candidate endpoints')
    write_json(run/'keyframes.json', dict(indices=keys, candidates=candidates,
        selector='author_SKEM_PSSS_0.35', precision='bf16', actual_eof_added=False))


def ntscc(cfg, repo, run):
    import logging
    import random
    import torch
    from .codec_transport import load_codec
    from .workers import load_module
    sys.path.insert(0, str(repo/'.local/vendor/NTSCC_JSAC22'))
    author = load_module(source_path('03_jscc_transmission/ntscc/main_save.py', repo), 'official_ntscc')
    torch.manual_seed(cfg['channel_seed'])
    random.seed(cfg['channel_seed'])
    model, checkpoint = load_codec(repo)
    if model.channel.chan_param != 10:
        raise ValueError('Released quality_4 baseline requires AWGN 10 dB')
    logger = logging.getLogger('official_ntscc')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    uses = []
    original = model.channel.forward
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        uses.append(int(result[1]))
        emit(len(uses), len(read(run/'keyframes.json')['indices']))
        return result
    model.channel.forward = measured
    directory = run/'data/frames/sample'/cfg['method']
    # Keep the author's glob order, full forward (including its own AWGN), PNG
    # conversion and CBR calculation. The transport wrapper is not used.
    order = [Path(p).name for p in author.glob(str(directory/'*.png'))]
    try:
        cbr = float(author.test(model, str(directory), logger, '10', save=True))
    finally:
        model.channel.forward = original
    if not math.isfinite(cbr) or len(uses) != len(order):
        raise ValueError('Incomplete official NTSCC transmission')
    keys = read(run/'keyframes.json')['indices']
    denominator = 3*cfg['width']*cfg['height']*(keys[-1]+1)
    # Same rate-index formula as the pinned author test(). These bits are
    # capacity-costed there, NOT sent through a simulated digital channel.
    rate_index_bits = (cfg['width'] * cfg['height'] / (16*16)
                       * math.log2(len(author.config.multiple_rate)) * len(keys))
    rate_index_uses = rate_index_bits / math.log2(1 + 10 ** (cfg['snr_db']/10))
    if not math.isclose(cbr * denominator, sum(uses) + rate_index_uses,
                        rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError('Author CBR does not match observed visual and rate-index components')
    write_json(run/'ntscc.json', dict(status='COMPLETE_OFFICIAL_NTSCC', checkpoint_sha256=sha256(checkpoint),
        seed=cfg['channel_seed'], snr_db=10, order=order, per_key_visual_uses=uses,
        visual_complex_channel_uses=sum(uses), official_reference_cbr=cbr,
        official_denominator=denominator, normalized_source_denominator=3*cfg['width']*cfg['height']*cfg['frames'],
        reference_cbr_includes='visual JSCC plus capacity-approximated rate-index side information',
        reference_cbr_excludes=['captions', 'flow', 'headers', 'physical link overhead'],
        rate_index_information_bits=rate_index_bits,
        rate_index_capacity_estimate_channel_uses=rate_index_uses,
        rate_index_physically_transmitted=False,
        actual_complete_transmission_cbr=None, author_training_checkpoint_identity_verified=False))


def channel(cfg, repo, run):
    import io
    from .metadata_channel import transmit
    rows = read(run/'metadata_tx.json')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['path', 'text', 'flow'])
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode('utf-8')
    received, report = transmit(payload, 10, cfg['channel_seed'], k=6144, n=9216, bits_per_symbol=4)
    output = run/'received'
    output.mkdir()
    (output/'sent.csv').write_bytes(payload)
    (output/'received.csv').write_bytes(received)
    report.update(serialization='local CSV UTF-8 adapter; author release does not specify a text packet format',
                  payload_sha256=sha256(output/'sent.csv'), exact_match=received == payload,
                  status='PASSED' if received == payload else 'FAILED')
    write_json(run/'metadata_channel.json', report)
    if received != payload:
        raise ValueError('Caption/flow LDPC decode differs from sent CSV; no fallback or retransmission')


def reconstruct(cfg, repo, run):
    from .decoder_runner import run as run_decoder
    from .codec_transport import decoder_config_text
    inputs = dict(video={k: cfg[k] for k in ('width','height','fps','frames')},
                  decoder=dict(policy='official_release', concatenation_policy='official_release', seed=42, steps=30),
                  indices=read(run/'keyframes.json')['indices'])
    receiver = run/'receiver'
    receiver.mkdir()
    write_json(receiver/'decoder_inputs.json', inputs)
    (receiver/'frames/sample').mkdir(parents=True)
    source = run/'data/frames/sample'/(cfg['method']+'_10')
    (receiver/'frames/sample/key_frames_received').symlink_to(source, target_is_directory=True)
    shutil.copyfile(run/'received/received.csv', receiver/'metadata.csv')
    config = receiver/'decoder_config.py'
    config.write_text(decoder_config_text(cfg, repo, inputs))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo/'src'), str(repo/'.local/vendor/Open-Sora')]),
               LGVSC_OFFICIAL_TRACE=str(receiver/'reference_trace.json'),
               LGVSC_OFFICIAL_MEMORY=str(receiver/'memory_execution.json'))
    run_decoder(receiver/'metadata.csv', receiver/'reconstruction', 'key_frames_received', receiver,
                decoder=run/'release/mydemo_new_align_sh.py', config=config, environment=env)


def evaluate(cfg, repo, run):
    from .video_io import probe
    from .temporal import output_source_indices
    from .research_quality import read_video
    from .official_quality import OfficialMetrics
    from .official_reporting import EVALUATION_PROFILE, evaluate_same_index
    keys = read(run/'keyframes.json')['indices']
    mapping = output_source_indices(keys, 'official_release')
    movies = list((run/'receiver/reconstruction').glob('*.mp4'))
    if len(movies) != 1:
        raise ValueError('Expected one fresh official reconstruction')
    movie = movies[0]
    info = probe(movie)
    if info['frames'] != len(mapping) or any(info[k] != cfg[k] for k in ('fps','width','height')):
        raise ValueError('Official output timeline/geometry mismatch')
    source = read_video(run/'data/normalized.mp4')
    if len(source) != cfg['frames']:
        raise ValueError('Source frame count mismatch')
    reconstructed = read_video(movie)
    if len(reconstructed) != len(mapping):
        raise ValueError('Decoded output frame count mismatch')
    summary, rows = evaluate_same_index(source, reconstructed, OfficialMetrics(), progress=emit)
    common = len(rows)
    result = dict(status='PASSED', video=str(movie), video_sha256=sha256(movie), output_frames=len(mapping),
                  source_frames=cfg['frames'], output_source_indices=mapping,
                  omitted_tail_frames=cfg['frames']-1-keys[-1],
                  evaluation_profile=EVALUATION_PROFILE,
                  evaluator_source_sha256=sha256(source_path('06_evaluation/final_score_calc_100.py', repo)),
                  evaluation='delivered MP4, same numbered frames, common prefix; author first/last-100 and video means',
                  compared_frames=common, excluded_source_frames=len(source)-common,
                  excluded_output_frames=len(reconstructed)-common,
                  comparison_pairs='original[i] versus delivered_mp4[i]; no source-index remapping',
                  official=summary,
                  repeated_boundaries_preserved=True, paper_reproduction=False)
    with (run/'quality_delivered_mp4.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run/'quality.json', result)


def worker(stage, repo, run):
    from . import workers
    cfg = read(run/('request.json' if stage == 'prepare' else 'run_config.json'))
    if stage in ('caption', 'flow'):
        return getattr(workers, stage)(cfg, repo, run)
    return globals()[stage](cfg, repo, run)
