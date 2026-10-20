"""Regenerate paired controls and compressed variants of the frozen 14 videos.

Deploy beside the baseline's unchanged source snapshot. Only the metadata wire
representation changes within each fresh pair; symbols and AWGN match history.
Both arms use deterministic NTSCC+ decoding because the historical decoder has
small run-to-run rounding differences. Historical quality is diagnostic only.
This is an opt-in experiment, not a default migration.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from . import bsu_ltx_comparison as previous, ltx_receiver as ltx
from . import ntscc_plus_codec as codec, bsu_metadata_compression as compression
from .packets import accounting
from .webvid5 import fingerprint
from .webvid_ablation import Stages, environment, snapshot
from .qwen35_pairs_video import freeze
from .wire import unpack

REPO=Path(__file__).resolve().parents[2]
MODULE='semantic_transmission.bsu_metadata_e2e'
COMPLETE='COMPLETE_BSU_METADATA_E2E_QUALITY'
DEFAULT_OUTPUT=REPO/'outputs/bsu_metadata_e2e_20261019_v2'
DEFAULT_REFERENCE=REPO.parent/'semantic_transmission_bsu_ltx_plus/outputs/bsu_ltx_plus_20261019'
read=previous.read


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    write_json(temp,value);temp.replace(path)


def condition_identity(request):
    windows=[]
    for window in request['windows']:
        images=[{k:v for k,v in image.items() if k!='path'} |
                dict(sha256=request['inputs'][image['path']]) for image in window['images']]
        windows.append({k:v for k,v in window.items() if k!='images'}|dict(images=images))
    return dict(profile=request['profile'],video=request['video'],windows=windows,
                source_pixels_available=request['source_pixels_available'])


def condition_structure(request):
    value=condition_identity(request)
    for window in value['windows']:
        for image in window['images']:image.pop('sha256')
    return value


def audit_reference(reference):
    final=read(reference/'RESULT.json');old=read(reference/'protocol.json')
    original=read(reference/'reference_audit.json')
    if (final['status']!='COMPLETE_LTX_NTSCC_PLUS_COMPARISON' or final['videos']!=14
            or final['signature']!=fingerprint(old) or old['reference_identity']!=fingerprint(original)):
        raise ValueError('need the complete frozen 14-video NTSCC+ baseline')
    previous.verify_files(old['code']);previous.verify_files(old['runtime']['vendor_files'])
    old_repo=reference.parents[1]
    # Existing baseline semantics must survive deployment to the isolated tree.
    for path,expected in old['code'].items():
        try:current=REPO/Path(path).relative_to(old_repo)
        except ValueError:current=Path(path)
        if sha256(current)!=expected:raise ValueError(f'baseline implementation differs: {current}')
    files={str(reference/name):sha256(reference/name) for name in ('RESULT.json','protocol.json','reference_audit.json')}
    rows=[]
    for original_row in original['rows']:
        baseline=reference/'videos'/original_row['id'];record=read(baseline/'RESULT.json')
        if record['status']!='COMPLETE_LTX_NTSCC_PLUS_VIDEO' or record['signature']!=final['signature']:
            raise ValueError('incomplete baseline video')
        files[str(baseline/'RESULT.json')]=sha256(baseline/'RESULT.json')
        # Hash every artifact used for byte/pixel/metric comparisons, without
        # replaying generation or requiring obsolete ancestor output trees.
        for name,expected in record['artifacts'].items():
            if name.startswith(('run/transmitter/','run/received/','run/receiver/',
                    'trajectory_sender/','trajectory_received/','conditions/','raw/frames/','refined/frames/')) or name in (
                    'raw/quality.json','refined/quality.json','raw/reconstruction.mp4','refined/reconstruction.mp4'):
                files[str(baseline/name)]=expected
        cfg=baseline/'run/run_config.json';files[str(cfg)]=sha256(cfg)
        config=read(cfg)
        for k,v in dict(snr_db=10,channel_seed=42,seed=2025,visual_channel='source_frame_seeded_numpy_awgn_v1').items():
            if config.get(k)!=v:raise ValueError(f'baseline channel mismatch: {k}')
        prepared=Path(original_row['prepared']);files[str(prepared)]=sha256(prepared)
        source=read(prepared)
        source_path=Path(original_row['base'])/'reconstruction/event_pairs/run/data/normalized.mp4'
        files[str(source_path)]=record['quality']['source_sha256']
        if source['normalized_sha256']!=record['quality']['source_sha256']:
            raise ValueError('baseline quality reference differs')
        packet=(baseline/'run/transmitter/metadata.bin').read_bytes()
        if packet!=(baseline/'run/received/metadata.bin').read_bytes():raise ValueError('baseline metadata not exact')
        header,_=compression.validate_original(packet)
        request=read(baseline/'conditions/request.json')
        if condition_identity(request)['profile']!=ltx.profile():raise ValueError('LTX profile changed')
        row=dict(id=original_row['id'],frames=record['frames'],baseline=str(baseline),base=original_row['base'],
            video=header['video'],keys=[k['index'] for k in header['keyframes']],
            expected_conditions=condition_identity(request),intervals=len(request['windows']),
            baseline_quality=record['quality'],baseline_raw_quality=record['raw_quality'],
            baseline_total_channel_uses=record['total_channel_uses'])
        rows.append(row)
    previous.verify_files(files)
    return dict(reference=str(reference),files=files,rows=rows,
        baseline_protocol=old,frames=sum(r['frames'] for r in rows),intervals=sum(r['intervals'] for r in rows))


def plan(root,reference):
    from .cli import settings
    from .qwen35_pairs_comparison import forbid_overlap
    root=root.resolve();reference=reference.resolve()
    forbid_overlap(root,[reference,REPO/'src',REPO/'configs',REPO/'.local'])
    if (root/'protocol.json').exists():
        value=read(root/'protocol.json')
        if value['reference']!=str(reference):raise ValueError('reference path changed')
        verify_contract(value)
        return value
    if root.exists() and any(p.name not in ('run.log','execution.lock') for p in root.iterdir()):
        raise ValueError('unrecognized existing output')
    audit=audit_reference(reference);old=audit['baseline_protocol'];local=settings(REPO)
    assets=ltx.runtime(REPO/'.local/checkpoints/ltx25',REPO/'.local/vendor/LTX-2')
    if assets!=old['runtime'] or ltx.profile()!=old['ltx_profile']:
        raise ValueError('generator weights/source/profile differ from baseline')
    if codec.installation(REPO)!=old['codec']:raise ValueError('NTSCC+ environment changed')
    if sha256(REPO/codec.CHECKPOINT)!=codec.CHECKPOINT_SHA256:raise ValueError('NTSCC+ checkpoint changed')
    runtimes=dict(metrics=local['python'],channel=local['channel_python'],
        codec=str(REPO/'.local/ntscc_plus_env/bin/python'),ltx=str(REPO/'.local/envs/bsu-ltx25/bin/python'))
    for runtime in runtimes.values():
        if not Path(runtime).is_file():raise FileNotFoundError(runtime)
    env_file=Path(runtimes['ltx']).parent.parent/'bsu-environment.txt'
    if sha256(env_file)!=old['ltx_environment_sha256']:raise ValueError('LTX package environment changed')
    files=previous.code_identity()
    files[str(REPO/'scripts/run_bsu_metadata_e2e.sh')]=sha256(REPO/'scripts/run_bsu_metadata_e2e.sh')
    files.update(assets['vendor_files'])
    rows=[]
    for row in audit['rows']:
        for variant in ('control','compressed'):
            rows.append(dict(row,id=row['id']+'__'+variant,source_id=row['id'],variant=variant,
                compression_method='raw' if variant=='control' else 'auto',
                comparison_baseline=row['baseline'] if variant=='control' else str(root/'videos'/(row['id']+'__control'))))
    value=dict(policy='bsu_metadata_lossless_paired_e2e_v2',reference=str(reference),reference_files=audit['files'],
        rows=rows,source_videos=len(audit['rows']),frames=audit['frames'],generated_frames=2*audit['frames'],
        intervals=2*audit['intervals'],code=files,
        runtime=assets,codec=old['codec'],interpreters=runtimes,
        channel=dict(snr_db=10,seed=42,trajectory_seed=43,k=6144,n=9216,bits_per_symbol=4),
        compression='auto_min_actual_wire_bytes_with_raw_fallback',generation_seed=2025,
        decoder_numerics=dict(seed=2025,deterministic_algorithms=True,cudnn_benchmark=False,
            cudnn_deterministic=True,matmul_tf32=False,cudnn_tf32=False,cublas_workspace_config=':4096:8'),
        primary_comparison='new deterministic-decoder uncompressed control versus compressed run',
        historical_comparison='diagnostic only; baseline decoder has small repeated-run pixel variation',
        conditioning='frozen_baseline_endpoint_pair',receiver_mode='combined_context',
        sender_encoding_reused=True,visual_awgn_reexecuted_and_required_byte_exact=True,
        neural_key_decoding_reexecuted=True,ltx_generation_reexecuted=True,
        raw_and_final_quality_reexecuted=True,default_adopted=False,independent_semantic_review='PENDING')
    value['signature']=fingerprint(value)
    root.mkdir(parents=True,exist_ok=True);freeze(root/'protocol.json',value)
    for path,expected in files.items():
        try:relative=Path(path).relative_to(REPO)
        except ValueError:continue
        if '.local' in relative.parts:continue
        dest=root/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        if sha256(dest)!=expected:raise ValueError('archived source mismatch')
    freeze(root/'code_snapshot/MANIFEST.json',files)
    atomic(root/'status.json',dict(status='PLANNED_FILES_VERIFIED',stage='planned',completed=0,total=len(value['rows'])))
    return value


def verify_contract(value,reference=True):
    if fingerprint({k:v for k,v in value.items() if k!='signature'})!=value['signature']:
        raise ValueError('experiment signature changed')
    previous.verify_files(value['code'])
    if reference:previous.verify_files(value['reference_files'])


def prepare_sender(row,dest):
    base=Path(row['baseline']);run=dest/'run';tx=run/'transmitter';tx.mkdir(parents=True)
    packet=(base/'run/transmitter/metadata.bin').read_bytes()
    encoded=compression.encode(packet,row.get('compression_method','auto'))
    if compression.decode(encoded)!=packet:raise ValueError('non-lossless sender envelope')
    (tx/'metadata.bin').write_bytes(packet)
    (tx/'metadata_wire.bin').write_bytes(encoded)
    shutil.copyfile(base/'run/transmitter/visual.c64',tx/'visual.c64')
    shutil.copyfile(base/'run/run_config.json',run/'run_config.json')
    sender=dest/'trajectory_sender';sender.mkdir()
    track=base/'trajectory_sender/trajectory.bin'
    if track.exists():shutil.copyfile(track,sender/'trajectory.bin')
    write_json(sender/'sender.json',dict(packet_present=track.exists(),original_metadata_sha256=sha256(tx/'metadata.bin')))
    write_json(dest/'sender.json',dict(original_packet_sha256=sha256(tx/'metadata.bin'),
        wire_packet_sha256=sha256(tx/'metadata_wire.bin'),method=compression.method_name(encoded),
        original_packet_bytes=len(packet),wire_packet_bytes=len(encoded),
        original_representation_not_transmitted=True,source_pixels_read=False,
        reused_frozen_encoded_symbols=True))


def channel(row,dest):
    import numpy as np
    from .metadata_channel import transmit
    from .qwen35_pairs_comparison import frame_noise
    from .bsu_sequence_refinement import decode_packet
    run=dest/'run';tx=run/'transmitter';base=Path(row['baseline'])
    original=(tx/'metadata.bin').read_bytes();wire=(tx/'metadata_wire.bin').read_bytes()
    received,cost=transmit(wire,10,42)
    restored=compression.decode(received)
    if restored!=original or received!=wire or cost['bit_errors']:raise ValueError('compressed metadata channel failure')
    if any(cost[k]!=v for k,v in accounting(len(wire)).items()):raise ValueError('wire cost mismatch')
    header,_=unpack(restored);sent=np.fromfile(tx/'visual.c64',dtype='<c8')
    if len(sent)!=sum(k['complex_count'] for k in header['keyframes']) or not np.isfinite(sent).all():
        raise ValueError('invalid visual symbols')
    values=sent.copy();cursor=0
    for key in header['keyframes']:
        a,n=key['complex_offset'],key['complex_count']
        if a!=cursor:raise ValueError('noncontiguous symbols')
        values[a:a+n]+=frame_noise(key['index'],n,42,10);cursor+=n
    rx=run/'received';rx.mkdir()
    (rx/'metadata_wire.bin').write_bytes(received);(rx/'metadata.bin').write_bytes(restored)
    values.tofile(rx/'visual.c64')
    if sha256(rx/'visual.c64')!=sha256(base/'run/received/visual.c64'):
        raise ValueError('visual AWGN realization differs from baseline')
    track_rx=dest/'trajectory_received';track_rx.mkdir()
    track=dest/'trajectory_sender/trajectory.bin'
    track_report=dict(status='PASSED',packet_present=False,complex_channel_uses=0)
    if track.exists():
        track_packet=track.read_bytes();received_track,track_report=transmit(track_packet,10,43)
        if received_track!=track_packet or track_report['bit_errors']:raise ValueError('trajectory channel failure')
        decode_packet(received_track,restored)
        (track_rx/'trajectory.bin').write_bytes(received_track)
        if received_track!=(base/'trajectory_received/trajectory.bin').read_bytes():raise ValueError('trajectory changed')
        track_report.update(status='PASSED',packet_present=True,exact_match=True)
    write_json(track_rx/'channel.json',track_report)
    digital=cost['complex_channel_uses'];extra=track_report['complex_channel_uses']
    total=len(sent)+digital+extra
    if total>row['baseline_total_channel_uses']:raise ValueError('compression increases channel cost')
    cost.update(status='PASSED',original_metadata_restored_exactly=True,metadata_wire_exact_match=True,
        decoder_metadata_bytes=len(restored),wire_packet_bytes=len(wire),visual_complex_channel_uses=len(sent),
        digital_complex_channel_uses=digital,total_complex_channel_uses=len(sent)+digital,
        trajectory_complex_channel_uses=extra,total_with_trajectory=total,
        visual_awgn_byte_exact_to_baseline=True,trajectory_byte_exact_to_baseline=True,
        physical_link_overhead_included=False,original_metadata_not_counted_as_transmitted=True,
        received_files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in rx.iterdir()})
    write_json(run/'channel_accounting.json',cost)


def audit_keys_and_conditions(row,dest):
    import numpy as np
    from PIL import Image
    base=Path(row.get('comparison_baseline',row['baseline']));run=dest/'run'
    account=read(run/'receiver_accounting.json')
    if account['source_pixels_read'] or account['visual_codec']!='NTSCC+':raise ValueError('receiver contract failed')
    keys=run/'receiver/frames/sample/key_frames_received';old=base/'run/receiver/frames/sample/key_frames_received'
    exact=[];differences=[]
    for index in row['keys']:
        new_path=keys/f'{index}.png';old_path=old/f'{index}.png'
        delta=np.abs(np.asarray(Image.open(new_path),dtype=np.int16)-np.asarray(Image.open(old_path),dtype=np.int16))
        same=not delta.any()
        if same:exact.append(index)
        else:differences.append(dict(index=index,changed_values=int(np.count_nonzero(delta)),max_rgb_difference=int(delta.max())))
        if row.get('variant')!='control' and (not same or sha256(new_path)!=sha256(old_path)):
            raise ValueError(f'paired deterministic key decode changed: {index}')
    modified=dict(received_packet=str(run/'received/metadata.bin'),received_keys=str(keys),trajectory=str(dest/'trajectory_received'))
    previous.copy_conditions(modified,dest/'conditions')
    request=read(dest/'conditions/request.json')
    if row.get('variant')=='control':
        expected=read(Path(row['baseline'])/'conditions/request.json')
        if condition_structure(request)!=condition_structure(expected):raise ValueError('control changed LTX timing/profile')
        inputs_exact=condition_identity(request)==condition_identity(expected)
    else:
        expected=read(base/'conditions/request.json')
        inputs_exact=condition_identity(request)==condition_identity(expected)
        if not inputs_exact:raise ValueError('paired LTX effective inputs differ')
    write_json(dest/'condition_audit.json',dict(status='PASSED',received_keys_exact=exact,
        neural_decoding_executed=True,effective_ltx_inputs_equal_to_comparison=inputs_exact,
        comparison=str(base),historical_key_differences=differences,source_pixels_read=False))


def compare_outputs(row,dest):
    import numpy as np
    from PIL import Image
    base=Path(row.get('comparison_baseline',row['baseline']));records={}
    for kind,baseline_key in (('raw','baseline_raw_quality'),('refined','baseline_quality')):
        q=read(dest/kind/'quality.json')
        old=read(base/kind/'quality.json') if row.get('variant')=='compressed' else row[baseline_key]
        if (q['source_sha256']!=old['source_sha256'] or q['evaluation_profile']!=old['evaluation_profile']
                or any(q[b]['frames']!=row['frames'] for b in ('lossless_frames','delivered_mp4'))):
            raise ValueError('quality source, evaluator or frame coverage changed')
        new_paths=sorted((dest/kind/'frames').glob('*.png'))
        old_paths=sorted((base/kind/'frames').glob('*.png'))
        if len(new_paths)!=row['frames'] or [p.name for p in new_paths]!=[p.name for p in old_paths]:
            raise ValueError('output timeline differs')
        exact=0;max_delta=0;total_abs=0
        for new_path,old_path in zip(new_paths,old_paths):
            if sha256(new_path)==sha256(old_path):exact+=1;continue
            a=np.asarray(Image.open(new_path).convert('RGB'),dtype=np.int16)
            b=np.asarray(Image.open(old_path).convert('RGB'),dtype=np.int16)
            difference=np.abs(a-b);delta=int(difference.max())
            if delta==0:exact+=1
            max_delta=max(max_delta,delta);total_abs+=int(difference.sum())
        records[kind]=dict(quality=q,delta={boundary:{k:q[boundary][k]-old[boundary][k] for k in previous.KEYS}
            for boundary in ('lossless_frames','delivered_mp4')},
            temporal_delta={k:q['temporal'][k]-old['temporal'][k] for k in
                ('flow_error_mean','cadence_error_mean','motion_retention_ratio','false_freeze_rate')},
            exact_rgb_frames=exact,total_frames=row['frames'],max_rgb_difference=max_delta,
            mean_absolute_rgb_difference=total_abs/(row['frames']*row['video']['height']*row['video']['width']*3),
            delivered_mp4_file_exact=sha256(dest/kind/'reconstruction.mp4')==sha256(base/kind/'reconstruction.mp4'))
    write_json(dest/'comparison.json',dict(status='COMPLETE_VIDEO_COMPARISON',id=row['id'],boundaries=records,
        comparison=str(base),primary_paired_comparison=row.get('variant')=='compressed',
        neural_generation_reexecuted=True,quality_reexecuted=True,
        nondeterminism_note='Any output difference despite identical inputs must be analyzed as runtime reproducibility, not attributed directly to lossless compression.'))


def worker(root,dest,stage):
    p=read(root/'protocol.json');verify_contract(p,reference=False)
    job=read(dest/'job.json')
    if job['signature']!=p['signature']:raise ValueError('worker belongs to another protocol')
    row=job['row']
    if stage=='channel':channel(row,dest)
    elif stage=='receive':
        import torch
        torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from .codec_transport import receive
        receive(read(dest/'run/run_config.json'),REPO,dest/'run')
    elif stage=='refine':previous.refine(dest)
    elif stage.startswith('evaluate_'):
        from .bsu_sequence_refinement import evaluate
        evaluate(dict(base=row['base']),dest/stage.removeprefix('evaluate_'))
    else:raise ValueError('unknown worker stage')


def finish(root,p,results):
    if [r['id'] for r in results]!=[r['id'] for r in p['rows']]:raise ValueError('incomplete cohort')
    controls=[r for r in results if r['variant']=='control']
    compressed=[r for r in results if r['variant']=='compressed']
    if len(controls)!=len(compressed) or len(controls)!=p['source_videos']:raise ValueError('incomplete paired cohort')
    baseline=sum(r['channel']['total_with_trajectory'] for r in controls)
    total=sum(r['channel']['total_with_trajectory'] for r in compressed)
    summary={}
    for kind in ('raw','refined'):
        summary[kind]=dict(exact_rgb_frames=sum(r['comparison']['boundaries'][kind]['exact_rgb_frames'] for r in compressed),
            total_frames=p['frames'],max_rgb_difference=max(r['comparison']['boundaries'][kind]['max_rgb_difference'] for r in compressed),
            delivered_mp4_video_mean_delta={k:sum(r['comparison']['boundaries'][kind]['delta']['delivered_mp4'][k] for r in compressed)/len(compressed) for k in previous.KEYS},
            temporal_video_mean_delta={k:sum(r['comparison']['boundaries'][kind]['temporal_delta'][k] for r in compressed)/len(compressed) for k in compressed[0]['comparison']['boundaries'][kind]['temporal_delta']})
    value=dict(status=COMPLETE,signature=p['signature'],videos=p['source_videos'],variants=len(results),
        frames=p['frames'],generated_frames=p['generated_frames'],intervals=p['intervals'],
        baseline_total_channel_uses=baseline,new_total_channel_uses=total,total_saving_percent=100*(1-total/baseline),
        summary=summary,paired_input_conditions_equal=True,neural_generation_reexecuted=True,quality_reexecuted=True,
        all_final_rgb_frames_exact=summary['refined']['exact_rgb_frames']==p['frames'],
        independent_semantic_review='PENDING',default_adopted=False,
        results=[dict(id=r['id'],result=str(root/'videos'/r['id']/'RESULT.json'),
                      sha256=sha256(root/'videos'/r['id']/'RESULT.json')) for r in results])
    write_json(root/'RESULT.json',value)


def run(root,reference):
    from .etri_60s_check import lock
    p=plan(root,reference);runtimes=p['interpreters'];env=dict(environment(2025),PYTHONPATH=str(REPO/'src'))
    env.pop('LD_LIBRARY_PATH',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    old=read(root/'status.json')
    state=dict(status='RUNNING',stage='starting',completed=0,total=len(p['rows']),
        started_epoch=old.get('started_epoch',time.time()),pid=os.getpid(),
        process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21])
    results=[]
    with lock(root/'execution.lock'),lock(REPO/'.local/etri_60s_check.lock'):
        atomic(root/'status.json',state)
        try:
            for row in p['rows']:
                dest=root/'videos'/row['id'];dest.mkdir(parents=True,exist_ok=True)
                freeze(dest/'job.json',dict(row=row,signature=p['signature']))
                stages=Stages(dest,p['signature'])
                def step(name,owned,required,callback):
                    state.update(video=row['id'],stage=name,completed=len(results));atomic(root/'status.json',state)
                    stages.step(name,owned,required,callback)
                def launch(stage,runtime,log,module=MODULE,extra=None):
                    current=codec.runtime_environment(runtimes[runtime],env) if runtime=='codec' else dict(env)
                    if runtime=='codec':current['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                    if runtime=='channel':current['CUDA_VISIBLE_DEVICES']='-1'
                    argv=extra or ['worker','--output',str(root),'--video-dir',str(dest),'--stage',stage]
                    previous.command([runtimes[runtime],'-m',module,*argv],log,current)
                step('prepare',['run','trajectory_sender','sender.json'],['run/transmitter','run/run_config.json','trajectory_sender','sender.json'],
                    lambda _:prepare_sender(row,dest))
                step('channel',['run/received','run/channel_accounting.json','trajectory_received'],
                    ['run/received','run/channel_accounting.json','trajectory_received'],lambda log:launch('channel','channel',log))
                step('receive',['run/receiver','run/receiver_accounting.json'],['run/receiver','run/receiver_accounting.json'],
                    lambda log:launch('receive','codec',log))
                step('conditions',['conditions','condition_audit.json'],['conditions','condition_audit.json'],
                    lambda _:audit_keys_and_conditions(row,dest))
                step('generate',[],['raw/frames','raw/reconstruction.mp4','raw/generation.json'],
                    lambda log:launch('generate','ltx',log,'semantic_transmission.ltx_receiver',
                        ['--request',str(dest/'conditions/request.json'),'--output',str(dest/'raw'),
                         '--model-dir',str(REPO/'.local/checkpoints/ltx25'),'--vendor',str(REPO/'.local/vendor/LTX-2')]))
                step('refine',['refined'],['refined/frames','refined/reconstruction.mp4','refined/refinement.json'],
                    lambda log:launch('refine','metrics',log))
                for kind in ('raw','refined'):
                    products=[f'{kind}/quality.json',f'{kind}/quality_lossless_frames.csv',f'{kind}/quality_delivered_mp4.csv']
                    step('evaluate_'+kind,products,products,lambda log,k=kind:launch('evaluate_'+k,'metrics',log))
                step('compare',['comparison.json'],['comparison.json'],lambda _:compare_outputs(row,dest))
                result=dict(status='COMPLETE_BSU_METADATA_E2E_VIDEO',id=row['id'],source_id=row['source_id'],
                    variant=row['variant'],signature=p['signature'],
                    channel=read(dest/'run/channel_accounting.json'),comparison=read(dest/'comparison.json'),
                    artifacts=snapshot(dest,['run/transmitter','run/received','run/receiver','conditions',
                        'trajectory_sender','trajectory_received','condition_audit.json','raw','refined','comparison.json']))
                write_json(dest/'RESULT.json',result);results.append(result)
                state.update(completed=len(results));atomic(root/'status.json',state)
            verify_contract(p);finish(root,p,results)
            state.update(status=COMPLETE,stage='complete',ended_epoch=time.time());atomic(root/'status.json',state)
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                error=f'{type(error).__name__}: {error}',ended_epoch=time.time());atomic(root/'status.json',state);raise


def maybe(path):
    try:return read(path)
    except (OSError,ValueError):return {}


def status_lines(root):
    state=maybe(root/'status.json');p=maybe(root/'protocol.json');status=state.get('status','NOT_STARTED')
    if status=='RUNNING':
        try:
            if Path(f'/proc/{state["pid"]}/stat').read_text().split()[21]!=state['process_start_ticks']:
                status='STOPPED_WITHOUT_FINAL_RESULT'
        except OSError:status='STOPPED_WITHOUT_FINAL_RESULT'
    def passed(pattern):return sum(maybe(path).get('status')=='PASSED' for path in root.glob(pattern))
    def count(label,done,total):return f'{label:<12}{done}/{total or "unknown"}'+(f' ({100*done/total:.1f}%)' if total else '')
    total=len(p.get('rows',[]));elapsed=max(0,state.get('ended_epoch',time.time())-state.get('started_epoch',time.time()))
    gpu='unavailable'
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True,timeout=3).strip()
    except (OSError,subprocess.SubprocessError):pass
    return [f'Status      {status}',f'Stage       {state.get("stage","not started")} | {state.get("video","-")}',
        count('Variants',state.get('completed',0),total),count('Channel',passed('videos/*/stages/channel.json'),total),
        count('Decoding',passed('videos/*/stages/receive.json'),total),
        count('Generation',passed('videos/*/stages/generate.json'),total),
        count('Intervals',passed('videos/*/raw/stages/interval_*.json'),p.get('intervals',0)),
        count('Evaluation',passed('videos/*/stages/evaluate_*.json'),2*total),
        f'Elapsed     {int(elapsed)//3600}:{int(elapsed)%3600//60:02d}:{int(elapsed)%60:02d} | ETA unknown',
        f'GPU         utilization / used VRAM / total VRAM: {gpu}',
        'Resume      identical run command; completed stages and LTX intervals hash-verified',
        f'Log         {root/"run.log"}',f'Result      {root/"RESULT.json"}',f'Failure     {state.get("error","none recorded")}']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','status','watch','worker'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--video-dir',type=Path)
    parser.add_argument('--stage',choices=('channel','receive','refine','evaluate_raw','evaluate_refined'))
    args=parser.parse_args();root=args.output.resolve()
    if args.action=='check':
        p=plan(root,args.reference);print(json.dumps(dict(status='PLANNED_FILES_VERIFIED',videos=p['source_videos'],
            variants=len(p['rows']),frames=p['frames'],generated_frames=p['generated_frames'],intervals=p['intervals'],
            evaluations=2*len(p['rows']),neural_inference_started=False)))
    elif args.action=='worker':worker(root,args.video_dir,args.stage)
    elif args.action=='run':
        def stop(*_):raise KeyboardInterrupt('termination requested')
        signal.signal(signal.SIGTERM,stop);run(root,args.reference)
    elif args.action=='status':print('\n'.join(status_lines(root)))
    else:
        first=True
        try:
            while True:
                lines=status_lines(root)
                if not first and os.isatty(1):print(f'\033[{len(lines)}F',end='')
                for line in lines:print(('\033[2K' if os.isatty(1) else '')+line)
                first=False
                if any(s in lines[0] for s in ('COMPLETE_','FAILED','INTERRUPTED','STOPPED_')):break
                time.sleep(1)
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
