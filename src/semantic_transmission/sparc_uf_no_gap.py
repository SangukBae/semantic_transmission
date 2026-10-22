"""UF QP30 with source-change auxiliary selection and no temporal spacing cap."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from . import bsu_predict as base, sparc, sparc_selection as selection
from . import sparc_semantics, sparc_uf as pilot, sparc_uf_rate as rate
from . import sparc_detail_selection as detail
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages, exclusive_lock

REPO = sparc.REPO
COMPLETE = 'COMPLETE_SPARC_UF_NO_GAP'


def preflight(args):
    if type(args.qp) is not int or not 0 <= args.qp <= 63:
        raise ValueError('QP must be 0..63')
    ref, root = args.reference.resolve(strict=True), args.output.resolve()
    source, old_plan = rate.audit_reference(ref)
    for protected in (ref, Path(source['directory']).resolve()):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('output overlaps reference or source')
    config = sparc.profile(no_max_gap=True)
    selector = getattr(args,'selector','coarse')
    if selector not in ('coarse','detail'):
        raise ValueError('unsupported source-only selector')
    selector_config = detail.config(getattr(args,'detail_scale',1.)) if selector == 'detail' else None
    semantic = sparc_semantics.validate_manifest(ref/'semantic.json',source,config)
    if semantic['keys'] != old_plan['semantic_keys']:
        raise ValueError('reference semantic keys disagree')
    old = base.read(ref/'protocol.json')
    runtimes = dict(codec=str(REPO/'.local/dcvc_uf_runtime/bin/python'),
                    receiver=old['runtimes']['receiver'],channel=old['runtimes']['channel'])
    for path in runtimes.values():
        if not Path(path).is_file() or not os.access(path,os.X_OK):
            raise ValueError('runtime missing: '+path)
    vendor, weights = REPO/'.local/dcvc_uf_vendor', REPO/'.local/dcvc_uf_weights'
    for name,digest in pilot.WEIGHTS.items():
        if sha256(weights/name) != digest:
            raise ValueError('UF weight changed: '+name)
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
            != pilot.transport.UPSTREAM or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor changed')
    for path,digest in old['assets'].items():
        if '/rife/' in path or '/Practical-RIFE/' in path:
            if sha256(path) != digest:
                raise ValueError('RIFE asset changed: '+path)
    p = dict(policy='sparc_uf_source_change_no_gap_v1',output=str(root),
        reference=str(ref),reference_protocol_sha256=sha256(ref/'protocol.json'),
        semantic_sha256=sha256(ref/'semantic.json'),source=source,profile=config,
        semantic_keys=semantic['keys'],qp=args.qp,runtimes=runtimes,
        vendor=str(vendor),weights_directory=str(weights),weights=pilot.WEIGHTS,
        quality_gate=base.read(REPO/'configs/sparc_auxiliary_search.json'),
        snr_db=10,channel_seed=42,total_trials=1,new_training=False,
        code={str(f.relative_to(REPO)):sha256(f)
              for f in sorted((REPO/'src/semantic_transmission').glob('*.py'))})
    p.update(auxiliary_selector=selector,selector_config=selector_config)
    if selector == 'detail':
        p['policy'] = 'sparc_uf_source_detail_no_gap_v1'
    p['signature'] = fingerprint(p)
    return p


def run(p):
    root = Path(p['output'])
    freeze(root/'protocol.json',p)
    if not (root/'started.json').exists():
        write_json(root/'started.json',dict(unix=time.time()))

    class Experiment(rate.Experiment):
        def state(self,*args,**kwargs):
            super().state(*args,**kwargs)
            status = base.read(root/'status.json')
            status['resume'] = ('rerun the same run_sparc_uf_no_gap.sh run command; '
                                'completed stage hashes are checked')
            write_json(root/'status.json',status)

    experiment = Experiment(p)
    experiment.state('source','no gap','select',root,root/'logs/select.log')
    stages = Stages(root,p['signature'])

    def select(_):
        started = time.perf_counter()
        if p.get('auxiliary_selector','coarse') == 'detail':
            plan = detail.select(p['source'],p['semantic_keys'],p['profile'],p['selector_config'],
                                 lambda n,t:sparc.progress(root,'select',n,t))
        else:
            proxies,cuts = selection.signals(p['source'],p['profile'])
            plan = selection.plan(proxies,p['semantic_keys'],cuts,p['profile'])
        plan.setdefault('selection_seconds',time.perf_counter()-started)
        plan['semantic_sha256'] = p['semantic_sha256']
        write_json(root/'selection.json',plan)

    stages.step('select',['selection.json'],['selection.json'],select)
    plan = base.read(root/'selection.json')
    chunks = rate.transport.descriptors(plan['indices'],plan['cuts'])
    clip = dict(id='source')  # no LGVSC budget is assumed
    result = experiment.evaluate(clip,p['source'],plan,dict(
        name=f'uniform_q{p["qp"]}',kind='uniform',center=p['qp'],strength=0,
        qps=[p['qp']]*len(chunks),changed_chunks=0))
    write_json(root/'RESULT.json',dict(
        status=COMPLETE,protocol_sha256=sha256(root/'protocol.json'),result=result,
        max_gap=None,observed_max_gap=max(b-a for a,b in zip(plan['indices'],plan['indices'][1:])),
        semantic_keys=plan['semantic_keys'],indices=plan['indices'],cuts=plan['cuts'],
        auxiliary_selector=p.get('auxiliary_selector','coarse'),
        selection_seconds=plan['selection_seconds'],
        new_training_performed=False,source_pixels_at_receiver=False,
        receiver_quality_used_for_selection=False,
        quality_preservation_proven=False,independent_semantic_evaluation=False))
    experiment.state('source','complete','complete',status=COMPLETE,exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run'])
    parser.add_argument('--reference',type=Path,required=True,
                        help='completed SPARC run; reuse its verified semantic keys and source')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qp',type=int,default=30)
    parser.add_argument('--selector',choices=['coarse','detail'],default='coarse',
                        help='source-only auxiliary selector; neither uses trial reconstruction')
    parser.add_argument('--detail-scale',type=float,default=1.,
                        help='source-detail tolerance multiplier; lower selects more detail')
    args = parser.parse_args()
    if args.action == 'check':
        print(dict(status='PREFLIGHT_PASSED',signature=preflight(args)['signature']))
        return
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'execution.lock'),exclusive_lock(REPO/'.local/etri_60s_check.lock'):
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        try:
            run(preflight(args))
        except BaseException as error:
            status = base.read(root/'status.json') if (root/'status.json').exists() else {}
            status.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                          error=str(error),exit_code=130 if isinstance(error,KeyboardInterrupt) else 1)
            write_json(root/'status.json',status)
            raise


if __name__ == '__main__':
    main()
