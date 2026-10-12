"""Predeclared geometry-only supplement; preserves the original pilot protocol.

Reuse its receiver implementation and actual noise, bypass only the learned CNN
residual. The checkpoint is loaded by the shared harness but its parameters do
not participate in the geometry-only forward pass. No base files are changed.
"""
import argparse
from pathlib import Path
import sys

from .artifacts import sha256,write_json
from .webvid5 import read_json
from . import bsu_motion_experiment as experiment


def plan(root):
    p=experiment.load(root)
    receipt=dict(parent_signature=p['signature'],arm='motion_geometry',
        contrast='motion_learned minus motion_geometry isolates the trained CNN residual',
        intervention='bypass MotionPrior.net; retain motion warping, endpoint mixing and .15 RFLOW guidance',
        trained_parameters_used_for_generation=False,output_pixel_warping=False,
        declared_before_learned_generation=not (root/'generation/zea/motion_learned/result.json').exists(),
        code_sha256=sha256(Path(__file__)))
    path=root/'geometry_ablation_plan.json'
    if path.exists():
        previous=read_json(path)
        if previous['parent_signature']!=receipt['parent_signature'] or previous['code_sha256']!=receipt['code_sha256']:
            raise ValueError('supplement changed')
        return previous
    write_json(path,receipt);return receipt


def generate(root):
    from . import motion_condition_adapter as adapter
    receipt=plan(root)
    class GeometryOnly(adapter.MotionPrior):
        def forward(self,features):
            time=features[:,8:9]
            return features[:,:4]*(1-time)+features[:,4:8]*time
    original=adapter.MotionPrior;old_arms=experiment.ARMS
    try:
        adapter.MotionPrior=GeometryOnly
        experiment.ARMS=['motion_geometry']
        experiment.generate(root)
    finally:
        adapter.MotionPrior=original;experiment.ARMS=old_arms
    write_json(root/'geometry_ablation_generation.json',dict(receipt,status='GENERATED'))


def evaluate(root):
    import numpy as np
    import torch
    from .official_quality import OfficialMetrics,summarize
    from .research_quality import read_frames
    from .receiver_alignment_evaluation import temporal_scores,comparison_video
    from .receiver_alignment_experiment import complete_arm
    p=experiment.load(root);receipt=plan(root);torch.set_num_threads(2);rows=[]
    for c in [c for c in p['cases'] if c['split']=='test']:
        directory=root/'generation'/c['name']/'motion_geometry'
        if not complete_arm(directory,p['signature']):raise ValueError('incomplete supplemental generation')
        source=experiment.source_window(c['source_video'],*c['keys']);pixels=read_frames(directory/'frames')
        summary,frames=OfficialMetrics().evaluate(source,pixels)
        temporal=temporal_scores(source,pixels,1)
        temporal['mad_p95']=float(np.percentile(temporal['per_transition']['generated_mad'],95))
        learned=read_json(root/'generation'/c['name']/'motion_learned/evaluation.json')
        row=dict(case=c['name'],all_frames=summary,temporal=temporal,
            trained_residual_effect=dict(psnr_db=learned['all_frames']['psnr_db']-summary['psnr_db'],
                lpips_vgg=learned['all_frames']['lpips_vgg']-summary['lpips_vgg'],
                flow_error=learned['temporal']['flow_error_mean']-temporal['flow_error_mean']))
        if c['name']=='zea':row['common381']=summarize(frames[:381],(381,320,576,3))
        write_json(directory/'evaluation.json',row);rows.append(row)
        comparison_video(root/'review'/c['name']/'geometry_vs_learned.mp4',
            [source,pixels,read_frames(root/'generation'/c['name']/'motion_learned/frames')],
            ['Source','Motion geometry only','Motion + learned residual'],24)
        print(f'ABLATION {len(rows)}/4 ({len(rows)*25}%) {c["name"]}: trained residual {row["trained_residual_effect"]}',flush=True)
    write_json(root/'geometry_ablation_result.json',dict(receipt,status='COMPLETE',cases=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=experiment.DEFAULT)
    parser.add_argument('--stage',choices=['plan','generate','evaluate'],required=True)
    args=parser.parse_args();sys.path.insert(0,str(experiment.REPO/'.local/vendor/Open-Sora'))
    globals()[args.stage](args.root.resolve())


if __name__=='__main__':main()
