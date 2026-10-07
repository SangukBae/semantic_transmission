"""Audit paired training inputs, initialization, frozen heads and checkpoints."""
from pathlib import Path
import torch
from safetensors.torch import load_file

from semantic_transmission.qwen35_astra_transfer import RUN,ARMS,verify_teachers
from semantic_transmission.qwen35_training import checkpoint_path
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256


def audit_sheets():
    from PIL import Image,ImageDraw
    sheets={};panels=0
    for packet in sorted((RUN/'teacher').glob('packet[48]_*.json')):
        for row in read_json(packet)['records']:
            images=[Image.open(f['path']).convert('RGB') for f in row['frames']]
            width=max(im.width for im in images);height=max(im.height for im in images)
            expected=Image.new('RGB',(2*width,((len(images)+1)//2)*(height+26)),'white')
            draw=ImageDraw.Draw(expected)
            for n,(frame,im) in enumerate(zip(row['frames'],images)):
                x=(n%2)*width;y=(n//2)*(height+26)
                draw.text((x+4,y+4),f"i{frame['i']} | frame {frame['index']} | {frame['index']/row['fps']:.3f}s",fill='black')
                expected.paste(im,(x,y+26))
            actual=Image.open(row['sheet']).convert('RGB')
            assert actual.size==expected.size and actual.tobytes()==expected.tobytes()
            sheets[row['sheet']]=sha256(row['sheet']);panels+=len(images)
    return dict(status='PASS_NATIVE_PIXELS_AND_LABELS',sheets=len(sheets),source_panels=panels,files=sheets,
        scope='Deterministic pixel/label comparison against current frozen source PNGs; not evidence of identical internal visual tokenization across different model families.')


def main():
    verify_teachers(RUN)
    protocol=read_json(RUN/'protocol.json')
    control=read_json(RUN/'train_control.json')['records']
    teacher=read_json(RUN/'train_teacher.json')['records']
    assert len(control)==len(teacher)==76
    for left,right in zip(control,teacher):
        assert all(left[k]==right[k] for k in ['id','video_id','source_sha256','frames','keys','origin'])
    rows={s:read_json(RUN/f'{s}_inputs.json')['records'] for s in ['validation','test']}
    source_sets={'train':{r['source_sha256'] for r in control},**{s:{r['source_sha256'] for r in rr} for s,rr in rows.items()}}
    for left in source_sets:
        for right in source_sets:
            if left!=right:assert not source_sets[left]&source_sets[right]
    initial={};reports={}
    original=torch.load(Path(protocol['checkpoint'])/'head.pt',weights_only=True,map_location='cpu')
    implementations=set();orders={}
    from semantic_transmission.qwen35_caption_repair import order_for
    for arm,(target,rank) in ARMS.items():
        stage=RUN/arm;record=read_json(stage/'RESULT.json');identity=read_json(stage/'protocol.json')
        assert record['status']=='COMPLETE' and record['state']['updates']==44 and record['state']['next_epoch']==2
        assert sha256(stage/'protocol.json')==record['state']['protocol_sha256']
        assert sha256(stage/'implementation.py')==identity['implementation_sha256']
        implementations.add(identity['implementation_sha256'])
        for path,hashed in identity['dependency_sha256'].items():
            assert sha256(RUN.parents[1]/path)==hashed
        first=stage/'update-000000-e000-n0000'
        final=checkpoint_path(stage)
        for path in [first,final]:
            for name,hashed in read_json(path/'complete.json')['files'].items():assert sha256(path/name)==hashed
            head=torch.load(path/'head.pt',weights_only=True,map_location='cpu')
            assert set(head)==set(original) and all(torch.equal(head[k],v) for k,v in original.items())
        init=load_file(str(first/'adapter/adapter_model.safetensors'))
        learned=load_file(str(final/'adapter/adapter_model.safetensors'))
        assert set(init)==set(learned)
        changed=sum(not torch.equal(init[k],learned[k]) for k in init)
        assert changed>0
        cfg=read_json(final/'adapter/adapter_config.json')
        assert cfg['r']==rank and cfg['lora_alpha']==rank*2
        initial[arm]=init
        orders[arm]=[[r['id'] for r in order_for(control if target=='control' else teacher,protocol,e)] for e in range(2)]
        reports[arm]=dict(rank=rank,alpha=cfg['lora_alpha'],adapter_tensors=len(init),changed_tensors=changed,
            trainable_parameters=sum(v.numel() for v in learned.values()),head_unchanged=True,
            final_checkpoint=str(final),updates=record['state']['updates'],
            training_seconds=record['state']['training_seconds'],peak_allocated_gib=record['state']['peak_allocated_gib'])
        write_json(stage/'status.json',dict(status='COMPLETE',epoch=2,updates=44,result_sha256=sha256(stage/'RESULT.json')))
    assert len(implementations)==1
    assert all(v==orders['control_r8'] for v in orders.values())
    for rank in [8,32]:
        a=initial[f'control_r{rank}'];b=initial[f'teacher_r{rank}']
        assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
    sheet_report=audit_sheets()
    write_json(RUN/'teacher_sheet_audit.json',sheet_report)
    result=dict(status='PASS',paired_inputs=True,identical_initialization_within_rank=True,
        identical_training_orders=True,identical_implementation=True,
        independent_source_counts={s:len(v) for s,v in source_sets.items()},arms=reports,
        teacher_freeze_sha256=sha256(RUN/'teacher_freeze.json'),
        teacher_sheet_audit_sha256=sha256(RUN/'teacher_sheet_audit.json'),
        script_sha256=sha256(Path(__file__)))
    write_json(RUN/'training_audit.json',result)
    print(result)


if __name__=='__main__':main()
