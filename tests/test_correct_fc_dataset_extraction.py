import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('fc_correction', Path(__file__).resolve().parents[1] / 'scripts/correct_fc_dataset_extraction.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_caption_reuse_requires_interval_and_sample_identity_not_segment_number():
    old = dict(records=[dict(segment=7,start=24,end_exclusive=48,source_indices=[26,32,38,43],text='Visible dog.')],
               source_frame_hashes={str(i):str(i) for i in [26,32,38,43]})
    samples=[dict(segment=2,start=24,end_exclusive=48,source_indices=[26,32,38,43]),
             dict(segment=3,start=25,end_exclusive=48,source_indices=[26,32,38,43])]
    reused,changed=module.reusable_caption_records(old,samples,old['source_frame_hashes'])
    assert reused=={'2':'Visible dog.'}
    assert changed==[samples[1]]


def test_same_indices_with_changed_source_pixels_cannot_reuse_caption():
    row=dict(segment=0,start=0,end_exclusive=1,source_indices=[0,0,0,0])
    old=dict(records=[dict(row,text='A person.')],source_frame_hashes={'0':'original'})
    reused,changed=module.reusable_caption_records(old,[row],{'0':'changed'})
    assert reused=={} and changed==[row]
