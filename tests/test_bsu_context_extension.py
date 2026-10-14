import json

import pytest

from semantic_transmission import bsu_context_extension as extension
from semantic_transmission.artifacts import write_json
from test_bsu_receiver_compare import remote


def test_extension_inventory_excludes_unneeded_movies_and_raw_generation(remote):
    result_path=remote['job']/'RESULT.json'
    result=json.loads(result_path.read_text())
    for b in extension.BOUNDARIES:
        path=remote['job']/f'refined/quality_{b}.csv';path.write_text('frame,psnr_db\n0,20\n')
        result['artifacts'][f'refined/quality_{b}.csv']=extension.digest(path)
    write_json(result_path,result)
    for name in ('official_quality.py','research_quality.py','stutter_metrics.py'):
        path=remote['repo']/'src/semantic_transmission'/name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# evaluator\n')
    before={str(p):extension.digest(p) for p in remote['repo'].rglob('*') if p.is_file()}
    m=extension.inventory_selected(str(remote['repo']),'outputs/bench',[dict(id=remote['vid'],snrs=[6])])
    assert m['cases'][0]['raw_frames']==[] and m['cases'][0]['base_movie'] is None
    assert m['sources'][0]['movie'] is None
    assert all('_frames/' not in n or 'key_frames_received/' in n for n in m['files'])
    assert len([n for n in m['files'] if '/refined/frames/' in n])==9
    assert sum(n.endswith('.csv') for n in m['files'])==2
    assert before=={str(p):extension.digest(p) for p in remote['repo'].rglob('*') if p.is_file()}


def test_primary_macro_aggregation_separates_partial_snr_sources():
    cases=[];rows=[]
    for vid,snrs,gain in [('a',[6,8,10],1),('b',[6,8,10],3),('partial',[6,8],100)]:
        for snr in snrs:
            cases.append(dict(id=vid,snr=snr))
            for arm in extension.ARMS:
                rows.append(dict(id=vid,snr=snr,arm=arm,**{b:dict(psnr_db=20+(gain if arm=='context_only' else 0)) for b in extension.BOUNDARIES}))
    sources,groups=extension.aggregate(rows,cases)
    assert len(sources)==3
    primary=next(g for g in groups if g['cohort']=='complete_snr_sources' and g['boundary']=='delivered_mp4')
    assert primary['source_count']==2 and primary['delta']['psnr_db']==pytest.approx(2)
    # An incomplete receiver pair must not silently enter the mean.
    rows.pop()
    sources,groups=extension.aggregate(rows,cases)
    assert len(sources)==2 and all(g['cohort']=='complete_snr_sources' for g in groups)


def test_offline_missing_manifest_never_connects(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(extension.subprocess,'run',lambda *a,**k:pytest.fail('network attempted'))
    with pytest.raises(ValueError,match='frozen transfer manifest'):
        extension.obtain_manifest(SimpleNamespace(root=tmp_path,offline=True))
