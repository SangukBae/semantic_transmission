import pytest

from semantic_transmission.artifacts import write_json, sha256
from semantic_transmission.auto_extraction import read
from semantic_transmission.qwen35_state_delta_eval import freeze_write, reuse_identical
from semantic_transmission.qwen35_feedback_eval import validate_judgments


def test_packet_freeze_refuses_changed_candidate_without_overwriting(tmp_path):
    path=tmp_path/'packet.json'
    freeze_write(path,dict(caption='A dog is visible.'))
    before=sha256(path)
    with pytest.raises(ValueError,match='frozen artifact'):
        freeze_write(path,dict(caption='A cat is visible.'))
    assert sha256(path)==before


def test_correspondence_schema_lists_every_clause_preserving_input_and_rules():
    import json
    from semantic_transmission.qwen35_state_delta import MATCH
    from semantic_transmission.qwen35_state_delta_match import complete_clause_schema
    metadata = dict(target=[12,24],target_ids=[12,16,20,23],context_ids=[6,29],
                    clauses=[dict(id='anchor',text='A dog stands.'),dict(id='change_0',text='The dog leaves.')])
    prefix = 'INPUT METADATA (not the output schema):\n'+json.dumps(metadata)+'\n'
    result = complete_clause_schema(prefix+MATCH)
    assert result.startswith(prefix+MATCH[:MATCH.index('Return ONLY JSON:')])
    assert result.endswith(MATCH[MATCH.index('Do not convert uncertainty'):])
    schema = json.loads(next(line for line in result.splitlines() if line.startswith('{"verdicts"')))
    assert [r['id'] for r in schema['verdicts']] == ['anchor','change_0']
    assert all(r['match']=='target|neighbor|neither|uncertain' for r in schema['verdicts'])


def test_only_identical_caption_and_time_can_reuse_real_astra_judgment(tmp_path):
    dest=tmp_path/'evaluation'
    packets=[]
    for method in ('M71','M93'):
        bins=[]
        for i in range(2):
            text='A dog is visible.' if method=='M71' or i==0 else 'A person is visible.'
            bins.append(dict(reference=dict(id=i,facts=[dict(id=f'f{i}')]),candidates=[dict(method=method,
                captions=[dict(caption_id=f'{method}_{i:03d}',start=i*24,end_exclusive=(i+1)*24,caption=text)])]))
        path=dest/'packets'/f'{method}.json'
        write_json(path,dict(bins=bins));packets.append(path)
    write_json(dest/'protocol.json',dict(reference_sha256='reference'))
    write_json(dest/'input_manifest.json',{str(p):sha256(p) for p in packets})
    prior=dict(evaluator_model='gpt-6-astra',reference_sha256='reference',packet_sha256=sha256(packets[0]),
        judgments=[dict(bin_id=i,method='M71',covered_fact_ids=[f'f{i}'],unsupported_claims=[],uncertain_claims=[]) for i in range(2)])
    write_json(dest/'judgments/base.json',prior)
    result=reuse_identical(tmp_path)
    assert result==dict(changed_bins=[1],reused_count=1)
    rows=read(dest/'judgments/exact_input_reuse.json')['judgments']
    assert rows[0]['covered_fact_ids']==['f0'] and rows[0]['reuse_of']['method']=='M71'
    with pytest.raises(ValueError,match='incomplete Astra review'):
        validate_judgments(tmp_path)
    write_json(dest/'judgments/new.json',dict(evaluator_model='gpt-6-astra',reference_sha256='reference',
        packet_sha256=sha256(packets[1]),judgments=[dict(bin_id=1,method='M93',covered_fact_ids=[],unsupported_claims=[],uncertain_claims=[])]))
    assert len(validate_judgments(tmp_path))==4
