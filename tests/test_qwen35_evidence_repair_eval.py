import copy
from collections import Counter
import pytest
from semantic_transmission.qwen35_evidence_repair_eval import ARMS, cell_order, input_row, validate_record


def test_five_conditions_rotate_and_targets_cannot_enter_model_input():
    for position in range(5):
        assert Counter(cell_order(i)[position] for i in range(5)) == Counter(ARMS)
    row = dict(id='r', source_id='s', fps=24, frames=[], keys=[0,2,5,7], caption='secret', facts=['secret'], claims=['secret'])
    assert set(input_row(row)) == {'id', 'source_id', 'fps', 'frames', 'keys'}


@pytest.mark.parametrize('mutation', ['keys', 'hash', 'tokens', 'grid', 'missing_arm'])
def test_comparison_rejects_unmatched_or_incomplete_inputs(mutation):
    row = dict(id='r', keys=[0,2,5,7], frames=[dict(sha256=f'h{i}') for i in range(8)])
    cell = dict(keys=row['keys'], source_hashes=['h0','h2','h5','h7'], input_tokens=123, grid=[[1,12,12]]*4)
    record = dict(id='r', protocol_sha256='p', cells={a:copy.deepcopy(cell) for a in ARMS})
    validate_record(row, record, 'p')
    bad = record['cells']['preference']
    if mutation == 'keys': bad['keys'] = [0,1,5,7]
    elif mutation == 'hash': bad['source_hashes'][0] = 'different'
    elif mutation == 'tokens': bad['input_tokens'] += 1
    elif mutation == 'grid': bad['grid'] = [[1,12,12]]*3
    else: del record['cells']['evidence']
    with pytest.raises(ValueError): validate_record(row, record, 'p')
