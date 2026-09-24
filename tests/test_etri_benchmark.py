"""Checks that prevent leakage or empty annotation files becoming evidence."""
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from etri_benchmark import fps_time_mapping
from etri_benchmark_package import source_reviews_complete, validate_splits, validate_job_inputs


def record(**updates):
    row={'id':'a','source_id':'source_a','source_sha256':'hash_a','content_group':'group_a',
         'split':'development','previously_piloted':False}
    return dict(row,**updates)


@pytest.mark.parametrize('field',['source_id','source_sha256','content_group'])
def test_no_source_or_group_leakage(field):
    a=record()
    b=record(id='b',source_id='source_b',source_sha256='hash_b',content_group='group_b',split='test')
    b[field]=a[field]
    assert any(field in error for error in validate_splits([a,b]))


def test_extension_shares_parent_partition():
    assert validate_splits([record(),record(id='a_120s')])==[]


def test_pilot_cannot_be_promoted_to_test():
    assert validate_splits([record(previously_piloted=True,split='test')])


def review(name='a',**updates):
    return dict({'id':'video','reviewer_id':name,'status':'COMPLETE','independent':True,
                 'adjudicated':True,'coverage_start_sec':0,'coverage_end_sec':60},**updates)


def test_empty_annotations_are_not_zero_error_evidence():
    assert not source_reviews_complete([],{'video':60})
    assert not source_reviews_complete([], {})


@pytest.mark.parametrize('bad',[
    review('b',status='NOT_STARTED'), review('b',independent=False),
    review('b',adjudicated=False), review('b',coverage_end_sec=59),review('a'),review('')])
def test_two_distinct_independent_complete_reviews_required(bad):
    assert not source_reviews_complete([review(),bad],{'video':60})


def test_complete_adjudicated_reviews_pass_only_source_gate():
    assert source_reviews_complete([review('a'),review('b')],{'video':60})


def test_fps_mapping_uses_actual_selected_source_timestamp():
    log='''Read frame with in pts 10000, out pts 0
Read frame with in pts 43367, out pts 1
Writing frame with pts 0 to pts 0
Read frame with in pts 76733, out pts 2
Writing frame with pts 1 to pts 1'''
    rows=fps_time_mapping(log,10,2)
    assert rows[0]['source_time_sec']==pytest.approx(10.01)
    assert rows[1]['source_time_sec']==pytest.approx(10.043367)


def test_cfr_repeat_is_visible_in_time_mapping():
    log='''Read frame with in pts 0, out pts 0
Read frame with in pts 83333, out pts 2
Writing frame with pts 0 to pts 0
Writing frame with pts 0 to pts 1'''
    rows=fps_time_mapping(log,20,2)
    assert rows[0]['source_time_sec']==rows[1]['source_time_sec']
    with pytest.raises(ValueError):fps_time_mapping(log,20,3)


def test_matching_method_hashes_still_must_match_frozen_input():
    row={'id':'a','processed_sha256':'correct','frames':1440}
    jobs=[{'id':'a','input_sha256':'wrong','expected_frames':1440} for _ in range(2)]
    assert len(validate_job_inputs([row],jobs))==2


def test_job_cannot_silently_evaluate_a_truncated_input():
    row={'id':'a','processed_sha256':'correct','frames':1440}
    jobs=[{'id':'a','input_sha256':'correct','expected_frames':384}]
    assert validate_job_inputs([row],jobs)
