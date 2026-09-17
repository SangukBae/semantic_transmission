#!/usr/bin/env python3
"""Serialize NumPy scalar results of the unchanged frozen statistics program.

The registered program produces np.bool_ flags for measured event queries.
Only their JSON representation is normalized; no score or calculation changes.
"""
from pathlib import Path
import runpy
import sys
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission import metric_v3_formal


def json_types(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return json_types(value.tolist())
    if isinstance(value,dict):return {key:json_types(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_types(item) for item in value]
    return value


def main():
    original_save=metric_v3_formal.save
    metric_v3_formal.save=lambda path,value:original_save(path,json_types(value))
    root=Path(sys.argv[sys.argv.index('--output')+1])
    path=root/'report_serialization_operation.json'
    if not path.exists():
        original_save(path,{'recorded_unix':time.time(),'wrapper_sha256':sha256(Path(__file__)),
            'frozen_statistics_sha256':sha256(Path('scripts/report_metric_v4.py')),
            'reason':'np.bool_ measured flags are not accepted by standard json; convert NumPy scalar types to native values',
            'metric_changed':False,'statistics_changed':False,'threshold_changed':False})
    elif metric_v3_formal.read(path)['wrapper_sha256']!=sha256(Path(__file__)):
        raise ValueError('Serialization wrapper changed')
    runpy.run_path('scripts/report_metric_v4.py',run_name='__main__')


if __name__=='__main__':main()
