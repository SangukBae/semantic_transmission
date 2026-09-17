#!/usr/bin/env python3
"""Independent byte-decoding checks for technical STA abstention, not semantic accuracy."""
import argparse
from pathlib import Path
import pickle
import time

from semantic_transmission.metric_v3_formal import array,read,save
from semantic_transmission.sta_video_validation import receive
from semantic_transmission.sta_selective import predict
from semantic_transmission.artifacts import sha256


def read_and_score_input_failure(reference,packets,model):
    if reference is None:
        readable=False;reason='missing_reference'
    else:
        try:
            receive(packets)
            readable=True;reason=None
        except ValueError as error:
            readable=False;reason=str(error)
    # A failure to read mandatory input cannot produce semantic features.
    result=predict({k:None for k in ('edr','clr','gfr','ghr')},model['fit'],model['rejection'],observable=readable)
    return {'transport_readable':readable,'transport_reason':reason,'decision':result}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.output;model=read(root/'calibration.json')['models']['sta'];results=[]
    for source in read(root/'sources.json'):
        stem=source['source_stem'];a=array(root/'sources'/stem/'source.npz')
        path=root/'cases'/(stem+'__sta_control_identity')/'packets.pkl'
        with path.open('rb') as f:packets=pickle.load(f)['rx_packets']
        decoded,statuses,_=receive(packets)
        assert (decoded==a).all() and set(statuses)=={'ok'}
        damaged=[]
        for packet in packets:
            data=bytearray(packet['payload']);data[len(data)//2]^=1
            damaged.append({**packet,'payload':bytes(data)})
        for kind,reference,received in (('missing_reference',None,packets),('all_packets_lost',a,[None]*len(packets)),
                                         ('all_checksums_failed',a,damaged)):
            result=read_and_score_input_failure(reference,received,model)
            assert not result['transport_readable'] and result['decision']['prediction']=='abstain'
            results.append({'source_id':source['source_id'],'kind':kind,**result})
    save(root/'input_adapter_audit.json',{'status':'PASSED','recorded_unix':time.time(),'script_sha256':sha256(Path(__file__)),
        'valid_packet_roundtrips':len(read(root/'sources.json')),'failure_cases':len(results),'results':results,
        'scope':'Actual PNG decode/checksum/mandatory-input failures routed to abstention; deterministic technical test, not semantic uncertainty accuracy'})
    print('input adapter checks passed',len(results),flush=True)


if __name__=='__main__':main()
