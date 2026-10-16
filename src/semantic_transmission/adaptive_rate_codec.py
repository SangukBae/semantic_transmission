"""Variable-QP extension with one continuous DCVC prediction state."""
import io
from .bsu_dcvc_keys import Codec, WEIGHTS
from .wire import pack


def encode(codec, images, indices, qps):
    if (len(images) != len(indices) or len(indices) != len(qps) or not len(qps) or
            any(type(q) is not int or not 0 <= q <= 63 for q in qps)):
        raise ValueError('invalid per-frame QPs')
    t,s=codec.torch,codec.stream
    n,h,w,_=images.shape
    output,helper=io.BytesIO(),s.SPSHelper()
    padding_r,padding_b=codec.i.get_padding_size(h,w,16)
    codec.p.clear_dpb();codec.p.set_curr_poc(0)
    last_qp=0
    with t.no_grad():
        for index,(image,qp) in enumerate(zip(images,qps)):
            x=t.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255
            x=codec.pad(codec.rgb2ycbcr(x).half(),padding_b,padding_r)
            is_i=index==0;ada=not is_i and index%32==1
            sps=dict(sps_id=-1,height=h,width=w,ec_part=0,use_ada_i=int(ada))
            if is_i:
                curr_qp=qp;encoded=codec.i.compress(x,curr_qp)
                codec.p.clear_dpb();codec.p.add_ref_frame(None,encoded['x_hat'])
            else:
                if ada:codec.p.prepare_feature_adaptor_i(last_qp)
                curr_qp=codec.p.shift_qp(qp,[0,1,0,2,0,2,0,2][index%8])
                encoded=codec.p.compress(x,curr_qp);last_qp=curr_qp
            sps_id,new=helper.get_sps_id(sps);sps['sps_id']=sps_id
            if new:s.write_sps(output,sps)
            s.write_ip(output,is_i,sps_id,curr_qp,encoded['bit_stream'])
    return pack(dict(schema='bsu_dcvc_rt_keys_v1',indices=indices,height=h,width=w,
        weights=WEIGHTS,color='full_range_ycbcr444_bt709',
        upstream='cbdae87a5445114cdc7f48816da63ea80bdeac40'),output.getvalue())
