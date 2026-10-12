"""Source-paired cadence measurements; never used by receiver inference."""
import cv2
import numpy as np


def evaluate(source, generated, progress=None):
    if source.shape != generated.shape or len(source)<3:
        raise ValueError('matching video timelines required')
    def gray(frames):
        return [cv2.cvtColor(cv2.resize(f,(288,160)),cv2.COLOR_RGB2GRAY) for f in frames]
    a,b=gray(source),gray(generated)
    source_motion=[]; generated_motion=[]; flow_error=[]; cadence_error=[]
    duplicate=[]; previous=None
    for i in range(1,len(a)):
        flows=[cv2.calcOpticalFlowFarneback(x[i-1],x[i],None,.5,3,15,3,5,1.2,0) for x in (a,b)]
        magnitudes=[np.linalg.norm(f,axis=2).reshape(5,32,8,36).mean(axis=(1,3)) for f in flows]
        source_motion.append(magnitudes[0]);generated_motion.append(magnitudes[1])
        flow_error.append(float(np.linalg.norm(flows[0]-flows[1],axis=2).mean()))
        if previous is not None:
            cadence_error.append(float(np.abs((magnitudes[1]-previous[1])-(magnitudes[0]-previous[0])).mean()))
        previous=magnitudes
        duplicate.append(bool(np.array_equal(generated[i],generated[i-1])))
        if progress is not None:progress(i,len(a)-1)
    ma,mb=np.asarray(source_motion),np.asarray(generated_motion)
    active=ma>.15
    frozen=active & (mb<.25*ma)
    eligible=active[:-1]&active[1:]
    jumps=eligible & frozen[:-1] & (mb[1:]>1.8*ma[1:])
    luma_error=np.asarray([y.mean()-x.mean() for x,y in zip(a,b)])
    luma_impulse=np.abs(luma_error[1:-1]-.5*(luma_error[:-2]+luma_error[2:]))
    mad=[float(np.abs(y.astype(float)-x).mean()) for x,y in zip(b,b[1:])]
    return dict(flow_error_mean=float(np.mean(flow_error)),
                cadence_error_mean=float(np.mean(cadence_error)),
                source_motion_mean=float(ma.mean()),generated_motion_mean=float(mb.mean()),
                motion_retention_ratio=float(mb.sum()/max(ma.sum(),1e-9)),
                active_motion_cells=int(active.sum()),false_freeze_cells=int(frozen.sum()),
                false_freeze_rate=float(frozen.sum()/max(1,active.sum())),
                stall_jump_cells=int(jumps.sum()),stall_jump_eligible_cells=int(eligible.sum()),
                stall_jump_rate=float(jumps.sum()/max(1,eligible.sum())),
                exact_duplicate_transitions=int(sum(duplicate)),
                max_luma_impulse=float(luma_impulse.max()),mean_luma_impulse=float(luma_impulse.mean()),
                mad_p95=float(np.percentile(mad,95)),mad_max=float(max(mad)),
                per_transition=dict(source_motion=ma.mean(axis=(1,2)).tolist(),
                    generated_motion=mb.mean(axis=(1,2)).tolist(),flow_error=flow_error,
                    cadence_error=[None,*cadence_error],false_freeze_cells=frozen.sum(axis=(1,2)).tolist(),
                    generated_mad=mad),
                definition='288x160 grayscale Farneback, 8x5 spatial cells; source active >0.15 px/frame, false freeze <25% of source; stall then jump >180% next source speed',
                limitation='diagnostic proxies, not human perceptual stutter labels; lower cadence error alone can reward freezing')
