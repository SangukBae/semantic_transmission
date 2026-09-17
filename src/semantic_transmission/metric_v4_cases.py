"""New scene families and independent state truth for duration/delay validation."""
import cv2
import numpy as np

from .metric_v3_cases import annotation_events

FPS, COUNT, HEIGHT, WIDTH = 8., 32, 192, 320
FAMILIES = ('cross_axis_exit', 'entrance_and_stop', 'reentry_and_restart')
CONTROLS = ('identity', 'brightness20', 'texture', 'camera', 'palette')


def scene(seed, family):
    rng = np.random.default_rng(seed)
    c = np.zeros((COUNT, 4, 2), np.float32); v = np.ones((COUNT, 4), bool)
    exit_t = int(rng.integers(19, 23)); event_t = int(rng.integers(6, 10))
    origins = np.array([[55, 43], [235, 47], [155, 140], [265, 137]], float)
    origins += rng.uniform(-5, 5, origins.shape)
    c[:] = origins
    if family == 'cross_axis_exit':
        c[:, 0, 0] += np.arange(COUNT) * 1.4
        c[:, 2, 0] += np.where(np.arange(COUNT)<12, np.arange(COUNT)*2., 24-(np.arange(COUNT)-12)*2.)
        v[:, 1] = np.arange(COUNT) >= event_t
        target_type = 'enter'
    elif family == 'entrance_and_stop':
        c[:, 0, 1] += np.minimum(np.arange(COUNT), 12) * 1.5
        c[:, 1, 0] -= np.minimum(np.arange(COUNT), event_t) * 2.5
        v[:, 2] = np.arange(COUNT) >= 3
        target_type = 'stop'
    elif family == 'reentry_and_restart':
        v[:, 0] = ((np.arange(COUNT) >= 3) & (np.arange(COUNT) < 10)) | (np.arange(COUNT) >= 14)
        c[:, 1, 0] -= np.maximum(np.arange(COUNT) - event_t, 0) * 2.1
        c[:, 3, 1] += np.minimum(np.arange(COUNT), 9) * .8
        target_type = 'start'
    else:
        raise ValueError(family)
    v[exit_t:, 0] = False
    colors = np.array([[210,100,75],[90,195,130],[90,130,225],[225,190,90]], float) + rng.uniform(-18,18,(4,3))
    return {'centers':c,'visible':v,'colors':colors,'sizes':rng.integers(15,20,4),
            'seed':seed,'family':family,'exit_t':exit_t,'event_t':event_t,'target_type':target_type}


def draw(state, centers=None, visible=None, style='identity'):
    c = state['centers'] if centers is None else centers
    v = state['visible'] if visible is None else visible
    yy, xx = np.mgrid[:HEIGHT,:WIDTH]; frames=[]; labels=[]
    for t in range(COUNT):
        base = 42 + 5*np.sin(xx*.08) + 4*np.cos(yy*.11)
        if style == 'texture':base += 20*np.sin(xx*.4)*np.cos(yy*.29)
        frame = np.clip(base[...,None]+np.array([0,8,13]),0,255).astype(np.uint8)
        mask = np.zeros((HEIGHT,WIDTH),np.uint8)
        for k in range(4):
            if not v[t,k]:continue
            x,y=c[t,k];size=state['sizes'][k];u=(xx-x)/size;w=(yy-y)/size
            shape = (k + FAMILIES.index(state['family'])) % 3
            if shape==0:inside=(np.abs(u)+np.abs(w))<=1.3
            elif shape==1:inside=(u/1.2)**2+(w/.85)**2<=1
            else:inside=(np.abs(u)<=1)&(np.abs(w)<=1)
            detail = 9*np.sin(u*4)*np.cos(w*5)
            if style=='texture':detail=23*np.sin(u*10)*np.cos(w*8)
            color = np.clip(state['colors'][k]+detail[...,None],0,255).astype(np.uint8)
            frame[inside]=color[inside];mask[inside]=k+1
        if style=='brightness20':frame=np.clip(frame.astype(float)*1.15+20,0,255).astype(np.uint8)
        elif style=='palette':
            hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV);hsv[...,0]=(hsv[...,0].astype(int)+45)%180
            frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        elif style=='camera':
            matrix=cv2.getRotationMatrix2D((WIDTH/2,HEIGHT/2),1.5*np.sin(t*.4),1.)
            matrix[:,2]+=[2*np.sin(t*.8),2*np.cos(t*.6)]
            frame=cv2.warpAffine(frame,matrix,(WIDTH,HEIGHT),borderMode=cv2.BORDER_REFLECT_101)
            mask=cv2.warpAffine(mask,matrix,(WIDTH,HEIGHT),flags=cv2.INTER_NEAREST)
        frames.append(frame);labels.append(mask)
    return np.stack(frames), {'centers':c.copy(),'visible':v.copy(),'masks':np.stack(labels)}


def variants(state):
    for style in CONTROLS:
        frames,truth=draw(state,style=style)
        yield style,frames,truth,{'target':'control','kind':'control','severity':0.,'target_object':None,'target_event':None}
    for kind,delays in (('ghost_hold',(2,4,8)),('ghost_return',(2,)),('event_delay',(2,4,8)),('never_update',(8,))):
        for d in delays:
            c=state['centers'].copy();v=state['visible'].copy()
            if kind.startswith('ghost'):
                t=state['exit_t'];k=0;event='exit'
                times=range(t,min(COUNT,t+d)) if kind=='ghost_hold' else range(t+2,min(COUNT,t+4))
                for q in times:v[q,k]=True;c[q,k]=c[t-1,k]
            else:
                t=state['event_t'];k=1;event=state['target_type']
                for q in range(t,COUNT):
                    if event=='stop':
                        previous_velocity=state['centers'][t-1,k]-state['centers'][t-2,k]
                        extension=min(q-t,d) if kind=='event_delay' else q-t
                        c[q,k]=state['centers'][t,k]+extension*previous_velocity
                    else:
                        index=max(0,q-d) if kind=='event_delay' else t-1
                        c[q,k]=state['centers'][index,k];v[q,k]=state['visible'][index,k]
            frames,truth=draw(state,c,v)
            yield f'{kind}_{d}',frames,truth,{'target':'ghost' if kind.startswith('ghost') else 'delay',
                'kind':kind,'severity':d/FPS,'target_object':k+1,'target_event':event,'target_frame':t}


def truth_query(source,reconstruction,object_id,event_type,frame):
    """Known simulation entity/time; numerical oracle independent of RGB observer."""
    k=object_id-1; a=source['visible'];b=reconstruction['visible']
    result={}
    if event_type=='exit':
        times=list(range(frame,min(COUNT,frame+4)))
        assert not a[times,k].any()
        result.update(ghost_auc=float(b[times,k].mean()),ghost_duration_s=float(b[times,k].sum()/FPS),
                      ghost_complete_horizon=len(times)==4)
    events=annotation_events(reconstruction['centers'],reconstruction['visible'])
    candidates=[e for e in events if e['object_id']==object_id and e['type']==event_type
                and frame/FPS<=e['time_s']<=frame/FPS+1.]
    delay=min((e['time_s']-frame/FPS for e in candidates),default=None)
    result.update(delay_s=delay,delay_penalty=delay if delay is not None else 1.,
                  delay_status='observed' if delay is not None else 'deadline_miss')
    return result
