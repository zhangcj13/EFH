import os.path as osp
import numpy as np
import numba
import cv2
import os
import json
from scipy.interpolate import InterpolatedUnivariateSpline
import random
import copy
from ..registry import PROCESS

@numba.jit(nopython=True)
def render_events_to_spikes(x, y, p, spikes):
    for x_, y_, p_ in zip(x, y, p):
        spikes[p_, y_, x_] = 1
    return spikes

@numba.jit(nopython=True)
def render_events_to_accumulation(x, y, p, V):
    for x_, y_, p_ in zip(x, y, p):
        V[p_, y_, x_] += 1
    return V

@PROCESS.register_module
class EncodeEvents(object):
    def __init__(self,  
                 encode_method='spike_with_dt',
                 wh=(640, 360), 
                 dt=1,
                 cfg=None):
        
        self.img_w, self.img_h = wh
        self.encode_method=encode_method
        self.dt=dt
        self.cfg=cfg

    # def _crop_events(self,events):
    #     not_crop_idx= (events['x']>0 ) & (events['x']<self.img_w )&(events['y']>0) & (events['y']<self.img_h)
    #     events = {k: v[not_crop_idx] if 'ts_' not in k else v for k, v in events.items()}
    #     return events
    
    def __call__(self, sample):
        if 'events' not in sample.keys():
            return
        if sample['events'] is None:
            return
        # cv2.imwrite('./work_dirs/DsecDetection/gt_images/events/rgb.jpg',sample['img'])
        # events = self._crop_events(sample['events'])
        # self._3d_plot_events(events)
        events=sample['events']

        if self.encode_method == 'spike_with_dt':
            sample['events']=self.split_to_spike(events)
        elif self.encode_method == 'accum_with_dt':
            sample['events']=self.accum_to_spike(events)
        else:
            sample['events']=events
        
        # self.draw_save_event_img(sample['events'])

        return sample

    def split_to_spike(self,events):
        ts = events['t']-events['ts_start']
        tr = events['ts_end'] - events['ts_start'] 

        time_step=int(tr/self.dt+0.5)
        t_spikes = np.zeros((2,self.img_h,self.img_w,time_step),dtype=np.float32)
        cur_ts=0
        for t in range(time_step):
            nxt_ts = cur_ts + self.dt
            index =  (ts>=cur_ts ) & (ts<nxt_ts)
            cur_p=events['p'][index]
            cur_x=events['x'][index]
            cur_y=events['y'][index]
            render_events_to_spikes(cur_x,cur_y,cur_p, t_spikes[...,t])
            # t_spikes[...,t] = render_events_to_spikes(cur_x,cur_y,cur_p, t_spikes[...,t])
            cur_ts = nxt_ts
        return t_spikes
    
    def accum_to_spike(self,events):
        ts = events['t']-events['ts_start']
        tr = events['ts_end'] - events['ts_start'] 

        time_step=int(tr/self.dt+0.5)
        t_values = np.zeros((2,self.img_h,self.img_w,time_step),dtype=np.float32)
        cur_ts=0
        for t in range(time_step):
            nxt_ts = cur_ts + self.dt
            index =  (ts>=cur_ts ) & (ts<nxt_ts)
            cur_p=events['p'][index]
            cur_x=events['x'][index]
            cur_y=events['y'][index]
            render_events_to_accumulation(cur_x,cur_y,cur_p, t_values[...,t])
            # t_spikes[...,t] = render_events_to_spikes(cur_x,cur_y,cur_p, t_spikes[...,t])
            cur_ts = nxt_ts
        return t_values
    
    def as_image(self,events):
        

        return

    def _3d_plot_events(self,events):
        import matplotlib.pyplot as plt
        import numpy as np

        np.save('e_x.npy',events['x'])
        np.save('e_y.npy',events['y'])
        np.save('e_t.npy',events['t'])
        np.save('e_p.npy',events['p'])

        # 示例数据
        x = events['x']
        y = events['y']
        z = events['t']-events['ts_0']

        # 创建三维散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='r', marker='.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('T Label')
        plt.title('3D Scatter Plot')
        
        plt.show()
        plt.savefig('event3d.png')

    
    def draw_save_event_img(self,spikes,
                          save_dir='./work_dirs_trainer/DsecDetection/gt_images/events/'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        _,h,w,Ts=spikes.shape
        
        pos=0
        neg=0
        for t in range(Ts):
            img=np.zeros((h,w,3),dtype=np.uint8)
            img[...,0]=spikes[0,:,:,t]*255
            img[...,2]=spikes[1,:,:,t]*255
            cv2.imwrite(save_dir+str(t)+'.jpg',img)

            # img=np.ones((h,w,3),dtype=np.uint8)
            # img[...,0]=(1-spikes[0,:,:,t])*255
            # img[...,1]=(1-spikes[0,:,:,t])*255
            # img[...,1]=(1-spikes[1,:,:,t])*255
            # img[...,2]=(1-spikes[1,:,:,t])*255
            # cv2.imwrite(save_dir+str(t)+'.jpg',img)

            pos+=spikes[0,:,:,t]
            neg+=spikes[1,:,:,t]

        mimg=np.zeros((h,w,3),dtype=np.uint8)
        pos=(pos>0)*255
        neg=(neg>0)*255
        pos=pos.astype(np.uint8)
        neg=neg.astype(np.uint8)

        mimg[...,0]=pos
        mimg[...,2]=neg
        
        cv2.imwrite(save_dir+'acm.jpg',mimg)
        print(f'image save to {save_dir}')
    
    

