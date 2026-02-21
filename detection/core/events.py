# from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import numba

@numba.njit
def _insert_events(x0,y0,t0,p0, x1,y1,t1,p1):
    len0=len(t0)
    len1=len(t1)
    new_len = len0 + len1
    indexes = np.searchsorted(t0,t1)

    new_t = np.empty(new_len, dtype=t0.dtype)
    new_x = np.empty(new_len, dtype=x0.dtype)
    new_y = np.empty(new_len, dtype=y0.dtype)
    new_p = np.empty(new_len, dtype=p0.dtype)

    new_x[:len0] = x0
    new_y[:len0] = y0
    new_t[:len0] = t0
    new_p[:len0] = p0
    for x,y,t,p, idx in zip(x1,y1,t1,p1,indexes):
        np.insert(new_x, idx, x)
        np.insert(new_y, idx, y)
        np.insert(new_t, idx, t)
        np.insert(new_p, idx, p)
    
    return new_x,new_y,new_t,new_p

class EventBuf:
    def __init__(self, x,y,p,t,ts_start,ts_end):
        super(EventBuf, self).__init__()
        self.x = x
        self.y = y
        self.t = t    
        self.p = p
        self.ts_start = ts_start
        self.ts_end = ts_end
    
    def __len__(self,):
        return len(self.t)
    
    def __getitem__(self, key):
        return getattr(self,key)

    def ts_norm(self,):
        self.t -=self.ts_start
        self.ts_end-=self.ts_start
        self.ts_start = 0
        return 
    
    def insert(self,other):
        # new_length = len(self) + len(other)
        # indexes = np.searchsorted(self.t, other.t)

        # new_t = np.empty(new_length, dtype=self.t.dtype)

        # new_t[:len(self.t)] = self.t
        # for val, idx in zip(other.t, indexes):
        #     np.insert(other.t, idx, val)
        x,y,t,p=_insert_events(self.x,self.y,self.t,self.p,
                               other.x,other.y,other.t,other.p)
        
        self.x = x
        self.y = y
        self.t = t    
        self.p = p

        self.ts_start = min(self.ts_start,other.ts_start)
        self.ts_end = max(self.ts_end,other.ts_end)
    
    def concat(self,other):
        self.x = np.concatenate((self.x, other.x), axis=0)
        self.y = np.concatenate((self.y, other.y), axis=0)
        self.t = np.concatenate((self.t, other.t), axis=0)
        self.p = np.concatenate((self.p, other.p), axis=0)

        self.ts_start = min(self.ts_start,other.ts_start)
        self.ts_end = max(self.ts_end,other.ts_end)
    
    def reset_with_mask(self,mask):
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.t = self.t[mask]
        self.p = self.p[mask]
    
    def crop(self,xmin,ymin,xmax,ymax):
        mask = (self.x >= xmin) & (self.x < xmax) & (self.y>=ymin) & (self.y<ymax)
        self.reset_with_mask(mask)
    
    def pad(self,pad_x=0,pad_y=0):
        if pad_x>=0:
            self.x += pad_x
        else:
            self.x -= abs(pad_x)
        if pad_y>=0:
            self.y += pad_y
        else:
            self.y -= abs(pad_y)
    
    def hflip(self, width):
        self.x = width - self.x
    
    def scale_wh(self,scale_w=1.0,scale_h=1.0):
        self.x = (scale_w*self.x).astype(np.uint16)
        self.y = (scale_h*self.y).astype(np.uint16)

       
        

    


    

