# from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class BBox:
    def __init__(self, label,l,t,r,b):
        super(BBox, self).__init__()
        self.c = label  # class
        self.l = l      # left bound
        self.t = t      # top bound
        self.r = r      # right bound
        self.b = b      # bottom bound
        self.set_xywh()

    def set_xywh(self,):
        self.w = self.r-self.l    # width
        self.h = self.b-self.t    # height 

        self.x = (self.r+self.l)*0.5 # center x of box
        self.y = (self.b+self.t)*0.5 # center y of box

    def get(self, order='cltrb',to_np=False):
        out=[]
        for s in order:
            out.append(getattr(self,s))
        if to_np:
            return np.array(out)
        return out
    
    def area(self,):
        if self.w<0 or self.h<0:
            return 0
        return self.w*self.h

    def clip(self,xmin,ymin,xmax,ymax):
        self.l = max(xmin,self.l)
        self.t = max(ymin,self.t)
        self.r = min(xmax,self.r)
        self.b = min(ymax,self.b)

        self.set_xywh()
    
    def norm(self,xnorm=1.0,ynorm=1.0):
        self.l *= xnorm
        self.r *= xnorm

        self.t *= ynorm
        self.b *= ynorm

        self.set_xywh()
    
    def scale_pad(self,scale=1.0,pad_w=0,pad_h=0):
        self.l = scale*self.l+pad_w
        self.r = scale*self.r+pad_w

        self.t = scale*self.t+pad_h
        self.b = scale*self.b+pad_h

        self.set_xywh()

    def hflip(self, width):
        temp = self.l
        self.l = width - self.r
        self.r = width - temp
        self.set_xywh()
    
    def scale_wh(self,scale_w=1.0,scale_h=1.0):
        self.l = scale_w*self.l
        self.r = scale_w*self.r

        self.t = scale_h*self.t
        self.b = scale_h*self.b

        self.set_xywh()
    
class TBBox(BBox):
    def __init__(self, label,l,t,r,b, track_id, timestamp):
        super(TBBox, self).__init__(label,l,t,r,b)
        self.track_id = track_id
        self.timestamp = timestamp
    
    def reset_ltrb(self,l,t,r,b):
        self.l = l      # left bound
        self.t = t      # top bound
        self.r = r      # right bound
        self.b = b      # bottom bound
        self.set_xywh()
        

    
    

