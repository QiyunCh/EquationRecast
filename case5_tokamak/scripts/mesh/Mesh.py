import numpy as np
import h5py
from matplotlib.tri import Triangulation

class Mesh2D:
    def __init__(self, a, b, c, angle, x, z, bound, region=None):
        self.a=a; self.b=b; self.c=c
        self.co=np.cos(angle); self.sn=np.sin(angle)
        self.x=x; self.z=z
        self.bound=bound; self.region=region
        self.nelms=len(self.a)

    def triangles_RZ(self):
        R0, Z0 = self.x, self.z
        R1 = self.x + (self.a + self.b) * self.co
        Z1 = self.z + (self.a + self.b) * self.sn
        R2 = self.x + self.b * self.co - self.c * self.sn
        Z2 = self.z + self.c * self.co + self.b * self.sn
        RV = np.stack([R0, R1, R2], 1)
        ZV = np.stack([Z0, Z1, Z2], 1)
        return RV, ZV

    def triangulation(self, tol=1e-11):
        RV, ZV = self.triangles_RZ()
        P = np.c_[RV.ravel(), ZV.ravel()]
        key = np.round(P / tol).astype(np.int64)
        _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        pts = P[idx]
        tri = inv.reshape(self.nelms, 3)

        x0 = pts[tri[:,0],0]; y0 = pts[tri[:,0],1]
        x1 = pts[tri[:,1],0]; y1 = pts[tri[:,1],1]
        x2 = pts[tri[:,2],0]; y2 = pts[tri[:,2],1]
        area2 = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
        good = np.abs(area2) > 1e-14*np.maximum(1.0, np.maximum.reduce([np.abs(x0)+np.abs(y0),
                                                                        np.abs(x1)+np.abs(y1),
                                                                        np.abs(x2)+np.abs(y2)]))
        tri_good = tri[good]
        elem_ids = np.nonzero(good)[0]
        if tri_good.size == 0:
            raise RuntimeError("No valid triangles after dedup.")
        tri_obj = Triangulation(pts[:,0], pts[:,1], tri_good)
        return tri_obj, elem_ids
