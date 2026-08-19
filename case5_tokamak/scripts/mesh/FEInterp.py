import numpy as np
import h5py
from Mesh import Mesh2D


# ----------------------------- Constants (match FusionIO) -----------------------------
MI = np.array([0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,5,3,2,1,0], dtype=int)
NI = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,2,3,4,5], dtype=int)
NBA = 20

# ----------------------------- Exact FE evaluator -----------------------------
class FEField2D:
    def __init__(self, mesh: Mesh2D, coeff: np.ndarray):
        assert coeff.shape[0] == mesh.nelms and coeff.shape[1] == NBA
        self.mesh = mesh
        self.coeff = coeff
        self._tri = None
        self._elem_ids = None
        self._finder = None

    def _ensure_finder(self):
        if self._finder is None:
            self._tri, self._elem_ids = self.mesh.triangulation()
            self._finder = self._tri.get_trifinder()

    def eval_on_grid(self, R, Z):
        self._ensure_finder()
        RR, ZZ = np.meshgrid(R, Z)
        tri_id = self._finder(RR.ravel(), ZZ.ravel())
        inside = tri_id >= 0
        vals = np.full(RR.size, np.nan, dtype=float)

        eidx = np.full(RR.size, -1, dtype=int)
        eidx[inside] = self._elem_ids[tri_id[inside]]

        idxs = np.where(inside)[0]
        if idxs.size:
            R_in = RR.ravel()[idxs]; Z_in = ZZ.ravel()[idxs]; e_in = eidx[idxs]
            dotL = (R_in - self.mesh.x[e_in]) * self.mesh.co[e_in] + (Z_in - self.mesh.z[e_in]) * self.mesh.sn[e_in]
            xi   = dotL - self.mesh.b[e_in]   # exact shift
            eta  = -(R_in - self.mesh.x[e_in]) * self.mesh.sn[e_in] + (Z_in - self.mesh.z[e_in]) * self.mesh.co[e_in]

            xip = np.ones((xi.size, 6)); etap = np.ones((eta.size, 6))
            for k in range(1, 6):
                xip[:, k]  = xip[:, k-1]  * xi
                etap[:, k] = etap[:, k-1] * eta

            C = self.coeff[e_in, :]
            X = xip[:, MI]
            Y = etap[:, NI]
            vals[idxs] = np.einsum('ij,ij,ij->i', C, X, Y)

        return vals.reshape(len(Z), len(R))

def read_field_coeffs(h5path, tg, varname, nelms):
    with h5py.File(h5path, "r") as f:
        base = f"{tg}/fields"
        if base not in f:
            raise RuntimeError(f"'{base}' not found")
        g = f[base]
        path = f"{base}/{varname}" if varname in g else None
        if path is None:
            lower = {k.lower(): k for k in g.keys()}
            if varname.lower() in lower:
                path = f"{base}/{lower[varname.lower()]}"
        if path is None:
            raise RuntimeError(f"{varname} not under {base}")
        arr = np.array(f[path][...], dtype=float)

        # Normalize to (nelms, 20) polynomial coeffs like C++: data[e, p], p=0..19
        NBA = 20
        if arr.ndim == 1:
            if arr.size != nelms:
                raise ValueError("1D field length != nelms")
            coeff = np.zeros((nelms, NBA), dtype=float)
            coeff[:, 0] = arr
        elif arr.ndim == 2:
            if arr.shape == (nelms, NBA):
                coeff = arr
            elif arr.shape == (NBA, nelms):
                coeff = arr.T
            elif arr.shape == (nelms, 1):
                coeff = np.zeros((nelms, NBA), dtype=float)
                coeff[:, 0] = arr[:, 0]
            else:
                raise ValueError(f"Unexpected shape {arr.shape} (want (nelms,20) or (20,nelms) or (nelms,1))")
        else:
            raise ValueError(f"Unsupported ndim={arr.ndim} for field dataset")

        return coeff