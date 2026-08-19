#!/usr/bin/env python3
"""Effective-source diagnostics for NS recast: shows that the SOLUTION spectrum
is amplitude-only (shape invariant) while the effective source's NONLINEAR
composition (u.grad w) grows and broadens with Re. Model-independent (uses the
high-fidelity reference solutions + the canonical operator)."""
import numpy as np, h5py, torch
import matplotlib.pyplot as plt
from Train_PINN_Canonical import make_kgrids
import VorticityNS_2D as ns
N=128;L=1.0;RES=250.0;EPS=1e-12;SEED=13;NSRC=15
RE=[50,150,250,350,400]
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
kx,ky,ikx,iky,ls,ik,deal=ns.make_spectral_operators(N)
Kx,Ky,_=make_kgrids(N,L,dev);Kx=Kx.double();Ky=Ky.double()
K2=Kx**2+Ky**2; K2inv=torch.where(K2>0,1.0/K2,torch.zeros_like(K2))
rng=np.random.default_rng(SEED);src=[ns.generate_source(N,kx,ky,deal,rng) for _ in range(NSRC)]
def adv(om):
    Oh=torch.fft.rfft2(om,dim=(-2,-1),norm="backward");psih=Oh*K2inv
    u=torch.fft.irfft2(1j*Ky*psih,s=om.shape[-2:],dim=(-2,-1),norm="backward")
    v=torch.fft.irfft2(-1j*Kx*psih,s=om.shape[-2:],dim=(-2,-1),norm="backward")
    ox=torch.fft.irfft2(1j*Kx*Oh,s=om.shape[-2:],dim=(-2,-1),norm="backward")
    oy=torch.fft.irfft2(1j*Ky*Oh,s=om.shape[-2:],dim=(-2,-1),norm="backward")
    return u*ox+v*oy
def lapf(om):
    Oh=torch.fft.rfft2(om,dim=(-2,-1),norm="backward");return torch.fft.irfft2(-K2*Oh,s=om.shape[-2:],dim=(-2,-1),norm="backward")
k1=np.fft.fftfreq(N)*N;KX,KY=np.meshgrid(k1,k1,indexing='ij');rb=np.floor(np.sqrt(KX**2+KY**2)+0.5).astype(int);kmax=int(rb.max())
def shell(a):
    a=a-a.mean();P=np.abs(np.fft.fft2(a))**2;return np.bincount(rb.ravel(),weights=P.ravel(),minlength=kmax+1)
Ew={};Enl={};Elin={};Es={};ratio={};rms={}
for Re in RE:
    sw=np.zeros(kmax+1);snl=np.zeros(kmax+1);slin=np.zeros(kmax+1);ss=np.zeros(kmax+1);rr=[];rm=[]
    for s in range(NSRC):
        ns.RE=float(Re);w=ns.solve_steady_vorticity(src[s],ikx,iky,ls,ik,deal)[0]
        wt=torch.tensor(w,dtype=torch.float64,device=dev)
        NL=adv(wt);LIN=(1.0/RES)*lapf(wt);Se=NL-LIN
        sw+=shell(w);snl+=shell(NL.cpu().numpy());slin+=shell(LIN.cpu().numpy());ss+=shell(Se.cpu().numpy())
        rr.append(float(NL.norm()/(LIN.norm()+EPS)));rm.append(float((wt**2).mean().sqrt()))
    Ew[Re]=sw/NSRC;Enl[Re]=snl/NSRC;Elin[Re]=slin/NSRC;Es[Re]=ss/NSRC
    ratio[Re]=np.median(rr);rms[Re]=np.median(rm)
# recast error for overlay
err={}
try:
    with h5py.File("results/test3_compare.h5","r") as f:
        Rl=list(f["Re_list"][:]);e=f["canonical_dataonly"]["err_l2"][:].mean(axis=1)*100
        for Re in RE: err[Re]=e[Rl.index(Re)]
except Exception: err=None

plt.rcParams.update({"font.size":18,"axes.titlesize":22,"axes.titleweight":"bold","axes.labelsize":20,"axes.labelweight":"bold","xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":15,"lines.linewidth":2.6,"savefig.dpi":300})
cols=plt.cm.viridis(np.linspace(0,0.85,len(RE)))
fig,ax=plt.subplots(2,2,figsize=(17,13))
kk=np.arange(kmax+1)

# (a) composition ratio + error vs Re
a=ax[0,0]
a.plot(RE,[ratio[r] for r in RE],marker="s",color="tab:red",label=r"$\|u\!\cdot\!\nabla\omega\|/\|\frac{1}{Re^*}\Delta\omega\|$")
a.set_xlabel("Re");a.set_ylabel("nonlinear / linear norm ratio",color="tab:red")
a.tick_params(axis="y",labelcolor="tab:red");a.grid(True,alpha=0.3)
a.axvline(RES,color="red",ls="--",lw=2,alpha=0.6)
a.set_title("(a) Effective-source composition shifts with Re")
if err:
    a2=a.twinx();a2.plot(RE,[err[r] for r in RE],marker="o",color="tab:blue",label="recast rel $L^2$ error")
    a2.set_ylabel("recast rel $L^2$ error (%)",color="tab:blue");a2.tick_params(axis="y",labelcolor="tab:blue")
    l1,la1=a.get_legend_handles_labels();l2,la2=a2.get_legend_handles_labels();a.legend(l1+l2,la1+la2,loc="upper left")
else: a.legend(loc="upper left")

# (b) normalized solution spectrum (shape collapse)
b=ax[0,1]
for c,Re in zip(cols,RE):
    y=Ew[Re]/Ew[Re].sum()
    b.semilogy(kk,y+1e-30,color=c,label=f"Re={Re}")
b.set_xlim(0,40);b.set_ylim(1e-6,1);b.set_xlabel("radial wavenumber $k$");b.set_ylabel(r"normalized $E_\omega(k)$")
b.set_title(r"(b) Solution spectrum: shape invariant (amplitude-only)");b.grid(True,which="both",alpha=0.3);b.legend()

# (c) nonlinear-term spectrum (absolute) grows + broadens
c_=ax[1,0]
for c,Re in zip(cols,RE):
    c_.semilogy(kk,Enl[Re]+1e-30,color=c,label=f"Re={Re}")
c_.set_xlim(0,40);c_.set_xlabel("radial wavenumber $k$");c_.set_ylabel(r"$E_{u\cdot\nabla\omega}(k)$ (absolute)")
c_.set_title(r"(c) Nonlinear term $u\!\cdot\!\nabla\omega$: grows \& broadens with Re");c_.grid(True,which="both",alpha=0.3);c_.legend()

# (d) S_eff decomposition at Re=400
d=ax[1,1]
d.semilogy(kk,Es[400]+1e-30,color="k",lw=3.2,label=r"$E_{S_{eff}}(k)$")
d.semilogy(kk,Elin[400]+1e-30,color="tab:blue",ls="--",label=r"$E_{\frac{1}{Re^*}\Delta\omega}(k)$ (linear)")
d.semilogy(kk,Enl[400]+1e-30,color="tab:red",ls="--",label=r"$E_{u\cdot\nabla\omega}(k)$ (nonlinear)")
d.set_xlim(0,40);d.set_xlabel("radial wavenumber $k$");d.set_ylabel("shell energy")
d.set_title("(d) Effective-source decomposition at Re=400");d.grid(True,which="both",alpha=0.3);d.legend()
fig.tight_layout();fig.savefig("results/Fig_Test3_effsource_diag.png",dpi=300,bbox_inches="tight");plt.close(fig)
print("ratio:",{r:round(ratio[r],3) for r in RE})
print("Saved results/Fig_Test3_effsource_diag.png")
