import numpy as np, torch
import matplotlib.pyplot as plt
from FNO2D import FNO2d
from Train_Canonical_PINN import make_kgrids
import VorticityNS_2D as ns

N=128; L=1.0; RE_STAR=250.0; K_HARD=21; EPS=1e-12; MAXIT=80; SEED=13; NSRC=20; TOL=1e-5
RE_LIST=[50,150,250,350,400]
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load("models/redistributed/redistributed_data.pt",map_location="cpu",weights_only=False)
model=FNO2d(modes_x=ck["modes"],modes_y=ck["modes"],width=ck["width"],in_channels=1,out_channels=1,n_layers=ck["n_layers"]).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval(); [p.requires_grad_(False) for p in model.parameters()]
s_min,s_max,o_min,o_max=ck["S_min"],ck["S_max"],ck["omega_min"],ck["omega_max"]
kx,ky,ikx,iky,lap_s,inv_k,deal=ns.make_spectral_operators(N)
Kx,Ky,_=make_kgrids(N,L,dev); Kx=Kx.double(); Ky=Ky.double()
tp=2*np.pi; krad=((Kx/tp)**2+(Ky/tp)**2).sqrt(); mask=(krad<=K_HARD).to(torch.complex128)
rng=np.random.default_rng(SEED)
src=np.stack([ns.generate_source(N,kx,ky,deal,rng) for _ in range(NSRC)])
Sb=torch.tensor(src,dtype=torch.float64,device=dev)
def bl(f):
    Fh=torch.fft.rfft2(f,dim=(-2,-1),norm="backward"); return torch.fft.irfft2(Fh*mask,s=f.shape[-2:],dim=(-2,-1),norm="backward")
def lap(f):
    Fh=torch.fft.rfft2(f,dim=(-2,-1),norm="backward"); return torch.fft.irfft2(-(Kx**2+Ky**2)*Fh,s=f.shape[-2:],dim=(-2,-1),norm="backward")
def pred(Sf):
    sn=(2*(Sf-s_min)/(s_max-s_min)-1).unsqueeze(1).float()
    with torch.no_grad(): o=model(sn).squeeze(1).double()
    return 0.5*(o+1)*(o_max-o_min)+o_min

# Q1: self-consistency delta trajectory, batched per-source, median
delta_traj={}
for Re in RE_LIST:
    omega=bl(pred(Sb)); w=torch.full((NSRC,),0.35,dtype=torch.float64,device=dev); rprev=None; ds=[]
    for it in range(1,MAXIT+1):
        Seff=bl(Sb+(1/Re-1/RE_STAR)*lap(omega)); r=bl(pred(Seff))-omega
        if rprev is not None:
            dr=r-rprev; num=(rprev*dr).sum(dim=(-2,-1)); den=(dr*dr).sum(dim=(-2,-1)).clamp(min=EPS)
            c=-w*num/den; c=torch.where(torch.isfinite(c),c,w); w=c.clamp(0.02,0.85)
        upd=w.view(NSRC,1,1)*r
        d=(upd.flatten(1).norm(dim=1)/(omega.flatten(1).norm(dim=1)+EPS)).cpu().numpy()
        omega=omega+upd; rprev=r; ds.append(d)
    delta_traj[Re]=np.median(np.stack(ds),axis=1)  # (MAXIT,)
    k_stop=int(np.argmax(delta_traj[Re]<TOL))+1
    print(f"Re={Re:3d}: iters to delta<{TOL:.0e} = {k_stop}")

# Q2: per-shell spectrum benchmark vs recast at Re=400 and 250
k1=np.fft.fftfreq(N)*N; KX,KY=np.meshgrid(k1,k1,indexing='ij'); rb=np.floor(np.sqrt(KX**2+KY**2)+0.5).astype(int)
def shell(a):
    a=a-a.mean(); P=np.abs(np.fft.fft2(a))**2
    return np.bincount(rb.ravel(),weights=P.ravel(),minlength=int(rb.max())+1)
print("\nQ2: per-shell energy benchmark vs recast (1 source)")
S0=Sb[0:1]
for Re in [250,400]:
    ns.RE=float(Re); ref=ns.solve_steady_vorticity(src[0],ikx,iky,lap_s,inv_k,deal)[0]
    omega=bl(pred(S0)); w=torch.tensor([0.35],dtype=torch.float64,device=dev); rprev=None
    for it in range(MAXIT):
        Seff=bl(S0+(1/Re-1/RE_STAR)*lap(omega)); r=bl(pred(Seff))-omega
        if rprev is not None:
            dr=r-rprev; c=-w*(rprev*dr).sum(dim=(-2,-1))/((dr*dr).sum(dim=(-2,-1)).clamp(min=EPS))
            c=torch.where(torch.isfinite(c),c,w); w=c.clamp(0.02,0.85)
        omega=omega+w.view(1,1,1)*r; rprev=r
    om=omega[0].cpu().numpy()
    Eb=shell(ref); Er=shell(om)
    print(f" Re={Re}: shell-energy ratio recast/bench by k-band:")
    for lo,hi in [(1,5),(6,10),(11,15),(16,21),(22,40)]:
        rb_e=Er[lo:hi+1].sum(); bb_e=Eb[lo:hi+1].sum()
        print(f"   k={lo:2d}-{hi:2d}: recast/bench={rb_e/(bb_e+1e-30):.3f}  (bench frac={bb_e/Eb.sum()*100:5.2f}%)")

# ---- Q1 figure ----
plt.rcParams.update({"font.size":20,"axes.titlesize":26,"axes.titleweight":"bold","axes.labelsize":23,"axes.labelweight":"bold","xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":18,"lines.linewidth":3.0,"savefig.dpi":300})
colors=plt.cm.viridis(np.linspace(0,0.85,len(RE_LIST)))
fig,ax=plt.subplots(figsize=(11,7.5))
for c,Re in zip(colors,RE_LIST):
    y=delta_traj[Re]; lab=f"Re = {Re}"+("  (canonical)" if Re==250 else "")
    ax.plot(np.arange(1,len(y)+1),y,color=c,marker="o",markersize=4,markevery=10,label=lab)
ax.axhline(TOL,color="red",linestyle="--",linewidth=2.5,alpha=0.85,label=f"stopping tol = {TOL:.0e}")
ax.set_xlabel("recast iteration"); ax.set_ylabel(r"self-consistency  $\|u^{k}-u^{k-1}\|/\|u^{k-1}\|$")
ax.set_yscale("log"); ax.set_xlim(0,40); ax.grid(True,which="both",alpha=0.3)
ax.set_title("Convergence of the deployment criterion",fontsize=26,fontweight="bold")
ax.legend(loc="upper right",framealpha=0.95)
fig.tight_layout(); fig.savefig("results/Fig_Test4_selfconsistency.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("\nSaved results/Fig_Test4_selfconsistency.png")
