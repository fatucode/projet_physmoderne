import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
trapz=np.trapezoid
 
HBAR=MASS=1.0
NX=5000; x=np.linspace(-200,200,NX); dx=x[1]-x[0]
K0,X0,SIG=2.0,-50.0,5.0
 
def paquet():
    p=np.exp(-((x-X0)**2)/(4*SIG**2))*np.exp(1j*K0*x)
    return p/np.sqrt(trapz(np.abs(p)**2,x))
 
def make_step(V,dt):
    r=1j*HBAR*dt/(4*MASS*dx**2); s=1j*dt/(2*HBAR)*V
    Ab=np.zeros((3,NX),dtype=complex); Ab[0,1:]=-r;Ab[1,:]=1+2*r+s;Ab[2,:-1]=-r
    Ab[1,0]=1;Ab[0,1]=0;Ab[1,-1]=1;Ab[2,-2]=0
    def step(p):
        b=(1-2*r-s)*p; b[1:-1]+=r*(p[2:]+p[:-2]);b[0]=0;b[-1]=0
        return solve_banded((1,1),Ab,b)
    return step
 
# parametres ou le tunnel est BIEN visible
a0,V00=1.0,2.5
cibles={0:None,22:None,45:None}
step=make_step(np.where(np.abs(x)<a0/2,V00,0.0),0.01)
psi=paquet()
for n in range(4600):
    tc=n*0.01
    for k in list(cibles):
        if cibles[k] is None and abs(tc-k)<0.005:
            cibles[k]=(np.abs(psi)**2).copy()
    psi=step(psi)
T=trapz(np.abs(psi)**2*(x>a0/2),x)
 
nv=cibles[0].max()
inc=cibles[0]/nv; coll=cibles[22]/nv; fin=cibles[45]/nv
# on plafonne la collision pour qu'elle ne defonce pas le graphe
coll=np.clip(coll,0,1.15)
 
plt.style.use('default')
fig,ax=plt.subplots(figsize=(13,6.5))
fig.patch.set_facecolor('white')
 
# barriere : rectangle plein semi-transparent + contour
h=1.25
ax.fill_between(x,0,h,where=(np.abs(x)<a0/2),color='#d62728',alpha=0.15)
ax.plot(x,np.where(np.abs(x)<a0/2,h,np.nan),color='#d62728',lw=0)
ax.vlines([-a0/2,a0/2],0,h,color='#d62728',lw=2.5)
ax.hlines(h,-a0/2,a0/2,color='#d62728',lw=2.5)
 
ax.plot(x,inc, color='#1f5fd6',lw=3,label="$t_0$ : paquet incident")
ax.plot(x,coll,color='#ff8c00',lw=2.2,label="$t_1$ : collision (interférences)")
ax.plot(x,fin, color='#2ca02c',lw=3,label="$t_2$ : état final")
 
ax.annotate("RÉFLÉCHI\n(~70%)",(-42,0.45),color='#1a7a1a',fontsize=13,
            ha='center',fontweight='bold')
ax.annotate(f"TRANSMIS\npar effet tunnel\n(~{T*100:.0f}%)",(38,0.28),color='#1a7a1a',fontsize=13,
            ha='center',fontweight='bold')
ax.annotate("Barrière $V_0$",(0,1.30),color='#d62728',fontsize=11,
            ha='center',fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3",fc='white',ec='#d62728',alpha=0.9))
 
# fleche vers le paquet transmis
ax.annotate("",xy=(33,0.07),xytext=(38,0.20),
            arrowprops=dict(arrowstyle="->",color='#1a7a1a',lw=1.8))
 
ax.set_xlabel("Position  $x$",fontsize=13)
ax.set_ylabel(r"Densité de probabilité  $|\Psi(x,t)|^2$  (u.a.)",fontsize=13)
ax.set_title("Effet tunnel : division du paquet d'ondes face à la barrière",
             fontsize=15,fontweight='bold',pad=15)
ax.set_xlim(-95,95); ax.set_ylim(0,1.5)
ax.legend(fontsize=12,loc='upper left',framealpha=0.95)
ax.grid(True,ls=":",alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig("beau.png",dpi=140)
print(f"OK - transmission T = {T:.3f}")
