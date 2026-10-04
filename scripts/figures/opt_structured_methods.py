"""Original synthetic figures for OPT07–08. No external data.
Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/opt_structured_methods.py
Writes only the OPT07 and OPT08 asset folders.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]/'src/assets/optimization'
OUT7=ROOT/'opt-07-constrained-interior-point';OUT8=ROOT/'opt-08-applications-modern-methods'
OUT7.mkdir(parents=True,exist_ok=True);OUT8.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white'})
def center(t,x):
    def value(z):return .5*t*(z-2)**2-np.log(z)-np.log1p(-z)
    for _ in range(100):
        grad=t*(x-2)-1/x+1/(1-x);hess=t+1/x**2+1/(1-x)**2
        if grad**2/hess<1e-16:break
        step=-grad/hess;alpha=1.
        for _ in range(100):
            trial=x+alpha*step
            if 0<trial<1 and value(trial)<=value(x)+.01*alpha*grad*step:break
            alpha*=.5
        else:raise RuntimeError('Line search failed')
        x=trial
    else:raise RuntimeError("Centering iteration limit reached")
    return x
weights=np.logspace(0,3,35);points=[];gaps=[];errors=[];x=.5
for t in weights:
    x=center(t,x);lo=1/(t*x);up=1/(t*(1-x))
    primal=.5*(x-2)**2;dual=2-up-.5*(-2-lo+up)**2
    points.append(x);gaps.append(primal-dual);errors.append(primal-.5)
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
axs[0].semilogx(weights,points,color='#8553a8',lw=2)
axs[0].axhline(1,color='#444444',ls='--',label='Optimal boundary')
axs[0].set(xlabel='Barrier weight t',ylabel='Interior solution x(t)',title='Central path',ylim=(.5,1.04));axs[0].legend()
axs[1].loglog(weights,gaps,label='Computed duality gap',color='#337d9f')
axs[1].loglog(weights,errors,label='Objective error',color='#b65332')
axs[1].set(xlabel='Barrier weight t',ylabel='Objective units',title='Certificate versus error');axs[1].legend()
fig.savefig(OUT7/'central-path.png',dpi=150);plt.close(fig)
grid=np.linspace(-1,2,301)
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
for ax,values,title,minimum in [(axs[0],np.maximum(grid,1-grid),'Infeasible interval',(.5,.5)),
                              (axs[1],np.abs(grid),'Boundary-only feasibility',(0,0))]:
    ax.plot(grid,values,lw=2,color='#8553a8');ax.scatter(*minimum,color='#b65332',zorder=3)
    ax.axhline(0,color='#777777',ls=':');ax.set(xlabel='x',ylabel='Maximum violation',title=title)
    ax.annotate(f'Minimum {minimum[1]:g}',minimum,xytext=(minimum[0]+.3,minimum[1]+.3),arrowprops={'arrowstyle':'->'})
fig.savefig(OUT7/'feasibility.png',dpi=150);plt.close(fig)
rng=np.random.default_rng(8);n=48;clean=np.r_[np.zeros(16),np.ones(16),np.full(16,.3)]
y=clean+rng.normal(0,.16,n);D=np.diff(np.eye(n),axis=0);lam=.4;rho=4.
factor=cho_factor(np.eye(n)+rho*D.T@D);x=y.copy();z=np.zeros(n-1);u=z.copy();r=[];s=[]
for k in range(2000):
    x=cho_solve(factor,y+rho*D.T@(z-u));old=z.copy();v=D@x+u
    z=np.sign(v)*np.maximum(np.abs(v)-lam/rho,0);u+=D@x-z
    r.append(np.linalg.norm(D@x-z));s.append(rho*np.linalg.norm(D.T@(z-old)))
    if max(r[-1],s[-1])<1e-7:break
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
axs[0].plot(y,'.',color='#8796a5',label='Noisy');axs[0].plot(clean,color='#222222',label='Synthetic truth');axs[0].plot(x,color='#8553a8',label='ADMM estimate')
axs[0].set(xlabel='Sample index',ylabel='Signal value',title='Regularized synthetic signal');axs[0].legend()
axs[1].semilogy(np.arange(1,len(r)+1),r,label='Primal residual',color='#337d9f');axs[1].semilogy(np.arange(1,len(s)+1),s,label='Dual residual',color='#b65332')
axs[1].set(xlabel='Iteration',ylabel='Residual norm',title='Two stopping diagnostics');axs[1].legend()
fig.savefig(OUT8/'denoising.png',dpi=150);plt.close(fig)
target=np.array([-.1,.6,.5]);optimum=np.array([0,.55,.45]);x=np.full(3,1/3);trajectory=[];gaps=[];errors=[]
for k in range(1000):
    grad=x-target;v=np.eye(3)[np.argmin(grad)];gap=grad@(x-v)
    trajectory.append(x.copy());gaps.append(gap);errors.append(.5*np.sum((x-target)**2)-.5*np.sum((optimum-target)**2))
    if gap<1e-4:break
    gamma=2/(k+2);x=(1-gamma)*x+gamma*v
corners=np.array([[0,0],[1,0],[.5,np.sqrt(3)/2]]);xy=np.array(trajectory)@corners
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
tri=np.vstack([corners,corners[:1]]);axs[0].plot(tri[:,0],tri[:,1],color='#444444');axs[0].plot(xy[:35,0],xy[:35,1],'.-',color='#8553a8',lw=1)
ref=optimum@corners;axs[0].scatter(*ref,marker='*',s=100,color='#b65332',label='Optimum');axs[0].set_aspect('equal');axs[0].set(xlabel='Simplex coordinate',ylabel='Simplex coordinate',title='First 35 feasible iterates');axs[0].legend()
axs[1].loglog(np.arange(1,len(gaps)+1),gaps,color='#337d9f',label='Frank–Wolfe gap');axs[1].loglog(np.arange(1,len(errors)+1),np.maximum(errors,1e-16),color='#b65332',label='Objective error')
axs[1].set(xlabel='Iteration',ylabel='Objective units',title='Error bounded by gap');axs[1].legend()
fig.savefig(OUT8/'conditional-gradient.png',dpi=150);plt.close(fig)
print('Generated four OPT07–08 figures')
