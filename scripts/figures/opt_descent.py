"""Original quadratic paths and deterministic synthetic logistic Newton convergence."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-06-gradient-newton"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":11, "figure.facecolor":"white", "axes.facecolor":"white"})
Q = np.diag([1.,40.])
path = [np.array([3.,3.])]
values = [.5*path[-1] @ Q @ path[-1]]
for _ in range(200):
    path.append(path[-1] - Q @ path[-1]/40)
    values.append(.5*path[-1] @ Q @ path[-1])
path = np.array(path)
fig,ax = plt.subplots(1,2,figsize=(8.6,3.8),constrained_layout=True)
for j in range(2):
    xx = np.linspace(-.2,3.3,200)
    yy = np.linspace(-.15,3.3 if j==0 else .4,200)
    X,Y = np.meshgrid(xx,yy)
    ax[j].contour(X,Y,.5*(X*X+40*Y*Y),levels=[.05,.2,.5,1,2,4,10,40,100,180],colors="#94a3b8")
    ax[j].plot(path[:,0],path[:,1],".-",color="#2563eb",label="Gradient, step 1/40",markersize=3)
    ax[j].plot([3,0],[3,0],"o--",color="#ea580c",label="Newton, one full step")
    ax[j].scatter([0],[0],marker="*",s=90,color="#16a34a",zorder=5)
    ax[j].set(xlabel="x₁",ylabel="x₂",title="Full path" if j==0 else "Near the flat direction",xlim=(-.2,3.3),ylim=(-.15,3.3 if j==0 else .4))
ax[0].legend(fontsize=10)
fig.savefig(OUT/"descent-paths.png",dpi=150)
plt.close(fig)

rng=np.random.default_rng(16)
A=rng.normal(size=(160,5))
truth=np.array([.8,-.5,.3,0.,.4])
prob=1/(1+np.exp(-(A @ truth)))
y=(rng.random(len(A))<prob).astype(float)
rho=.1
def evaluate(w):
    z=A @ w
    p=1/(1+np.exp(-z))
    return (np.mean(np.logaddexp(0,z)-y*z)+rho*(w @ w)/2,
            A.T @ (p-y)/len(y)+rho*w,
            (A.T*(p*(1-p))) @ A/len(y)+rho*np.eye(A.shape[1]))
w=np.zeros(5)
gradient_norms=[]
for _ in range(30):
    value,g,H=evaluate(w)
    gradient_norms.append(np.linalg.norm(g))
    if gradient_norms[-1]<1e-9:
        break
    d=np.linalg.solve(H,-g)
    step=1.
    for _ in range(50):
        candidate=w+step*d
        if evaluate(candidate)[0]<=value+.1*step*(g @ d):
            w=candidate
            break
        step *= .5
    else:
        raise RuntimeError("backtracking failed")
assert gradient_norms[-1]<1e-9
fig,ax=plt.subplots(1,2,figsize=(8.6,3.8),constrained_layout=True)
ax[0].semilogy(values,color="#2563eb")
ax[0].set(xlabel="Accepted gradient steps",ylabel="Quadratic objective gap",title="Fixed gradient step on Q = diag(1, 40)")
ax[1].semilogy(np.maximum(gradient_norms,1e-17),"o-",color="#ea580c")
ax[1].set(xlabel="Accepted Newton steps",ylabel="Gradient norm",title="Synthetic regularized logistic problem")
fig.savefig(OUT/"convergence.png",dpi=150)
plt.close(fig)
