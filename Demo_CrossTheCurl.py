import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from VISolver.Domains.Affine import Affine

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP
from IPython import embed

def Demo():
    # A(x-b) = Ax - Ab = 0 --> x=b
    # Define Dimension and Domain
    Askew = np.array([[0,-1],[1,0]])
    offset = np.ones(2)
    x = np.array([0,1])

    print('Askew^2',np.dot(Askew,Askew))
    print('Askew',Askew)

    Domain = Affine(A=Askew,b=-np.dot(Askew,offset))
    F = Domain.F(x)
    J = Domain.J(x)
    JF = np.dot(J,F)
    JTF = np.dot(J.T,F)
    vec = np.dot((J-J.T),F)
    uF = F/np.linalg.norm(F)
    uvec = vec/np.linalg.norm(vec)
    uJF = JF/np.linalg.norm(JF)
    uJTF = JTF/np.linalg.norm(JTF)
    print('#'*20)
    print('a',0,'b',0)
    print('-F',-F)
    print('vec',vec)
    between = np.array([(-uF)*np.cos(0.5*np.pi*t) + uvec*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
    norms = np.linalg.norm(between+x,axis=1)
    # print(between+x)
    # print(norms)
    idx = np.argmin(norms)
    err = norms[idx]
    print(err)
    print(between[idx])
    # print((x+uvec)/(uF+uvec))
    # print(between)

    # Construct figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(u)+offset[0],np.sin(u)+offset[1],color='k',alpha=0.4)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    ax.plot([offset[0]],[offset[1]],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
    ax.plot([x[0]],[x[1]],'ko',label='x')
    Farrow = np.vstack((x[None],x[None]-uF[None]))
    ax.plot(Farrow[:,0],Farrow[:,1],'->',label='-F')
    JFFarrow = np.vstack((x[None],x[None]+uJF[None]))
    ax.plot(JFFarrow[:,0],JFFarrow[:,1],'->',label='JF')
    JTFFarrow = np.vstack((x[None],x[None]-uJTF[None]))
    ax.plot(JTFFarrow[:,0],JTFFarrow[:,1],'-.>',label='-JTF')
    vecarrow = np.vstack((x[None],x[None]+uvec[None]))
    ax.plot(vecarrow[:,0],vecarrow[:,1],'-->',label='vec')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.1)
    plt.show()

    # for a in np.logspace(-1,1,num=3):
    #     for b in np.logspace(-1,1,num=3):
    #         Asym = np.array([[a,0],[0,b]])
    #         Domain = Affine(A=Asym+Askew,b=-np.dot(Askew,offset))
    #         F = Domain.F(x)
    #         J = Domain.J(x)
    #         JF = np.dot(J,F)
    #         JTF = np.dot(J.T,F)
    #         JmJT = 0.5*(J-J.T)
    #         print('A diff',np.linalg.norm(Asym+Askew-J))
    #         print('skew diff',np.linalg.norm(JmJT-Askew))
    #         Fskew = np.dot(JmJT,x) + Domain.b
    #         vec = np.dot(JmJT,Fskew)
    #         # vec = np.dot(JmJT,F)
    #         uF = F/np.linalg.norm(F)
    #         uvec = vec/np.linalg.norm(vec)
    #         uJF = JF/np.linalg.norm(JF)
    #         uJTF = JTF/np.linalg.norm(JTF)
    #         perp = np.dot(F,vec)
    #         rand = np.random.rand(2,2)
    #         randskew = rand - rand.T
    #         randperp = np.dot(F,np.dot(randskew,F))
    #         print('#'*20)
    #         print('a',a,'b',b)
    #         print('-F',-F)
    #         print('vec',vec)
    #         between = np.array([(-uF)*np.cos(0.5*np.pi*t) + uvec*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
    #         norms = np.linalg.norm(between+x,axis=1)
    #         # print(between+x)
    #         # print(norms)
    #         idx = np.argmin(norms)
    #         err = norms[idx]
    #         print(err)
    #         print(between[idx])
    #         # print((x+uvec)/(uF+uvec))
    #         # print(between)
    #         # print((F-x)/(vec))
    #         # print((x+uvec)/(uF+uvec))
    #         # print('perp',perp)
    #         # print('randperp',randperp)

    #         # Construct figure
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         u = np.linspace(0, 2 * np.pi, 100)
    #         plt.plot(np.cos(u)+offset[0],np.sin(u)+offset[1],color='k',alpha=0.4)
    #         ax.set_xlabel('X axis')
    #         ax.set_ylabel('Y axis')

    #         ax.plot([offset[0]],[offset[1]],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
    #         ax.plot([x[0]],[x[1]],'ko',label='x')
    #         Farrow = np.vstack((x[None],x[None]-uF[None]))
    #         ax.plot(Farrow[:,0],Farrow[:,1],'->',label='-F')
    #         JFFarrow = np.vstack((x[None],x[None]+uJF[None]))
    #         ax.plot(JFFarrow[:,0],JFFarrow[:,1],'->',label='JF')
    #         JTFFarrow = np.vstack((x[None],x[None]-uJTF[None]))
    #         ax.plot(JTFFarrow[:,0],JTFFarrow[:,1],'->',label='-JTF')
    #         vecarrow = np.vstack((x[None],x[None]+uvec[None]))
    #         ax.plot(vecarrow[:,0],vecarrow[:,1],'->',label='vec')
    #         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
    #         plt.show()

    # Define Dimension and Domain
    Askew = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    offset = np.zeros(3) # np.array([1,-1,1])
    x = np.array([0,1,0]) + offset
    print(np.linalg.eig(Askew))

    Domain = Affine(A=Askew,b=-np.dot(Askew,offset))
    F = Domain.F(x)
    J = Domain.J(x)
    JF = np.dot(J,F)
    JTF = np.dot(J.T,F)
    JmJT = 0.5*(J-J.T)
    Fskew = np.dot(JmJT,x) + Domain.b
    # vec = np.dot(JmJT,Fskew)
    vec = np.dot(JmJT,F)
    factor = np.dot(2*vec,2*vec)/(np.dot(JF,JF)+np.dot(JTF,JTF)+2*np.linalg.norm(JF)*np.linalg.norm(JTF))
    JF = JF*np.linalg.norm(F)/np.linalg.norm(JF)
    JTF = JTF*np.linalg.norm(F)/np.linalg.norm(JTF)
    vec = vec*np.linalg.norm(F)/np.linalg.norm(vec)
    print('factor',factor)
    new = (1-factor)*(-F) + factor*vec
    curl = np.array([J[2,1]-J[1,2],J[0,2]-J[2,0],J[1,0]-J[0,1]])
    # vec = np.dot((J-J.T),F)
    print('#'*20)
    print('a',0,'b',0,'c',0)
    print('-F',-F)
    print('vec',vec)
    print('JF',JF)
    print('-JTF',-JTF)
    print('new',new)
    uF = F/np.linalg.norm(F)
    uvec = vec/np.linalg.norm(vec)
    uJF = JF/np.linalg.norm(JF)
    uJTF = JTF/np.linalg.norm(JTF)
    ucurl = curl/np.linalg.norm(curl)
    unew = new/np.linalg.norm(new)
    print('curl is perp')
    print(np.dot(ucurl,uvec))
    print(np.dot(ucurl,uF))
    # print('vec_score',np.dot(uvec,x))
    # print('JF_score',np.dot(uJF,x))
    # print('-JTF_score',np.dot(-uJTF,x))
    between = np.array([(-uF)*np.cos(0.5*np.pi*t) + uvec*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
    norms = np.linalg.norm(between+x,axis=1)
    # print(between+x)
    # print(norms)
    idx = np.argmin(norms)
    err = norms[idx]
    print(err)
    print(between[idx])

    # Construct figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=90.,elev=90.)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X = 1 * np.outer(np.cos(u), np.sin(v)) + offset[0]
    Y = 1 * np.outer(np.sin(u), np.sin(v)) + offset[1]
    Z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + offset[2]
    # ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,
    #                   linewidth=0,antialiased=False)
    ax.plot_wireframe(X,Y,Z,rstride=10,cstride=10,color='k',alpha=0.4)
    ax.set_xlim3d(np.min(X),np.max(X))
    ax.set_ylim3d(np.min(Y),np.max(Y))
    ax.set_zlim3d(np.min(Z),np.max(Z))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    s=5
    Fxs = J[0,0]*X + J[0,1]*Y*0 + J[0,2]*Z + Domain.b[0]
    Fys = J[1,0]*X + J[1,1]*Y*0 + J[1,2]*Z + Domain.b[1]
    Fzs = J[2,0]*X + J[2,1]*Y*0 + J[2,2]*Z + Domain.b[2]
    ax.quiver(X[::s,::s], 0*Y[::s,::s], Z[::s,::s], -Fxs[::s,::s], -Fys[::s,::s], -Fzs[::s,::s], length=0.1, normalize=True)

    ax.plot([offset[0]],[offset[1]],[offset[2]],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
    ax.plot([x[0]],[x[1]],[x[2]],'ko',label='x')
    Farrow = np.vstack((x[None],x[None]-uF[None]))
    ax.plot(Farrow[:,0],Farrow[:,1],Farrow[:,2],'->',label='-F')
    JFFarrow = np.vstack((x[None],x[None]+uJF[None]))
    ax.plot(JFFarrow[:,0],JFFarrow[:,1],JFFarrow[:,2],'->',label='JF')
    JTFFarrow = np.vstack((x[None],x[None]-uJTF[None]))
    ax.plot(JTFFarrow[:,0],JTFFarrow[:,1],JTFFarrow[:,2],'-.>',label='-JTF')
    vecarrow = np.vstack((x[None],x[None]+uvec[None]))
    ax.plot(vecarrow[:,0],vecarrow[:,1],vecarrow[:,2],'-->',label='vec')
    newarrow = np.vstack((x[None],x[None]+unew[None]))
    ax.plot(newarrow[:,0],newarrow[:,1],newarrow[:,2],'-.>',label='new')
    curlarrow = np.vstack((x[None],x[None]+ucurl[None]))
    ax.plot(curlarrow[:,0],curlarrow[:,1],curlarrow[:,2],'-->',label='curl')
    # ax.arrow(x[0],x[1],x[2],Farrow[1][0],Farrow[1][1],Farrow[1][2],label='-F')
    # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax.set_zlabel('f(X,Y)',rotation=90)
    # ax.text2D(0.05, 0.95, 'Steepest Descent on the\nRosenbrock Function',
    #           transform=ax.transAxes)
    # lw = cycle(range(2*len(steps),0,-2))
    # plt.ion()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
    plt.show()
    # embed()
    for a in np.logspace(-1,1,num=2):
        for b in np.logspace(-1,1,num=2):
            for c in np.logspace(-1,1,num=2):
                Asym = np.array([[a,0,0],[0,b,0],[0,0,c]])
                Domain = Affine(A=Asym+Askew,b=-np.dot(Askew,offset))
                F = Domain.F(x)
                J = Domain.J(x)
                JF = np.dot(J,F)
                JTF = np.dot(J.T,F)
                JmJT = 0.5*(J-J.T)
                Fskew = np.dot(JmJT,x) + Domain.b
                # vec = np.dot(JmJT,Fskew)
                vec = np.dot(JmJT,F)
                factor = np.dot(2*vec,2*vec)/(np.dot(JF,JF)+np.dot(JTF,JTF)+2*np.linalg.norm(JF)*np.linalg.norm(JTF))
                JF = JF*np.linalg.norm(F)/np.linalg.norm(JF)
                JTF = JTF*np.linalg.norm(F)/np.linalg.norm(JTF)
                vec = vec*np.linalg.norm(F)/np.linalg.norm(vec)
                print('factor',factor)
                new = (1-factor)*(-F) + factor*vec
                curl = np.array([J[2,1]-J[1,2],J[0,2]-J[2,0],J[1,0]-J[0,1]])
                # vec = np.dot((J-J.T),F)
                perp = np.dot(F,vec)
                rand = np.random.rand(3,3)
                randskew = rand - rand.T
                randperp = np.dot(F,np.dot(randskew,F))
                print('#'*20)
                print('a',a,'b',b,'c',c)
                print('-F',-F)
                print('vec',vec)
                print('JF',JF)
                print('-JTF',-JTF)
                uF = F/np.linalg.norm(F)
                uvec = vec/np.linalg.norm(vec)
                uJF = JF/np.linalg.norm(JF)
                uJTF = JTF/np.linalg.norm(JTF)
                unew = new/np.linalg.norm(new)
                ucurl = curl/np.linalg.norm(curl)
                # print('vec_score',np.dot(uvec,x))
                # print('JF_score',np.dot(uJF,x))
                # print('-JTF_score',np.dot(-uJTF,x))
                # print((F-x)/(vec))
                # print('perp',perp)
                # print('randperp',randperp)
                between = np.array([(-uF)*np.cos(0.5*np.pi*t) + uvec*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
                norms = np.linalg.norm(between+x,axis=1)
                # print(between+x)
                # print(norms)
                idx = np.argmin(norms)
                err = norms[idx]
                print(err)
                # print(between[idx])
                betweenJF = np.array([(-uF)*np.cos(0.5*np.pi*t) + uJF*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
                normsJF = np.linalg.norm(betweenJF+x,axis=1)
                idxJF = np.argmin(normsJF)
                errJF = normsJF[idxJF]
                print(errJF)
                betweenJTF = np.array([(-uF)*np.cos(0.5*np.pi*t) + (-uJTF)*np.sin(0.5*np.pi*t) for t in np.linspace(0,1,100,endpoint=True)])
                normsJTF = np.linalg.norm(betweenJTF+x,axis=1)
                idxJTF = np.argmin(normsJTF)
                errJTF = normsJF[idxJTF]
                print(errJTF)

                # Construct figure
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(azim=90.,elev=90.)
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                X = 1 * np.outer(np.cos(u), np.sin(v)) + offset[0]
                Y = 1 * np.outer(np.sin(u), np.sin(v)) + offset[1]
                Z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + offset[2]
                # ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,
                #                   linewidth=0,antialiased=False)
                ax.plot_wireframe(X,Y,Z,rstride=10,cstride=10,color='k',alpha=0.4)
                ax.set_xlim3d(np.min(X),np.max(X))
                ax.set_ylim3d(np.min(Y),np.max(Y))
                ax.set_zlim3d(np.min(Z),np.max(Z))
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')

                s=5
                Fxs = J[0,0]*X + J[0,1]*Y + J[0,2]*Z + Domain.b[0]
                Fys = J[1,0]*X + J[1,1]*Y + J[1,2]*Z + Domain.b[1]
                Fzs = J[2,0]*X + J[2,1]*Y + J[2,2]*Z + Domain.b[2]
                ax.quiver(X[::s,::s], Y[::s,::s], Z[::s,::s], -Fxs[::s,::s], -Fys[::s,::s], -Fzs[::s,::s], length=0.1, normalize=True)

                ax.plot([offset[0]],[offset[1]],[offset[2]],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
                ax.plot([x[0]],[x[1]],[x[2]],'ko',label='x')
                Farrow = np.vstack((x[None],x[None]-uF[None]))
                ax.plot(Farrow[:,0],Farrow[:,1],Farrow[:,2],'->',label='-F')
                JFFarrow = np.vstack((x[None],x[None]+uJF[None]))
                ax.plot(JFFarrow[:,0],JFFarrow[:,1],JFFarrow[:,2],'->',label='JF')
                JTFFarrow = np.vstack((x[None],x[None]-uJTF[None]))
                ax.plot(JTFFarrow[:,0],JTFFarrow[:,1],JTFFarrow[:,2],'-.>',label='-JTF')
                vecarrow = np.vstack((x[None],x[None]+uvec[None]))
                ax.plot(vecarrow[:,0],vecarrow[:,1],vecarrow[:,2],'-->',label='vec')
                newarrow = np.vstack((x[None],x[None]+unew[None]))
                ax.plot(newarrow[:,0],newarrow[:,1],newarrow[:,2],'-.>',label='new')
                curlarrow = np.vstack((x[None],x[None]+ucurl[None]))
                ax.plot(curlarrow[:,0],curlarrow[:,1],curlarrow[:,2],'-->',label='curl')
                # ax.arrow(x[0],x[1],x[2],Farrow[1][0],Farrow[1][1],Farrow[1][2],label='-F')
                # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
                # ax.set_zlabel('f(X,Y)',rotation=90)
                # ax.text2D(0.05, 0.95, 'Steepest Descent on the\nRosenbrock Function',
                #           transform=ax.transAxes)
                # lw = cycle(range(2*len(steps),0,-2))
                # plt.ion()
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
                plt.show()

                # if a==0.1 and b==1.0 and c==0.1:
                #     embed()
                #     assert False

    # data_Euler = ListONP2NP(Euler_Results.PermStorage['Data'])
    # data_EG = ListONP2NP(EG_Results.PermStorage['Data'])
    

    # X, Y = np.meshgrid(np.arange(-2.5, 2.5, .2), np.arange(-2.5, 2.5, .2))
    # U = np.zeros_like(X)
    # V = np.zeros_like(Y)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         vec = -Domain.F([X[i,j],Y[i,j]])
    #         U[i,j] = vec[0]
    #         V[i,j] = vec[1]

    # # plt.figure()
    # # plt.title('Arrows scale with plot width, not view')
    # # Q = plt.quiver(X, Y, U, V, units='width')
    # # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    # #                    coordinates='figure')

    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    # plt.title("Extragradient vs Simultaneous Gradient Descent")
    # Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
    #                pivot='mid', units='inches')
    # vec_Euler = -Domain.F(data_Euler[-1,:])
    # vec_EG = -Domain.F(data_EG[-1,:])
    # print(data_Euler[-1])
    # print(vec_Euler)
    # plt.quiver(data_Euler[-1][0], data_Euler[-1][1], vec_Euler[0], vec_Euler[1], pivot='mid', units='inches', color='b',
    #         headwidth=5)
    # plt.quiver(data_EG[-1][0], data_EG[-1][1], vec_EG[0], vec_EG[1], pivot='mid', units='inches', color='r',
    #         headwidth=5)
    # plt.scatter(X[::3, ::3], Y[::3, ::3], color='gray', s=5)
    # plt.plot(data_Euler[:,0],data_Euler[:,1],'b',linewidth=5,label='Simultaneous\nGradient Descent')
    # plt.plot(data_EG[:,0],data_EG[:,1],'r',linewidth=5,label='Extragradient')
    # plt.plot([0],[0],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
    # plt.axis([-2.5,2.5,-2.5,2.5])
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
    # # plt.ion()
    # # plt.show()
    # plt.savefig('EGvsEuler.png')


if __name__ == '__main__':
    Demo()
