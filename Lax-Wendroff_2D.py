import numpy as np
from matplotlib import pyplot as plt

def LaxWendroff2D(
			lmb, tau, Ic, Id, b, Lx, Ly, T,
			Nx, Ny, dt,mu,rand=True,plot='ON',plotN='OFF'):
   
    
    x = np.linspace(0, Lx, Nx+1)       # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)       # mesh points in y dir
    dx = x[1] - x[0]                   # distance of mesh points in x dir 
    dy = y[1] - y[0]                   # distance of mesh points in y dir 

    dt = float(dt)                     # time step
    Nt = int(round(T/float(dt)))       #number of mesh points in time
    t = np.linspace(0, Nt*dt, Nt+1)    # mesh points in time

    
    #plot a graph 3 times, i.e. 1st plot at t=Nim, 
    #2nd plot at t=2*Nim, 3rd plot at t=3*Nim
    Nim= int(Nt/3)
    im=0 #index for subplot
       
    n1_     = np.zeros((Nx+1,Ny+1)) # n1 at the previous time level
    n1      = np.zeros((Nx+1,Ny+1)) # unknown n1 at new time level
    phi1_   = np.zeros((Nx+1,Ny+1)) # phi1 at the previous time level
    phi1    = np.zeros((Nx+1,Ny+1)) # unknown phi1 at new time level
    psi1_   = np.zeros((Nx+1,Ny+1)) # psi1 at the previous time level
    psi1    = np.zeros((Nx+1,Ny+1)) # unknown psi1 at new time level
    n2_     = np.zeros((Nx+1,Ny+1)) # n2 at the previous time level
    n2      = np.zeros((Nx+1,Ny+1)) # unknown n2 at new time level
    phi2_   = np.zeros((Nx+1,Ny+1)) # phi2 at the previous time level
    phi2    = np.zeros((Nx+1,Ny+1)) # unknown phi2 at new time level
    psi2_   = np.zeros((Nx+1,Ny+1)) # psi2 at the previous time level
    psi2    = np.zeros((Nx+1,Ny+1)) # unknown psi2 at new time level

    #payoff matrix
    P=np.array([[1,0],[b,0]])
    
    #artificial viscosity
    Fx=mu
    Fy=mu

    #Load initial conditions
    #random initial conditions
    if rand==True:
        n_r=np.random.rand(Nx+1, Ny+1)
        n1_ = n_r
        n2_ = 1-n_r
        
    #other initial conditions    
    else:
        for i in range(0,Nx+1):
            for j in range(0,Ny+1):
                n1_[i,j] = Ic(i,j)
                n2_[i,j] = Id(i,j)

    #set size and features of the plot of the total population
    if plotN=='ON':
        fig1 = plt.figure(figsize=(5,5))           
        ax1 = fig1.add_subplot(1,1,1)
        ax1.set_title('Number of total players (N)')
        ax1.set_xlabel('time (n*dt)')
        ax1.set_ylabel('N')
    
    #plot the initial conditions
    if plot=='ON':
        fig = plt.figure(figsize=(8,14))
        im+=1
        ax = fig.add_subplot(4,2,im)
        ax.set_aspect('equal')
        cc1=plt.contourf(x, y, n1_,cmap="summer",vmin=0, vmax=1)
        fig.colorbar(cc1)
        ax.set_title('coop   t=%1.3f' %(0))

        im+=1
        ax = fig.add_subplot(4,2,im)
        ax.set_aspect('equal')
        cc2=plt.contourf(x, y, n2_,cmap="cool",vmin=0, vmax=1)
        fig.colorbar(cc2)
        ax.set_title('defect   t=%1.3f' %(0))
    
    # Evolution start        
    for n in range(1,Nt+1):
        for j in range(1,Ny):
            for i in range(1,Nx):
                n1[i,j]  =(n1_[i,j]
                           -dt/(2*dx)*(phi1_[i+1,j]-phi1_[i-1,j])
                           -dt/(2*dy)*(psi1_[i,j+1]-psi1_[i,j-1])
                           +dt**2/(2*dx**2)*(phi1_[i+1,j]
						   -2*phi1_[i,j]+phi1_[i-1,j])
                           +dt**2/(2*dy**2)*(psi1_[i,j+1]
						   -2*psi1_[i,j]+psi1_[i,j-1])
                           +Fx*(n1_[i-1,j] - 2*n1_[i,j] + n1_[i+1,j])
                           +Fy*(n1_[i,j-1] - 2*n1_[i,j] + n1_[i,j+1])
                           -dt*((n1_[i,j]*n2_[i,j])/(n1_[i,j]+n2_[i,j])**2)*
                           ((P[1,0]-P[0,0])*n1_[i,j]
						   +(P[1,1]-P[0,1])*n2_[i,j]))
                
                phi1[i,j]=(phi1_[i,j]
                           -(dt*lmb**2)/(2*dx*tau)*(n1_[i+1,j]-n1_[i-1,j])
                           +((dt*lmb**2)/(2*dx*tau))**2
						   *(n1_[i+1,j]-2*n1_[i,j]+n1_[i-1,j])
                           +Fx*(phi1_[i-1,j] - 2*phi1_[i,j] + phi1_[i+1,j])
                           -dt*phi1_[i,j]/tau)
                
                psi1[i,j]=(psi1_[i,j]
                           -(dt*lmb**2)/(2*dy*tau)*(n1_[i,j+1]-n1_[i,j-1])
                           +((dt*lmb**2)/(2*dy*tau))**2
						   *(n1_[i,j+1]-2*n1_[i,j]+n1_[i,j-1])
                           +Fy*(psi1_[i,j-1] - 2*psi1_[i,j] + psi1_[i,j+1])
                           -dt*psi1_[i,j]/tau)

                n2[i,j]  =(n2_[i,j]
                           -dt/(2*dx)*(phi2_[i+1,j]-phi2_[i-1,j])
                           -dt/(2*dy)*(psi2_[i,j+1]-psi2_[i,j-1])
                           +dt**2/(2*dx**2)*(phi2_[i+1,j]
						   -2*phi2_[i,j]+phi2_[i-1,j])
                           +dt**2/(2*dy**2)*(psi2_[i,j+1]
						   -2*psi2_[i,j]+psi2_[i,j-1])
                           +Fx*(n2_[i-1,j] - 2*n2_[i,j] + n2_[i+1,j])
                           +Fy*(n2_[i,j-1] - 2*n2_[i,j] + n2_[i,j+1])
                           +dt*((n1_[i,j]*n2_[i,j])/(n1_[i,j]+n2_[i,j])**2)*
                           ((P[1,0]-P[0,0])*n1_[i,j]
						   +(P[1,1]-P[0,1])*n2_[i,j]))

                phi2[i,j]=(phi2_[i,j]
                           -(dt*lmb**2)/(2*dx*tau)*(n2_[i+1,j]-n2_[i-1,j])
                           +((dt*lmb**2)/(2*dx*tau))**2
						   *(n2_[i+1,j]-2*n2_[i,j]+n2_[i-1,j])
                           +Fx*(phi2_[i-1,j] - 2*phi2_[i,j] + phi2_[i+1,j])
                           -dt*phi2_[i,j]/tau)
                
                psi2[i,j]=(psi2_[i,j]
                           -(dt*lmb**2)/(2*dy*tau)*(n2_[i+1,j]-n2_[i-1,j])
                           +((dt*lmb**2)/(2*dy*tau))**2
						   *(n2_[i+1,j]-2*n2_[i,j]+n2_[i-1,j])
                           +Fy*(psi2_[i-1,j] - 2*psi2_[i,j] + psi2_[i+1,j])
                           -dt*psi2_[i,j]/tau)
        
        # Insert boundary conditions
        #upper bound
        j = Ny
        for i in range(0,Nx+1):
            n1[i,j]  = 0
            n2[i,j]  = 0
            phi1[i,j] = 0
            phi2[i,j] = 0
            psi1[i,j] = 0
            psi2[i,j] = 0
        
        #left bound
        i = 0 
        for j in range(0,Ny+1):
            n1[i,j]  = 0
            n2[i,j]  = 0
            phi1[i,j] = 0
            phi2[i,j] = 0
            psi1[i,j] = 0
            psi2[i,j] = 0
        #right bound
        i = Nx
        for j in range(0,Ny+1):
            n1[i,j]  = 0
            n2[i,j]  = 0
            phi1[i,j] = 0
            phi2[i,j] = 0
            psi1[i,j] = 0
            psi2[i,j] = 0
        #lower bound
        j = 0 
        for i in range(0,Nx+1):
            n1[i,j]  = 0
            n2[i,j]  = 0
            phi1[i,j] = 0
            phi2[i,j] = 0
            psi1[i,j] = 0
            psi2[i,j] = 0
        
        
        #plot n1 and n2
        if plot=='ON':
            if (n+1)%Nim==0:
                im+=1
                ax = fig.add_subplot(4,2,im)
                ax.set_aspect('equal')
                cc1=plt.contourf(x, y, n1,cmap="summer",vmin=0, vmax=1)
                fig.colorbar(cc1)
                ax.set_title('coop   t=%1.3f' %((n+1)*dt))
                im+=1
                ax = fig.add_subplot(4,2,im)
                ax.set_aspect('equal')
                cc2=plt.contourf(x, y, n2,cmap="cool",vmin=0, vmax=1)
                fig.colorbar(cc2)
                ax.set_title('defect   t=%1.3f' %((n+1)*dt))

                
        #plot total population n1+n2
        if plotN=='ON':
            ax1.set(ylim=(0, 400))
            N1=ax1.plot(n*dt,np.sum(n1+n2),'r.')          
        
        # Update before next step
        n1_, n1     = n1, n1_ 
        n2_, n2     = n2, n2_
        phi1_, phi1 = phi1, phi1_
        phi2_, phi2 = phi2, phi2_
        psi1_, psi1 = psi1, psi1_
        psi2_, psi2 = psi2, psi2_
        
    return n1, n2
