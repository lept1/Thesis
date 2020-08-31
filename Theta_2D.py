import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

def theta2D(
    Ic, Id, a, b, Lx, Ly, Nx, Ny, dt, T, theta=0.5,
    n_0x=0, n_0y=0, n_Lx=0, n_Ly=0,rand=False,plot='OFF',plotN='OFF'):
    
    x = np.linspace(0, Lx, Nx+1)       # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)       # mesh points in y dir
    dx = x[1] - x[0]                   # distance of mesh points in x dir 
    dy = y[1] - y[0]                   # distance of mesh points in y dir 

    dt = float(dt)   				  # time step
    Nt = int(round(T/float(dt)))       #number of mesh points in time
    t = np.linspace(0, Nt*dt, Nt+1)    # mesh points in time

    #CFL numbers
    Fx = a*dt/dx**2
    Fy = a*dt/dy**2
    
    #plot a graph 3 times, i.e. 1st plot at t=Nim, 
	#2nd plot at t=2*Nim, 3rd plot at t=3*Nim
    Nim= int(Nt/3)
    im=0 #index for subplot
    
    #Payoff matrix
    a11=1.0
    a12=0
    a21=b
    a22=0   
        
    n1  = np.zeros((Nx+1, Ny+1))    # unknown n1 at new time level
    n1_ = np.zeros((Nx+1, Ny+1))    # n1 at the previous time level
    n2  = np.zeros((Nx+1, Ny+1))    # unknown n2 at new time level
    n2_ = np.zeros((Nx+1, Ny+1))    # n2 at the previous time level
    
    #useful vectors to call a mesh point in each direction
    Ix = range(0, Nx+1)
    Iy = range(0, Ny+1)
    It = range(0, Nt+1)

    # Make n_0x, n_0y, n_Lx and n_Ly functions if they are float/int
    if isinstance(n_0x, (float,int)):
        _n_0x = float(n_0x)  # Make copy of n_0x
        n_0x = lambda t: _n_0x
    if isinstance(n_0y, (float,int)):
        _n_0y = float(n_0y)  # Make copy of n_0y
        n_0y = lambda t: _n_0y
    if isinstance(n_Lx, (float,int)):
        _n_Lx = float(n_Lx)  # Make copy of n_Lx
        n_Lx = lambda t: _n_Lx
    if isinstance(n_Ly, (float,int)):
        _n_Ly = float(n_Ly)  # Make copy of n_Ly
        n_Ly = lambda t: _n_Ly

    # Load initial conditions into n1_ and n2_
    #random initial conditions
    if rand==True:
        n_r=np.random.rand(Nx+1, Ny+1)
        n1_ = n_r
        n2_ = 1-n_r
    #other initial conditions
    else:
        for i in Ix:
            for j in Iy:
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
        


    
    # Two-dim coordinate arrays for vectorized function evaluations
    xv = x[:,np.newaxis]
    yv = y[np.newaxis,:]

    
    N = (Nx+1)*(Ny+1)
    #Allocate the matrix A1
    mainc   = np.zeros(N)            # diagonal
    lowerc  = np.zeros(N-1)          # subdiagonal
    upperc  = np.zeros(N-1)          # superdiagonal
    lower2c = np.zeros(N-(Nx+1))     # lower diagonal
    upper2c = np.zeros(N-(Nx+1))     # upper diagonal
    #Allocate c1, i.e. the RHS
    c1      = np.zeros(N)
    
    #Allocate the matrix A2
    maind   = np.zeros(N)            # diagonal
    lowerd  = np.zeros(N-1)          # subdiagonal
    upperd  = np.zeros(N-1)          # superdiagonal
    lower2d = np.zeros(N-(Nx+1))     # lower diagonal
    upper2d = np.zeros(N-(Nx+1))     # upper diagonal
    #Allocate c2, i.e. the RHS
    c2      = np.zeros(N)

    # Precompute sparse matrix
    lower_offsetc = 1
    lower2_offsetc = Nx+1
    
    lower_offsetd = 1
    lower2_offsetd = Nx+1

    #mapping function of the mesh points
    m = lambda i, j: j*(Nx+1) + i
    
    #Build the matrices A1 and A2
    # j=0 boundary line
    j = 0; mainc[m(0,j):m(Nx+1,j)] = 1  
    j = 0; maind[m(0,j):m(Nx+1,j)] = 1
    
    
    for j in Iy[1:-1]:             # Interior mesh lines j=1,...,Ny-1
        i = 0;   mainc[m(i,j)] = 1  # Boundary
        i = Nx;  mainc[m(i,j)] = 1  # Boundary
        
        i = 0;   maind[m(i,j)] = 1  # Boundary
        i = Nx;  maind[m(i,j)] = 1  # Boundary
        
        
        # Interior i points: i=1,...,N_x-1
        lower2c[m(1,j)-lower2_offsetc:m(Nx,j)-lower2_offsetc] = - theta*Fy
        lowerc[m(1,j)-lower_offsetc:m(Nx,j)-lower_offsetc] = - theta*Fx
        mainc[m(1,j):m(Nx,j)] = 1 + 2*theta*(Fx+Fy)
        upperc[m(1,j):m(Nx,j)] = - theta*Fx
        upper2c[m(1,j):m(Nx,j)] = - theta*Fy
        
        lower2d[m(1,j)-lower2_offsetd:m(Nx,j)-lower2_offsetd] = - theta*Fy
        lowerd[m(1,j)-lower_offsetd:m(Nx,j)-lower_offsetd] = - theta*Fx
        maind[m(1,j):m(Nx,j)] = 1 + 2*theta*(Fx+Fy)
        upperd[m(1,j):m(Nx,j)] = - theta*Fx
        upper2d[m(1,j):m(Nx,j)] = - theta*Fy
    
    
    j = Ny; mainc[m(0,j):m(Nx+1,j)] = 1  # Boundary line
    j = Ny; maind[m(0,j):m(Nx+1,j)] = 1  # Boundary line
    
    #Built the matrices as sparse matrices
    A1 = scipy.sparse.diags(
        diagonals=[mainc, lowerc, upperc, lower2c, upper2c],
        offsets=[0, -lower_offsetc, lower_offsetc,
                 -lower2_offsetc, lower2_offsetc],
        shape=(N, N), format='csc')
    
    A2 = scipy.sparse.diags(
        diagonals=[maind, lowerd, upperd, lower2d, upper2d],
        offsets=[0, -lower_offsetc, lower_offsetc,
                 -lower2_offsetd, lower2_offsetd],
        shape=(N, N), format='csc')

    # Evolution start
    for n in It[0:-1]:
        
        # Compute c1 and c2

        j = 0; c1[m(0,j):m(Nx+1,j)] = n_0y(t[n+1])      # Boundary
        j = 0; c2[m(0,j):m(Nx+1,j)] = n_0y(t[n+1])      # Boundary
        for j in Iy[1:-1]:
            i = 0;   p = m(i,j);  c1[p] = n_0x(t[n+1])  # Boundary
            i = Nx;  p = m(i,j);  c1[p] = n_Lx(t[n+1])  # Boundary
            i = 0;   p = m(i,j);  c2[p] = n_0x(t[n+1])  # Boundary
            i = Nx;  p = m(i,j);  c2[p] = n_Lx(t[n+1])  # Boundary
            
            imin = Ix[1]
            imax = Ix[-1]  # for slice, max i index is Ix[-1]-1
            c1[m(imin,j):m(imax,j)] = n1_[imin:imax,j]+ \
                  (1-theta)*(Fx*(n1_[imin+1:imax+1,j]- \
                  2*n1_[imin:imax,j]+n1_[imin-1:imax-1,j])+ \
                  Fy*(n1_[imin:imax,j+1]-2*n1_[imin:imax,j]+ \
                  n1_[imin:imax,j-1]))-dt*((n1_[imin:imax,j]* \
                  n2_[imin:imax,j])/(n1_[imin:imax,j]+ \
                  n2_[imin:imax,j])**2)*((a21-a11)*n1_[imin:imax,j]+ \
                  (a22-a12)*n2_[imin:imax,j])
            
            c2[m(imin,j):m(imax,j)] = n2_[imin:imax,j]+ \
                  (1-theta)*(Fx*(n2_[imin+1:imax+1,j]- \
                  2*n2_[imin:imax,j]+n2_[imin-1:imax-1,j])+ \
                  Fy*(n2_[imin:imax,j+1]-2*n2_[imin:imax,j]+ \
                  n2_[imin:imax,j-1]))+dt*((n1_[imin:imax,j]* \
                  n2_[imin:imax,j])/(n1_[imin:imax,j]+ \
                  n2_[imin:imax,j])**2)*((a21-a11)*n1_[imin:imax,j]+ \
                  (a22-a12)*n2_[imin:imax,j])
            
            
        j = Ny;  c1[m(0,j):m(Nx+1,j)] = n_Ly(t[n+1]) # Boundary
        j = Ny;  c2[m(0,j):m(Nx+1,j)] = n_Ly(t[n+1])

        #Solving the system
        b1 = scipy.sparse.linalg.spsolve(A1, c1)
        b2 = scipy.sparse.linalg.spsolve(A2, c2)
        
        # Fill n1 and n2 with vectors b1 and b2
        #for j in Iy:  # vectorize y lines
        n1[:,:] = b1.reshape(Ny+1,Nx+1).T
        n2[:,:] = b2.reshape(Ny+1,Nx+1).T

        #plot n1 and n2    
        if plot=='ON':
            if (n+1)%Nim==0:
                #fig = plt.figure(figsize=(6,3))
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
            
        # Update n1_ and n2_ before next step
        n1_, n1 = n1, n1_
        n2_, n2 = n2, n2_
    
    return n1, n2
