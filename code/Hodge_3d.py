
import scipy.sparse as ss
import numpy as np
from scipy.sparse import diags,csr_matrix
from scipy.sparse.linalg import spsolve
import time



epsilon = 1e-5
big_value = 9999999



def edge_number_to_vertex_index(n,N):
    # x-axis
    if n<(N-1)*N*N:
        layer = int( n/( (N-1)*N) )
        temp1 = n % ((N-1)*N)
        i = int( temp1/(N-1) ) * N + temp1 % (N-1) + layer * N*N
        return i,i+1
    # y-axis
    elif n>=(N-1)*N*N and n<2*(N-1)*N*N:
        n = n - (N-1)*N*N
        layer = int( n/( (N-1)*N ) )
        temp1 = n % ((N-1)*N)
        i = temp1 + layer * N*N
        return i,i+N
    # z-axis
    else:
        n = n - 2*(N-1)*N*N
        layer = int( n/(N*N) )
        temp1 = n % (N*N)
        i = temp1 + layer * N*N
        return i,i+N*N
    

def get_D0(N):
    nV = N*N*N
    nE = 3*(N-1)*N*N
    row,col,data = [],[],[]
    for i in range(nE):
        left,right = edge_number_to_vertex_index(i,N)
        #print(i,left,right)
        #D[i,left] = -1
        #D[i,right] = 1
        
        row.append(i)
        col.append(left)
        data.append(-1)
        
        row.append(i)
        col.append(right)
        data.append(1)
        
    D = csr_matrix((data,(row,col)),shape=(nE,nV))
    return D



def face_number_to_edge_index(n,N):
    # x-y palne
    if n<(N-1)*(N-1)*N:
        layer = int( n/((N-1)*(N-1)) )
        temp1 = n % ((N-1)*(N-1))
        e01 = temp1 + layer * (N-1)*N
        e23 = e01 + N-1
        
        layer2 = int( temp1/(N-1) )
        e02 = (N-1)*N*N + layer * (N-1)*N + layer2 * N + temp1 % (N-1)
        e13 = e02 + 1
        return e01,e23,e02,e13
        
    # x-z plane
    elif n>=(N-1)*(N-1)*N and n<2*(N-1)*(N-1)*N:
        n = n - (N-1)*(N-1)*N
        layer = int( n/((N-1)*N) )
        temp1 = n % ((N-1)*N)
        e01 = temp1 + layer * (N-1)*N
        e23 = e01 + (N-1)*N
        
        layer2 = int( temp1/(N-1) )
        e02 = (N-1)*N*N*2 + layer * N*N + layer2*N + temp1 % (N-1)
        e13 = e02 + 1
        return e01,e23,e02,e13
    # y-z plane
    else:
        n = n - (N-1)*(N-1)*N*2
        layer = int( n/((N-1)*N) )
        temp1 = n % ((N-1)*N)
        e02 = temp1 + layer * N*N + (N-1)*N*N*2
        e13 = e02 + N
        
        layer2 = int( temp1/N )
        e01 = N*(N-1)*N + layer * N*(N-1) + layer2 * N + temp1 % N
        e23 = e01 + (N-1)*N
        return e01,e23,e02,e13
        

def get_D1(N):
    nE = 3*(N-1)*N*N
    nF = 3*(N-1)*(N-1)*N
    row,col,data = [],[],[]
    for i in range(nF):
        e01,e23,e02,e13 = face_number_to_edge_index(i,N)
        
        row.append(i)
        col.append(e01)
        data.append(1)
        
        row.append(i)
        col.append(e23)
        data.append(-1)
        
        row.append(i)
        col.append(e02)
        data.append(-1)
        
        row.append(i)
        col.append(e13)
        data.append(1)
    D = csr_matrix((data,(row,col)),shape=(nF,nE))
    return D


def cube_number_to_face_index(n,N):
    layer = int( n/( (N-1)*(N-1) ) )
    temp1 = n % ( (N-1)*(N-1) )
    xy0 = layer * (N-1)*(N-1) + temp1
    xy1 = xy0 + (N-1)*(N-1)
    
    xz0 = (N-1)*(N-1)*N + layer * (N-1)*N + temp1
    xz1 = xz0 + (N-1)
    
    layer2 = int( temp1/(N-1) )
    temp2 = temp1 % (N-1)
    
    yz0 = (N-1)*(N-1)*N*2 + layer * (N-1)*N + layer2 * N + temp2
    yz1 = yz0 + 1
    return xy0,xy1,xz0,xz1,yz0,yz1
    
def get_D2(N):
    nF = 3*(N-1)*(N-1)*N
    nT = (N-1)*(N-1)*(N-1)
    D = ss.lil_matrix((nT, nF))
    row,col,data = [],[],[]
    for i in range(nT):
        xy0,xy1,xz0,xz1,yz0,yz1 = cube_number_to_face_index(i,N)
        #print(xy0,xy1,xz0,xz1,yz0,yz1)
        
        row.append(i)
        col.append(xy0)
        data.append(-1)
        
        row.append(i)
        col.append(xy1)
        data.append(1)
        
        row.append(i)
        col.append(xz0)
        data.append(1)
        
        row.append(i)
        col.append(xz1)
        data.append(-1)
        
        row.append(i)
        col.append(yz0)
        data.append(-1)
        
        row.append(i)
        col.append(yz1)
        data.append(1)
        
    D = csr_matrix((data,(row,col)),shape=(nT,nF))
    return D


def out_of_grid(x,y,z,N):
    if x<0 or x>=N or y<0 or y>=N or z<0 or z>=N:
        return 1
    return 0

def get_point_value(N,i,j,k,manifold):
    # in the grid vertex, in the cube center, out of the grid
    if abs(i%1)<epsilon and abs(j%1)<epsilon and abs(k%1)<epsilon:
        return manifold[round(i),round(j),round(k)]
    elif abs(i%0.5)<epsilon and abs(j%0.5)<epsilon and abs(k%0.5)<epsilon:
        x0,y0,z0 = int(i-0.5),int(j-0.5),int(k-0.5)
        x1,y1,z1 = int(i-0.5),int(j+0.5),int(k-0.5)
        x2,y2,z2 = int(i+0.5),int(j-0.5),int(k-0.5)
        x3,y3,z3 = int(i+0.5),int(j+0.5),int(k-0.5)
        x4,y4,z4 = int(i-0.5),int(j-0.5),int(k+0.5)
        x5,y5,z5 = int(i-0.5),int(j+0.5),int(k+0.5)
        x6,y6,z6 = int(i+0.5),int(j-0.5),int(k+0.5)
        x7,y7,z7 = int(i+0.5),int(j+0.5),int(k+0.5)
        if out_of_grid(x0,y0,z0,N)==1 or out_of_grid(x1,y1,z1,N)==1 or out_of_grid(x2,y2,z2,N)==1 or \
           out_of_grid(x3,y3,z3,N)==1 or out_of_grid(x4,y4,z4,N)==1 or out_of_grid(x5,y5,z5,N)==1 or \
           out_of_grid(x6,y6,z6,N)==1 or out_of_grid(x7,y7,z7,N)==1:
            return big_value   
        else:
            return min( manifold[x0,y0,z0],manifold[x1,y1,z1],manifold[x2,y2,z2],
                     manifold[x3,y3,z3],manifold[x4,y4,z4],manifold[x5,y5,z5],
                     manifold[x6,y6,z6],manifold[x7,y7,z7])
    else:
        print('error')
        return



def edge_number_to_index(e,N):
    if e<(N-1)*N*N:
        direction = 0
        z = int( e/((N-1)*N) )
        temp1 = e % ((N-1)*N)
        y = int(temp1/(N-1))
        x = temp1 % (N-1)
    elif e>=(N-1)*N*N and e<2*(N-1)*N*N:
        direction = 1
        e = e - (N-1)*N*N
        z = int( e/((N-1)*N) )
        temp1 = e % ((N-1)*N)
        y = int(temp1/N)
        x = temp1 % N
    else:
        direction = 2
        e = e - 2*(N-1)*N*N
        z = int( e/(N*N) )
        temp1 = e % (N*N)
        y = int(temp1/N)
        x = temp1 % N
    return x,y,z,direction


def get_P1_normal(N,manifold,cutoff=0):
    nE = 3*(N-1)*N*N
    row,column,data = [],[],[]
    n = 0
    for i in range(nE):
        x,y,z,direction = edge_number_to_index(i,N)
        # dual unchanged
        if direction==0:
            f0 = get_point_value(N,z,y,x,manifold)
            f1 = get_point_value(N,z,y,x+1,manifold)
        elif direction==1:
            f0 = get_point_value(N,z,y,x,manifold)
            f1 = get_point_value(N,z,y+1,x,manifold)
        elif direction==2:
            f0 = get_point_value(N,z,y,x,manifold)
            f1 = get_point_value(N,z+1,y,x,manifold)
        if f0<=cutoff or f1<=cutoff:
            row.append(n)
            column.append(i)
            data.append(1)
            n = n + 1
                
    P1n = csr_matrix((data, (row, column)), shape=(n, nE))
    return P1n



def vertex_number_to_index(n,N):
    z = int(n/( N*N ))
    temp1 = n % (N*N)
    y = int(temp1/N)
    x = temp1 % N
    return x,y,z


def get_P0_normal(N,manifold,cutoff=0):
    nV = N*N*N
    row,column,data = [],[],[]
    n = 0
    for i in range(nV):
        x,y,z = vertex_number_to_index(i,N)
        if manifold[z,y,x]<=cutoff:
            row.append(n)
            column.append(i)
            data.append(1)
            n = n + 1
                
    P0n = csr_matrix((data, (row, column)), shape=(n, nV))
    return P0n


def face_number_to_index(f,N):
    if f<(N-1)*(N-1)*N:
        direction = 0
        z = int( f/((N-1)*(N-1)) )
        temp1 = f % ((N-1)*(N-1))
        y = int( temp1/(N-1) )
        x = temp1 % (N-1)
    elif f>=(N-1)*(N-1)*N and f<2*(N-1)*(N-1)*N:
        direction = 1
        f = f - (N-1)*(N-1)*N
        z = int( f/((N-1)*N) )
        temp1 = f % ((N-1)*N)
        y = int( temp1/(N-1) )
        x = temp1 % (N-1)
    else:
        direction = 2
        f = f - 2*(N-1)*(N-1)*N
        z = int( f/((N-1)*N) )
        temp1 = f % ((N-1)*N)
        y= int( temp1/N )
        x = temp1 % N
    return x,y,z,direction


def get_P2_tangential(N,manifold,boundary,cutoff=0):
    nF = 3*(N-1)*(N-1)*N
    row,column,data = [],[],[]
    n = 0
    for f in range(nF):
        x,y,z,direction = face_number_to_index(f,N)
        #print(i,j,k,direction)
        if direction==0:
            f0 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
            f1 = get_point_value(N,z-0.5,y+0.5,x+0.5,manifold)
        elif direction==1:
            f0 = get_point_value(N,z+0.5,y-0.5,x+0.5,manifold)
            f1 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
        elif direction==2:
            f0 = get_point_value(N,z+0.5,y+0.5,x-0.5,manifold)
            f1 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
        if f0<=cutoff or f1<=cutoff:
            row.append(n)
            column.append(f)
            data.append(1)
            n = n + 1
            
    P2t = csr_matrix((data, (row, column)), shape=(n, nF))
    return P2t
    


def get_P1_tangential(N,manifold,boundary,cutoff=0):
    nE = 3*(N-1)*N*N
    row,column,data = [],[],[]
    n = 0
    for e in range(nE):
        x,y,z,direction = edge_number_to_index(e,N)
        if direction==0:
            f0 = get_point_value(N,z-0.5,y-0.5,x+0.5,manifold)
            f1 = get_point_value(N,z+0.5,y-0.5,x+0.5,manifold)
            f2 = get_point_value(N,z-0.5,y+0.5,x+0.5,manifold)
            f3 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
        elif direction==1:
            f0 = get_point_value(N,z-0.5,y+0.5,x-0.5,manifold)
            f1 = get_point_value(N,z-0.5,y+0.5,x+0.5,manifold)
            f2 = get_point_value(N,z+0.5,y+0.5,x-0.5,manifold)
            f3 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
        elif direction==2:
            f0 = get_point_value(N,z+0.5,y-0.5,x-0.5,manifold)
            f1 = get_point_value(N,z+0.5,y-0.5,x+0.5,manifold)
            f2 = get_point_value(N,z+0.5,y+0.5,x-0.5,manifold)
            f3 = get_point_value(N,z+0.5,y+0.5,x+0.5,manifold)
        if f0<=cutoff or f1<=cutoff or f2<=cutoff or f3<=cutoff:
            row.append(n)
            column.append(e)
            data.append(1)
            n = n + 1
    
    P1t = csr_matrix((data, (row, column)), shape=(n, nE))
    return P1t

def cube_number_to_index(t,N):
    z = int( t/((N-1)*(N-1)) )
    temp1 = t % ((N-1)*(N-1))
    y = int( temp1/(N-1) )
    x = temp1 % (N-1)
    return x,y,z


def get_P3_tangential(N,manifold,boundary,cutoff=0):
    nT = (N-1)*(N-1)*(N-1)
    row,column,data = [],[],[]
    n = 0
    for t in range(nT):
        x,y,z = cube_number_to_index(t,N)
        f0 = get_point_value(N, z+0.5, y+0.5, x+0.5, manifold)
        if f0<=cutoff:
            row.append(n)
            column.append(t)
            data.append(1)
            n = n + 1
    P3t = csr_matrix((data, (row, column)), shape=(n, nT))
    return P3t


            



    
def one_form_to_two_form(vector,N):
    H = vector[0:N*N*(N-1)].reshape(N,N,N-1)
    V = vector[N*N*(N-1):2*N*N*(N-1)].reshape(N,N-1,N)
    W = vector[2*N*N*(N-1)::].reshape(N-1,N,N)
    
    X1 = H[:-1, :, :] + H[1:, :, :]
    X2 = X1[:, 1:, :] + X1[:, :-1, :]
    X = X2/4 
    
    Y1 = V[:-1, :, :] + V[1:, :, :]
    Y2 = Y1[:, :, 1:] + Y1[:, :, :-1]
    Y = Y2/4
    
    Z1 = W[:, 1:, :] + W[:, :-1, :]
    Z2 = Z1[:, :, 1:] + Z1[:, :, :-1]
    Z = Z2/4
    
    res = np.array([X,Y,Z])
    return res





def get_BIG_decomposition(omega,D0,D1,D2,N,manifold,cutoff):
    P1n = get_P1_normal(N,manifold,cutoff)
    P0n = get_P0_normal(N,manifold,cutoff)
    P3t = get_P3_tangential(N,manifold,cutoff)
    P2t = get_P2_tangential(N,manifold,cutoff)
    P1t = get_P1_tangential(N,manifold,cutoff)
    
    
    D0n_int = P1n @ D0 @ P0n.transpose()
    D1t_int = P2t @ D1 @ P1t.transpose()
    D2t_int = P3t @ D2 @ P2t.transpose()
    
    # laplacian
    L0n = D0n_int.transpose() @ D0n_int
    L2t = D1t_int @ D1t_int.transpose() + D2t_int.transpose() @ D2t_int
    L2t = L2t + diags([1e-5],[0],shape=L2t.shape,format='csr')
    #print(omega.shape)

    # project omega to manifold
    omega_n = P1n @ omega
    omega_t = P1t @ omega
    
    # L0nX = D0^T*S1*omega
    cur = spsolve(L0n,D0n_int.transpose() @ omega_n)
    cur2 = D0n_int @ cur
    
    # L2tX = D1t*omega
    div = spsolve(L2t,D1t_int @ omega_t)
    div2 = D1t_int.transpose() @ div
    
    # projected back to the whole grid
    omega_all = P1t.transpose() @ omega_t
    cur_all = ( P1n.transpose() @ cur2 ).reshape(-1,1)
    div_all = ( P1t.transpose() @ div2 ).reshape(-1,1)
    
    # harmonic : w = d*alpha_n + delta*beta_t + eta
    har = omega_all - cur_all - div_all
    
    omega_all = one_form_to_two_form(omega_all,N)
    cur_all = one_form_to_two_form(cur_all,N)
    div_all = one_form_to_two_form(div_all,N)
    har = one_form_to_two_form(har,N)
    
    
    
    #print(omega_all.shape,cur_all.shape,div_all.shape,har.shape)
    return np.round(omega_all,8), np.round(cur_all,8), np.round(div_all,8), np.round(har,8)
    
    

    

