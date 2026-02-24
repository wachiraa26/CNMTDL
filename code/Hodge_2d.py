import scipy.sparse as ss
import numpy as np
from scipy.sparse import diags,csr_matrix
import time
from scipy.sparse.linalg import spsolve


epsilon = 1e-5
big_value = 9999999




def get_D0(M,N):
    # grid shape M*N
    nV = M*N
    nE = M*(N-1) + N*(M-1)
    row,col,data = [],[],[]
    for i in range(nE):
        # horizontal
        if i<M*(N-1):
            left = int( i/(N-1) ) * N + i%(N-1)
            right = left + 1
            
            row.append(i)
            col.append(left)
            data.append(-1)
            
            row.append(i)
            col.append(right)
            data.append(1)
        # vertical
        else:
            temp = i - M*(N-1)
            left = temp
            right = left + N
            
            row.append(i)
            col.append(left)
            data.append(-1)
            
            row.append(i)
            col.append(right)
            data.append(1)
    D = csr_matrix((data,(row,col)),shape=(nE,nV))
    return D


def get_D1(M,N):
    nE = M*(N-1) + N*(M-1)
    nF = (M-1)*(N-1)
    row,col,data = [],[],[]
    for i in range(nF):
        row2 = int(i/(N-1))
        col2 = i%(N-1)
        
        index1 = i
        index2 = index1 + N-1
        index3 = M*(N-1) + row2*N + col2
        index4 = index3 + 1
        
        row.append(i)
        col.append(index1)
        data.append(1)
        
        row.append(i)
        col.append(index2)
        data.append(-1)
        
        row.append(i)
        col.append(index3)
        data.append(-1)
        
        row.append(i)
        col.append(index4)
        data.append(1)
    
    D = csr_matrix((data,(row,col)),shape=(nF,nE))
    return D
    
    
def out_of_grid(x,y,M,N):
    if x<0 or x>=M or y<0 or y>=N:
        return 1
    return 0

def get_point_value(M,N,i,j,manifold):
    # in the grid vertex, in the center of a cube, out of the grid,
    if abs(i%1)<epsilon and abs(j%1)<epsilon:
        return manifold[round(i),round(j)]
    elif abs(i%0.5)<epsilon and abs(j%0.5)<epsilon:
        
        x0,y0 = int(i-0.5),int(j-0.5)
        x1,y1 = int(i+0.5),int(j-0.5)
        x2,y2 = int(i-0.5),int(j+0.5)
        x3,y3 = int(i+0.5),int(j+0.5)
        
        
        if out_of_grid(x0,y0,M,N)==1 or out_of_grid(x1,y1,M,N)==1 or \
        out_of_grid(x2, y2, M,N)==1 or out_of_grid(x3, y3, M,N)==1:
            return big_value
        else:
            #return (manifold[x0,y0] + manifold[x1,y1] + 
            #        manifold[x2,y2] + manifold[x3,y3])/4
            return min(manifold[x0,y0],manifold[x1,y1],manifold[x2,y2],manifold[x3,y3])
    else:
        print('error')
        return

def get_P0_normal(M,N,manifold,cutoff=0):
    nV = M*N
    row,column,data = [],[],[]
    n = 0
    for i in range(nV):
        x = i % N
        y = int(i/N)
        # dual unchanged
        if manifold[y,x]<=cutoff:
            #P0[n,i] = 1
            row.append(n)
            column.append(i)
            data.append(1)
            n = n + 1
    P0n = csr_matrix((data, (row, column)), shape=(n, nV))
    return P0n
    

def get_P1_normal(M,N,manifold,cutoff=0):
    nE = M*(N-1) + N*(M-1)
    row,column,data = [],[],[]
    n = 0
    for i in range(nE):
        if i<M*(N-1):
            x = i % (N-1)
            y = int(i/(N-1))
            direction = 0
        else:
            temp = i - M*(N-1)
            x = temp % N
            y = int(temp/N)
            direction = 1
        # dual unchanged
        if direction==0:
            f0 = get_point_value(M,N,y,x,manifold)
            f1 = get_point_value(M,N,y,x+1,manifold)
        else:
            f0 = get_point_value(M,N,y,x,manifold)
            f1 = get_point_value(M,N,y+1,x,manifold)
        if f0<=cutoff or f1<=cutoff:
            #S1[i,i] = 1/max(get_edge_length(f0,f1,cutoff),epsilon)
            #P1[n,i] = 1
            row.append(n)
            column.append(i)
            data.append(1)
            n = n + 1
                
    P1n = csr_matrix((data, (row, column)), shape=(n, nE))
    return P1n
    

def get_P1_tangential(M,N,manifold,cutoff=0):
    nE = M*(N-1) + N*(M-1)
    row,column,data = [],[],[]
    n = 0
    for i in range(nE):
        if i<M*(N-1):
            x = i % (N-1)
            y = int(i/(N-1))
            direction = 0
        else:
            temp = i - M*(N-1)
            x = temp % N
            y = int(temp/N)
            direction = 1
        # primal unchanged
        if direction==0:
            f0 = get_point_value(M,N,y-0.5,x+0.5,manifold)
            f1 = get_point_value(M,N,y+0.5,x+0.5,manifold)
        else:
            f0 = get_point_value(M,N,y+0.5,x-0.5,manifold)
            f1 = get_point_value(M,N,y+0.5,x+0.5,manifold)
            
        if f0<=cutoff or f1<=cutoff:
            #S1[i,i] = get_edge_length(f0,f1,cutoff)
            #P1[n,i] = 1
            row.append(n)
            column.append(i)
            data.append(1)
            n = n + 1
                
    P1t = csr_matrix((data, (row, column)), shape=(n, nE))
    return P1t

def get_P2_tangential(M,N,manifold,cutoff=0):
    nF = (M-1)*(N-1)
    row,column,data = [],[],[]
    n = 0
    for i in range(nF):
        x = i % (N-1)
        y = int(i/(N-1))
        # primal unchanged
        f0 = get_point_value(M,N,y+0.5,x+0.5,manifold)
            
        if f0<=cutoff:
            #S2[i,i] = 1
            #P2[n,i] = 1
            row.append(n)
            column.append(i)
            data.append(1)
            #P2[n,i] = 1
            n = n + 1
    P2t = csr_matrix((data, (row, column)), shape=(n, nF))
    return P2t
    
    
def one_form_to_two_form(vector,M,N):
    H = vector[0:M*(N-1)].reshape(M,N-1)
    V = vector[M*(N-1)::].reshape(M-1,N)
    X = (H[:-1, :] + H[1:, :]) / 2
    Y = (V[:, :-1] + V[:, 1:]) / 2
    res = np.array([X,Y])
    return res


def get_BIG_decomposition(omega,D0,D1,M,N,manifold,cutoff):
    P1n = get_P1_normal(M,N,manifold,cutoff)
    P0n = get_P0_normal(M,N,manifold,cutoff)
    P2t = get_P2_tangential(M,N,manifold,cutoff)
    P1t = get_P1_tangential(M,N,manifold,cutoff)
    
    
    D0n_int = P1n @ D0 @ P0n.transpose()
    D1t_int = P2t @ D1 @ P1t.transpose()
    
    # laplacian
    L0n = D0n_int.transpose() @ D0n_int
    L2t = D1t_int @ D1t_int.transpose()
    

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
    
    omega_all = one_form_to_two_form(omega_all,M,N)
    cur_all = one_form_to_two_form(cur_all,M,N)
    div_all = one_form_to_two_form(div_all,M,N)
    har = one_form_to_two_form(har,M,N)
    
    return np.round(omega_all,8), np.round(cur_all,8), np.round(div_all,8), np.round(har,8)