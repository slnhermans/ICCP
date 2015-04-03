#If lower, keep it. If higher, keep it with probability P = exp(-beta(Hj-Hi))
def growcluster(x, y, S, Cluster):
    S[x,y] = -S[x,y] #Flip spin at location
    ClusterSpin = S[x,y] #The spin of the cluster
    Cluster[x,y] = 1 #Add spin to cluster
    for [a, b] in [ [(x+1)%n,y], [(x-1)%n,y], [x,(y+1)%n], [x,(y-1)%n] ]:
        if Cluster[a,b] != 1:
            tryadd(a, b, S, Cluster)
    return S, Cluster

def tryadd(a, b, S, Cluster):
    if S[a,b] != ClusterSpin:
        if np.random.choice([0,1],p=[1-P, P]) == 1:
            growcluster(a, b, S, Cluster)
    return S, Cluster

def spin_flip_wolff(S,T,h):
    Cluster = np.zeros((n,n),dtype = int) #Matrix that says for every analog in S if it is in the cluster
    x, y = np.random.randint(0,n,size=2)
    #With a chance P, perimeter spins are added to the cluster
    P = 1 - np.exp(-2.*J/(kb*T))
    growcluster(x, y, S, Cluster)
    return S

#################################################################################
#################################################################################