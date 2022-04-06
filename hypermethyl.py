import numpy as np
import elpigraph
from scipy.spatial import distance
import math
import random
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans,  SpectralClustering
from queue import Queue, PriorityQueue
from sklearn.metrics import pairwise_distances


#Creating MST and Principal Graph
#Input: NxN adjacentry matrix of weights, 0 means no edge
#Output: returns a 1D array of edge-list
def get_edges_from_adj_matrix(adj_matrix):
    list_edges = []
    n_vertex = adj_matrix.shape[0]
    for k1 in range(n_vertex):
        for k2 in range(k1, n_vertex):
            if ( adj_matrix[k1,k2] != 0) or (adj_matrix[k2,k1] != 0):
                list_edges.append( (k1,k2) )
    return np.array(list_edges)


#Creating Tree
def create_tree_by_cluster_knn_mst(X, n_clusters='sqrt', n_neighbors= 10, clustering_method = 'Kmeans'):
    if n_clusters == 'sqrt':
        n_clusters = int(np.sqrt(X.shape[0])) 
    if isinstance(clustering_method ,str) and (clustering_method.lower() == 'Spectral'.lower()): #Spectral Clustering
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(X)
        predicted_clusters = clustering.labels_ # kmeans.predict(X)
        # Get cluster centers by averaging:
        l = len(np.unique(predicted_clusters))
        cluster_centers_ = np.zeros( (l, X.shape[1]))
        for i,v in enumerate(np.unique(predicted_clusters)):
            m = predicted_clusters==v 
            cluster_centers_[i,:] = np.mean(X[m,:],axis = 0 )
    else: # Kmeans clustering by defualt:
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        cluster_centers_ = clustering.cluster_centers_
        predicted_clusters = clustering.labels_ # kmeans.predict(X)
    if n_neighbors > len(cluster_centers_):
        n_neighbors = len(cluster_centers_) # To avoid exception for small number of clusters 
    csr_knn = kneighbors_graph(cluster_centers_, n_neighbors= n_neighbors, mode= 'distance', include_self=True)
    csr_mst = minimum_spanning_tree(csr_knn)

    dict_result = {}
    dict_result['csr_mst'] = csr_mst
    dict_result['csr_knn'] = csr_knn
    dict_result['nodes_positions'] = cluster_centers_
    dict_result['predicted_clusters'] = predicted_clusters
    dict_result['edges_mst'] = get_edges_from_adj_matrix( csr_mst )
    dict_result['edges_knn'] = get_edges_from_adj_matrix( csr_knn )

    return dict_result

def elastic_graph(X, n_clusters = None, a = 0.01):
    if n_clusters == None:
        n_clusters = int(2*math.sqrt(len(X)))
    n_mst_nodes = max(3, 2*n_clusters//3)
    
    dict_result =  create_tree_by_cluster_knn_mst(X, n_clusters= n_mst_nodes , n_neighbors= 10  )
    
    edges =  dict_result['edges_mst']
    nodes_positions = dict_result['nodes_positions']
    
    tree_elpi = elpigraph.computeElasticPrincipalTree(X ,  NumNodes=n_clusters,
                  InitNodePositions = nodes_positions,  InitEdges = edges,
                  alpha = a, FinalEnergy='Penalized', StoreGraphEvolution = True )
    
    edges = tree_elpi[0]['Edges'][0]
    nodes_positions = tree_elpi[0]['NodePositions']
    return {'edges': tree_elpi[0]['Edges'][0], 'node_positions': tree_elpi[0]['NodePositions']}



#Finding roots
def find_root(adj):
    N = len(adj)
    deg = np.zeros(N)
    q = Queue()
    for i in range(N):
        deg[i] = len(adj[i])
        if deg[i] == 1:
            q.put(i)
    
    left = N
    while(left > 2):
        p = q.qsize()
        left -= p
        for i in range(p):
            t = q.get()
            for j in adj[t]:
                deg[j] -= 1
                if deg[j] == 1:
                    q.put(j)
    return q.get()


def dfs(i, p, P, adj):
    P[i] = p
    for n in adj[i]:
        if not n == p:
            dfs(n, i, P, adj)
            
def find_parent(adj, root):
    P = np.zeros(len(adj), dtype = np.int32)
    P[root] = -1
    dfs(root, -1, P, adj)
    return P



def norm(x):
    return np.linalg.norm(x)
def metric(a, b):
    return distance.euclidean(a, b)
def reflect_to_zero(mu, x): #mu to zero
    if min(mu) == 0 and max(mu) == 0:
        return x
    a = mu/(norm(mu)**2) #(x,y)/x^2+y^2
    r2 = norm(a)**2 - 1
    return (r2/(norm(a-x)**2) * (x-a)) + a
def hyptoeuc(x):
    # return (exp.(x)-big(1))./(exp.(x)+big(1))
    return math.sqrt((np.cosh(x)-1)/(np.cosh(x)+1))
def add_child_c(p, x, edge_lengths, tau):
    p0 = reflect_to_zero(x, p)
    c = len(edge_lengths)
    q = norm(p0) 
    p_angle = np.arccos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi - p_angle
    #alpha = 2*math.pi/(c+1)
    points0 = np.zeros((c+1, 2))
    angles = np.zeros(c)
    alpha = 2*math.pi/(c+1)
    for i in range(c):
        angles[i] = p_angle + alpha*(i+1)
    for k in range(c):
        points0[k+1][0] = edge_lengths[k] * math.cos(angles[k])
        points0[k+1][1] = edge_lengths[k] * math.sin(angles[k])
    for k in range(c+1):
        points0[k] = reflect_to_zero(x, points0[k])  
    return points0[1:]
def add_child(p, x, edge_lengths, tau, rank):
    p0 = reflect_to_zero(x, p)
    c = len(edge_lengths)
    q = norm(p0) 
    p_angle = np.arccos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi - p_angle
    #alpha = 2*math.pi/(c+1)
    cone_angle = math.pi * math.pow(0.75, rank)
    alpha = 0
    points0 = np.zeros((c+1, 2))
    angles = np.zeros(c)
    if c > 1:
        alpha = cone_angle/(c-1)
        for i in range(c):
            angles[i] = p_angle + (math.pi-(cone_angle/2)) + alpha*i
    else:
        angles[0] = p_angle + math.pi
    for k in range(c):
        points0[k+1][0] = edge_lengths[k] * math.cos(angles[k])
        points0[k+1][1] = edge_lengths[k] * math.sin(angles[k])
    for k in range(c+1):
        points0[k] = reflect_to_zero(x, points0[k])  
    return points0[1:]
def hyp_embedding(G, root, tau, par, child):
    n = len(G)
    T = np.zeros((n, 2))
    root_children = child[root]
    d = len(root_children)
    edge_lengths = np.ones(d)
    k = 0
    for c in root_children:
        weight = G[root][c]
        edge_lengths[k] = hyptoeuc(weight*tau)
        k = k+1
    for i in range(d):
        V = np.array([math.cos(2*i*math.pi/d), math.sin(2*i*math.pi/d)])
        T[root_children[i]] = edge_lengths[i] * V
    
    q = []
    for c in root_children:
        q.append((c, 1))
    node_idx = 0
    while(len(q) > 0):
        h, rank = q[0][0], q[0][1]
        node_idx += 1
        children = child[h]
        parent = par[h]
        num_children = len(children)
        edge_lengths = np.ones(num_children)
        if len(children) > 1:
            rank += 1
        for c in children:
            q.append((c, rank))
        
        k = 0
        for c in children:
            weight = G[h][c]
            edge_lengths[k] = hyptoeuc(weight*tau)
            k = k+1
        if num_children > 0:
            R = add_child_c(T[parent], T[h], edge_lengths, tau)
            for i in range(num_children):
                T[children[i]] = R[i]
        q.pop(0)
    return T

def order(X, root_children, prev_coord):
    if prev_coord[0] == None:
        return root_children
    arr = []
    for i in range(len(root_children)):
        arr.append([distance.euclidean(prev_coord, X[root_children[i][0]]), i])
    arr.sort()
    ret = [[] for i in range(len(root_children))]
    for i in range(len(arr)):
        pos = arr[i][1]
        if i % 2 == 1:
            ret[len(arr)-1-(i//2)] = root_children[pos]
        else:
            ret[i//2] = root_children[pos]
    return ret

def bfs(G, root, num_nodes):
    adj = [[] for i in range(len(G))]
    q = PriorityQueue()
    q.put([0, root, -1])
    vis = set()
    while(len(vis) < num_nodes):
        weight, node, par = q.get()
        
        if node in vis:
            continue
        vis.add(node)
        if par != -1:
            adj[par].append([node, weight])
        for i in range(len(G)):
            if(G[node][i] == 0):
                continue
            q.put([G[node][i], i, node])
    return adj

def add_child2(X, BFS, n, root_coord, prev_coord, tau):
    T = {}
    root = n
    T[root] = root_coord
    root_children = BFS[root]
    d = len(root_children)
    edge_lengths = np.ones(d)
    root_children = order(X, root_children, prev_coord)
    for k in range(d):
        weight = root_children[k][1]
        edge_lengths[k] = hyptoeuc(weight*tau)
    for i in range(d):
        V = np.array([math.cos(2*i*math.pi/d), math.sin(2*i*math.pi/d)])
        T[root_children[i][0]] = reflect_to_zero(root_coord, edge_lengths[i] * V)
    
    q = []
    for i in range(d):
        q.append([root_children[i][0], root])
    while(len(q) > 0):
        h = q[0]
        node = h[0]
        children = BFS[node]
        parent = h[1]
        num_children = len(children)
        edge_lengths = np.ones(num_children)
        
        for k in range(num_children):
            weight = children[k][1]
            edge_lengths[k] = hyptoeuc(weight*tau)
        if num_children > 0:
            R = add_child_c(T[parent], T[node], edge_lengths, tau)
            for i in range(num_children):
                T[children[i][0]] = R[i]
        q.pop(0)
        for k in range(num_children):
            q.append([children[k][0], node])
    return T


def embedding(X, node_positions, edges, tau=0.02, root=None):
    adj = [[] for i in range(len(node_positions))]
    deg = np.zeros(len(node_positions))
    for e1, e2 in edges:
        adj[e1].append(e2)
        adj[e2].append(e1)
        deg[e1] += 1
        deg[e2] += 1
    print("Max degree: ", max(deg))
    if root == None:
        root = find_root(adj)
    P = find_parent(adj, root)
    closest_support = np.zeros((len(X), 2), dtype=np.float128)

    for i in range(len(X)):
        Z = [[metric(X[i], node_positions[j]), j] for j in range(len(node_positions))]
        closest_support[i] = min(Z)

    batch_nodes = [[] for i in range(len(node_positions))]
    for i in range(len(closest_support)):
        a, b = closest_support[i]
        batch_nodes[int(b)].append([i, a])
    
    G = np.zeros((len(node_positions), len(node_positions)), dtype=np.float128)
    maxlength = 0
    for i in range(len(node_positions)):
        for j in range(len(node_positions)):
            G[i][j] = metric(node_positions[i], node_positions[j])
            maxlength = max(maxlength, G[i][j])
    print("Max Length: ", maxlength)
    C = [[] for i in range(len(node_positions))]
    for i in range(len(node_positions)):
        C[i] = [int(z) for z in adj[i] if not z == P[i]]
    
    emb = hyp_embedding(G, root, tau, P, C)
    embX = np.zeros((len(X), 2), dtype=np.float128)
    
    for i in range(len(batch_nodes)):
        G = np.zeros((len(X)+1, len(X)+1))
        for a in batch_nodes[i]:
            for b in batch_nodes[i]:
                G[a[0]][b[0]] = distance.euclidean(X[a[0]], X[b[0]])
        for k in batch_nodes[i]:
            G[len(X)][k[0]] = k[1]
        BFS = bfs(G, len(X), len(batch_nodes[i])+1)
        par_coord = node_positions[P[i]]
        if P[i] == -1:
            par_coord = [None]
        T = add_child2(X, BFS, len(X), emb[i], par_coord, tau/(math.sqrt(len(X[0]))))
        for key, val in T.items():
            if key == len(X):
                continue
            embX[key] = val

    return emb, embX, batch_nodes



#quality measures
def map_row(H1, H2, n, row):
    edge_mask = (H1 != 0.0)
    m = np.sum(edge_mask).astype(int)
    assert m > 0
    d = H2
    sorted_dist = np.argsort(d)
    precs       = np.zeros(m)
    n_correct   = 0
    j = 0
    # skip yourself, you're always the nearest guy
    # TODO (A): j is redundant here
    mx = 0
    for i in range(1,n):
        if edge_mask[sorted_dist[i]]:
            mx = i
            n_correct += 1
            precs[j] = n_correct/float(i)
            j += 1
            if j == m:
                #print(i)
                break
    return np.sum(precs)/m

def map_score(H1, H2, n):
    maps = []
    for i in range(n):
        sorted_dist = np.argsort(H1[i])
        d = np.zeros(len(H1[i]))
        j = 0
        for z in sorted_dist:
            if H1[i][z] == 0:
                continue
            d[z] = 1
            j += 1
            if j == n//10:
                break
        maps.append(map_row(d, H2[i], n, i))
    return np.sum(maps)/n
def entry_is_good(h, h_rec): return (not np.isnan(h_rec)) and (not np.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec,me,mc):
    avg = abs(h_rec - h)/h
    if h_rec/h > me: me = h_rec/h
    if h/h_rec > mc: mc = h/h_rec
    return (avg,me,mc)

def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            (_avg,me,mc) = distortion_entry(H1[i], H2[i],me,mc)
            good        += 1
            avg         += _avg
    avg /= good if good > 0 else 1.0
    return (mc, me, avg, n-1-good)


def distortion(H1, H2, n):
    H1, H2 = np.array(H1), np.array(H2)
    dists = [distortion_row(H1[i,:],H2[i,:],n,i) for i in range(n)]
    dists = np.vstack(dists)
    mc = max(dists[:,0])
    me = max(dists[:,1])
    # wc = max(dists[:,0])*max(dists[:,1])
    avg = sum(dists[:,2])/n
    bad = sum(dists[:,3])
    return (mc, me, avg, bad)
def distance_matrix(X):
    return pairwise_distances(X, metric='minkowski')
def hyperbolic(x, y):
    return math.acosh(1 + (2 * norm(x-y)**2 / ((1 - norm(x)**2)*(1 - norm(y)**2))))
def hypmetric(feature, scale):
    n_samples = len(feature)
    D = np.ndarray(shape = (n_samples, n_samples))
    for i in range(len(D)):
        for j in range(len(D)):
            D[i][j] = scale*hyperbolic(feature[i], feature[j])
    return D
def quality_measure(x, p_x, tau):
    dist_matrix = distance_matrix(x)
    emb_matrix = hypmetric(p_x, 1/tau)
    mc, me, avg, bad = distortion(dist_matrix, emb_matrix, len(x))
    MAP = map_score(dist_matrix, emb_matrix, len(x))
    print("average distortion: ", avg)
    print("MAP: ", MAP)
    return avg, MAP
def quality_measure2(x, p_x):
    dist_matrix = distance_matrix(x)
    emb_matrix = distance_matrix(p_x)
    mc, me, avg, bad = distortion(dist_matrix, emb_matrix, len(x))
    MAP = map_score(dist_matrix, emb_matrix, len(x))
    print("average distortion: ", avg)
    print("MAP: ", MAP)
    return avg, MAP

def calculateQ(d):
    r = d.shape[0]
    q = np.zeros((r,r))
    for i in range(r):
        for j in range(r):
            if i == j:
                q[i][j] = 0
            else:
                sumI = 0
                sumJ = 0
                for k in range(r):
                    sumI += d[i][k]
                    sumJ += d[j][k]
                q[i][j] = (r-2) * d[i][j] - sumI - sumJ

    return q

def findLowestPair(q):
    r = q.shape[0]
    minVal = math.inf
    for i in range(0,r):
        for j in range(i,r):
            if (q[i][j] < minVal):
                minVal = q[i][j]
                minIndex = (i,j)
    return minIndex


def doDistOfPairMembersToNewNode(i,j,d):
    r = d.shape[0]
    sumI = 0
    sumJ = 0
    for k in range(r):
        sumI += d[i][k]
        sumJ += d[j][k]

    dfu = (1. / (2. * (r - 2.))) * ((r - 2.) * d[i][j] + sumI - sumJ)
    dgu = (1. / (2. * (r - 2.))) * ((r - 2.) * d[i][j] - sumI + sumJ)

    return (dfu,dgu)

def calculateNewDistanceMatrix(f,g,d):
    r = d.shape[0]
    nd = np.zeros((r-1,r-1))

    # Copy over the old data to this matrix
    ii = jj = 1
    for i in range(0,r):
        if i == f or i == g:
            continue
        for j in range(0,r):
            if j == f or j == g:
                continue
            nd[ii][jj] = d[i][j]
            jj += 1
        ii += 1
        jj = 1
            
    # Calculate the first row and column
    ii = 1
    for i in range (0,r):
        if i == f or i == g:
            continue
        nd[0][ii] = (d[f][i] + d[g][i] - d[f][g]) / 2.
        nd[ii][0] = (d[f][i] + d[g][i] - d[f][g]) / 2.
        ii += 1

    return nd
    
def doNeighbourJoining(d):
    #labels = ["A","B","C","D","E","F","G","H"]
    size = len(d)
    labels = [i for i in range(size-2, 2*size-2)]
    return_array = [[] for i in range(2*size - 2)]
    last = len(d) - 2
    while d.shape[0] > 2:
        q = calculateQ(d)
        lowestPair = findLowestPair(q)
        last -= 1
        newlabel = last
        i = lowestPair[0]
        j = lowestPair[1]
        if(i < j):
            i, j = j, i
        
        nodeI = labels[i]
        nodeJ = labels[j]
        del labels[i]
        del labels[j]
        labels.insert(0,newlabel)
         
        pairDist = doDistOfPairMembersToNewNode(i,j,d)
        realID = -1
        if(nodeI >= size-2):
            realID = nodeI - (size-2)
        
        return_array[last].append([nodeI, pairDist[0], realID])
        realID = -1
        if(nodeJ >= size-2):
            realID = nodeJ - (size-2)
        
        return_array[last].append([nodeJ, pairDist[1], realID])
        
        d = calculateNewDistanceMatrix(i,j,d)
    
    nodeI = labels[0]
    nodeJ = labels[1]
    realID = -1
    if(nodeJ >= size-2):
        realID = nodeJ - (size-2)
    return_array[nodeI].append([nodeJ, d[0][1], realID])
    
    return return_array
    
    
def run(distMatrix):
    return doNeighbourJoining(distMatrix)
def myadd_child2(NJ, root, root_coord, tau):
    T = {}
    Z = {}
    Z[root] = root_coord
    if(len(NJ) == 2):
        T[root] = root_coord
    
    
    root_children = NJ[root]
    d = len(root_children)
    edge_lengths = np.ones(d)

    for k in range(d):
        weight = root_children[k][1]
        edge_lengths[k] = hyptoeuc(weight*tau)
    for i in range(d):
        V = np.array([math.cos(2*i*math.pi/d), math.sin(2*i*math.pi/d)])
        if(root_children[i][2] != -1):
            T[root_children[i][2]] = reflect_to_zero(root_coord, edge_lengths[i] * V)
        Z[root_children[i][0]] = reflect_to_zero(root_coord, edge_lengths[i] * V)
    q = []
    for i in range(d):
        q.append([root_children[i][0], root])
    while(len(q) > 0):
        parent = q[0][1]
        node = q[0][0]
        children = NJ[node]
        num_children = len(children)
        edge_lengths = np.ones(num_children)
        
        for k in range(num_children):
            weight = children[k][1]
            edge_lengths[k] = hyptoeuc(weight*tau)
        if num_children > 0:
            R = add_child_c(Z[parent], Z[node], edge_lengths, tau)
            for i in range(num_children):
                if(children[i][2] != -1):
                    T[children[i][2]] = R[i]
                Z[children[i][0]] = R[i]
        q.pop(0)
        for k in range(num_children):
            q.append([children[k][0], node])
    return T
def myadd_child(p, x, proportions, edge_lengths, tau):
    p0 = reflect_to_zero(x, p)
    c = len(edge_lengths)
    q = norm(p0) 
    p_angle = np.arccos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi - p_angle
    
    points0 = np.zeros((c+1, 2))
    angles = np.zeros(c)
    
    running_sum = 0
    for i in range(c):
        angles[i] = p_angle + (math.pi/4) + (3/2)*math.pi*(running_sum + proportions[i]/2)
        running_sum += proportions[i]
    
    for k in range(c):
        points0[k+1][0] = edge_lengths[k] * math.cos(angles[k])
        points0[k+1][1] = edge_lengths[k] * math.sin(angles[k])
    for k in range(c+1):
        points0[k] = reflect_to_zero(x, points0[k])  
    return points0[1:]


def myhyp_embedding(G, root, tau, par, child, subtree_size):
    n = len(G)
    T = np.zeros((n, 2))
    
    root_children = child[root]
    
    
    d = len(root_children)
    edge_lengths = np.ones(d)
    k = 0
    for c in root_children:
        weight = G[root][c]
        edge_lengths[k] = hyptoeuc(weight*tau)
        k = k+1
        
    subtree_sum = sum([subtree_size[i] for i in root_children])    
    proportions = [subtree_size[i]/subtree_sum for i in root_children]
    running_sum = 0.0
    
    for i in range(d):
        V = np.array([math.cos(2*math.pi*(running_sum + proportions[i]/2)), 
                      math.sin(2*math.pi*(running_sum + proportions[i]/2))])
        T[root_children[i]] = edge_lengths[i] * V
        running_sum += proportions[i]
    q = []
    for c in root_children:
        q.append((c, 1))
    node_idx = 0
    
    
    while(len(q) > 0):
        h, rank = q[0][0], q[0][1]
        node_idx += 1
        children = child[h]
        parent = par[h]
        
        num_children = len(children)
        edge_lengths = np.ones(num_children)
        if len(children) > 1:
            rank += 1
        for c in children:
            q.append((c, rank))
        
        k = 0
        for c in children:
            weight = G[h][c]
            edge_lengths[k] = hyptoeuc(weight*tau)
            k = k+1
        if num_children > 0:
            subtree_sum = sum([subtree_size[i] for i in children])
            proportions = [subtree_size[i]/subtree_sum for i in children]
            R = myadd_child(T[parent], T[h], proportions, edge_lengths, tau)
            for i in range(num_children):
                T[children[i]] = R[i]
        q.pop(0)
    return T
def subtree_size_dfs(batch_nodes, subtree_size, P, C, current_node):
    subtree_size[current_node] = len(batch_nodes[current_node])
    for i in C[current_node]:
        subtree_size_dfs(batch_nodes, subtree_size, P, C, i)
        subtree_size[current_node] += subtree_size[i]
    
def rotate(T, root_coord, parent_coord, closest_coord):

    c = len(T)
    reflected = np.zeros((c, 2))
    ind = []
    k = 0
    for key, val in T.items():
        ind.append(key)
        reflected[k] = reflect_to_zero(root_coord, val) 
        k += 1
    p0 = reflect_to_zero(root_coord, parent_coord)
    q = norm(p0) 
    p_angle = np.arccos(p0[0]/q)
    if p0[1] < 0:
        p_angle = 2*math.pi - p_angle
    
    c0 = reflect_to_zero(root_coord, closest_coord)
    q = norm(c0) 
    c_angle = np.arccos(c0[0]/q)
    if c0[1] < 0:
        c_angle = 2*math.pi - c_angle
    
    angle_dif = p_angle - c_angle
    for k in range(c):
        cur = reflected[k]
        q = norm(cur)
        cur_angle = np.arccos(cur[0]/q)
        if cur[1] < 0:
            cur_angle = 2*math.pi - cur_angle
        new_angle = cur_angle + angle_dif
        reflected[k] = [q * math.cos(new_angle), q * math.sin(new_angle)]
        T[ind[k]] = reflect_to_zero(root_coord, reflected[k]) 
    return T

def myembedding(X, node_positions, edges, tau=0.02, root=None):
    pg_size = len(node_positions)
    
    
    adj = [[] for i in range(pg_size)]
    
    deg = np.zeros(len(node_positions))
    for e1, e2 in edges:
        adj[e1].append(e2)
        adj[e2].append(e1)
        deg[e1] += 1
        deg[e2] += 1
    
    if root == None:
        root = find_root(adj)
        
    
    P = find_parent(adj, root)
    C = [[] for i in range(pg_size)]
    for i in range(pg_size):
        C[i] = [int(z) for z in adj[i] if not z == P[i]]
        
    
    closest_support = np.zeros((len(X), 2), dtype=np.float128)

    for i in range(len(X)):
        Z = [[metric(X[i], node_positions[j]), j] for j in range(len(node_positions))]
        closest_support[i] = min(Z)

    batch_nodes = [[] for i in range(pg_size)]
    for i in range(len(closest_support)):
        a, b = closest_support[i]
        batch_nodes[int(b)].append([i, a])
    
    G = np.zeros((pg_size, pg_size), dtype=np.float128)
    maxlength = 0
    for i in range(pg_size):
        for j in range(pg_size):
            G[i][j] = metric(node_positions[i], node_positions[j])
            maxlength = max(maxlength, G[i][j])
            
    print("Max degree: ", max(deg))
    print("Max Length: ", maxlength)
    
    subtree_size = [[] for i in range(pg_size)]
    subtree_size_dfs(batch_nodes, subtree_size, P, C, root)
    
    empty_segments = []
    for i in range(pg_size):
        if(len(batch_nodes[i]) == 0 and sum([len(batch_nodes[j]) for j in C[i]]) == 0):
            empty_segments.append(i)
    
    emb = myhyp_embedding(G, root, tau, P, C, subtree_size)
    #emb = hyp_embedding(G, root, tau, P, C)
    embX = np.zeros((len(X), 2), dtype=np.float128)
    
    for i in range(len(batch_nodes)):
        
        if(len(batch_nodes[i]) == 0):
            continue
        print("batch_size: ", len(batch_nodes[i]))
        if(len(batch_nodes[i]) == 1):
            embX[batch_nodes[i][0][0]] = emb[i]
            continue
        if(len(batch_nodes[i]) > 100):
            max_local_dist = max([metric(X[j[0]], node_positions[i]) for j in batch_nodes[i]])
            sep_dist = np.array([metric(node_positions[i], node_positions[j]) for j in range(len(node_positions))])
            np.sort(sep_dist)
            for j in batch_nodes[i]:
                r_dist = random.random() * min(max_local_dist, sep_dist[3])
                r_angle = 2 * random.random() * math.pi



                tau_local = 1.5*tau/(math.sqrt(len(X[0])))
                edge_length = r_dist * tau_local
                edge_length = hyptoeuc(edge_length)

                p = emb[P[i]]
                x = emb[i]

                p_coord = [edge_length * math.cos(r_angle), edge_length * math.sin(r_angle)]
                p_coord = reflect_to_zero(x, p_coord)
                embX[j[0]][0], embX[j[0]][1]  = p_coord[0], p_coord[1]
            continue
        convert_to = {}
        convert_back = {}
        counter = 0
        for a in batch_nodes[i]:
            if a[0] not in convert_to.keys():
                convert_to[a[0]] = counter
                convert_back[counter] = a[0]
                counter += 1
        G = np.zeros((counter, counter))
        
        for a in batch_nodes[i]:
            for b in batch_nodes[i]:
                G[convert_to[a[0]]][convert_to[b[0]]] = distance.euclidean(X[a[0]], X[b[0]])

        NJ = run(G)
        T = myadd_child2(NJ, 0, emb[i], 3*tau/(math.sqrt(len(X[0]))))
        if(P[i] != -1):
            minkey = min([(metric(X[key], node_positions[i]), key) for key, val in T.items()])[1]
            closest_coord = T[minkey]
            T = rotate(T, emb[i], emb[P[i]], closest_coord)
        for ind, coord in T.items():
            embX[convert_back[ind]] = coord
        
        
        
        '''
        max_local_dist = max([metric(X[j[0]], node_positions[i]) for j in batch_nodes[i]])
        sep_dist = np.array([metric(node_positions[i], node_positions[j]) for j in range(len(node_positions))])
        np.sort(sep_dist)
        for j in batch_nodes[i]:
            r_dist = random.random() * min(max_local_dist, sep_dist[3])
            r_angle = 2 * random.random() * math.pi



            tau_local = 1.5*tau/(math.sqrt(len(X[0])))
            edge_length = r_dist * tau_local
            edge_length = hyptoeuc(edge_length)

            p = emb[P[i]]
            x = emb[i]

            p_coord = [edge_length * math.cos(r_angle), edge_length * math.sin(r_angle)]
            p_coord = reflect_to_zero(x, p_coord)
            embX[j[0]][0], embX[j[0]][1]  = p_coord[0], p_coord[1]
        '''
        
        '''
        G = np.zeros((len(X)+1, len(X)+1))
        for a in batch_nodes[i]:
            for b in batch_nodes[i]:
                G[a[0]][b[0]] = distance.euclidean(X[a[0]], X[b[0]])
        for k in batch_nodes[i]:
            G[len(X)][k[0]] = k[1]
        BFS = bfs(G, len(X), len(batch_nodes[i])+1)
        par_coord = node_positions[P[i]]
        if P[i] == -1:
            par_coord = [None]
        T = add_child2(X, BFS, len(X), emb[i], par_coord, tau/(math.sqrt(len(X[0]))))
        for key, val in T.items():
            if key == len(X):
                continue
            embX[key] = val 
        '''

    return emb, embX, batch_nodes

def run_hypermethyl(features, pg_nodes, scale):
    print("Doing PCA")
    X = PCA(n_components=50).fit_transform(features)
    print("Calculating Principal Graph")
    d = elastic_graph(X, n_clusters = pg_nodes)
    edges = d['edges']
    node_positions = d['node_positions']
    print("Embedding")
    emb, embX, batch_nodes = myembedding(X, node_positions, edges, tau = scale)
    return emb, embX, batch_nodes, node_positions, edges
