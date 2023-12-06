import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import scipy as sp
import scipy.io
from collections import Counter

def node_assortment(g,n,definition=1):
    """ node_assortment takes in a graph g and a node n in g, and outputs the influence assortment of node n """
    party_n = g.nodes[n]['party']
    poll = list(g.neighbors(n)) + [n] # list of nodes in the neighbourhood plus itself
    poll_size = len(poll)
    # empty neighbourhood or poll -> a_n = 1
    if poll_size<= 1:
        return 1

    poll_counts = Counter([g.nodes[m]['party'] for m in poll])
    max_poll_count = max(poll_counts.values())
    if max_poll_count == poll_counts[party_n]:
        return poll_counts[party_n]/poll_size
    else:
    	if definition == 1:
    	    return -max_poll_count/poll_size
    	elif definition ==2:
    	    return -(1-poll_counts[party_n]/poll_size)
    	else:
    	    raise ValueError('node_assortment definition can only be 1 or 2.')

def influence_gap(g,p=1,definition=1):
    """ influence_gap takes a graph g and party p and returns the
    value of influence gap for party p.
    definition takes values 1 or 2:
        - [1] A_p - max(A_Q)
        - [2] A_p - A_{argmax N_Q}
    """
    assortment_of_parties = all_party_assortment(g)
    A_p = assortment_of_parties.pop(p)
    if definition == 1:
        return A_p - max(assortment_of_parties.values())
    elif definition == 2:
        party_count = Counter(nx.get_node_attributes(g,'party').values())
        winning_votes = max(party_count.values())
        winning_parties = [q for q,v in party_count.items() if v==winning_votes]
        winning_assortments = {q:A_q for q,A_q in assortment_of_parties if q in winning_parties}
        return A_p - max(winning_assortments.values()) # fixed to give unique sols only

# ---------------- HELPER FUNCTIONS ----------------------

def graph_assortment(g):
    """ graph_assortment returns the node_assortment for every node in g as a numpy array """
    assortments = np.array([node_assortment(g,n) for n in g])
    return assortments

def overall_assortment(g):
    """ overall_assortment is a wrapper to find the overall assortment, ie mean assortment
    of a graph g"""
    return np.mean(graph_assortment(g))

def party_assortment(g,p):
    """ party_assortment takes a graph g, and a party p and returns A_p;
    i.e. the mean of all node_assortments of voters of p
    """
    return np.mean([node_assortment(g,n) for n in g if g.nodes[n]['party']==p])

def all_party_assortment(g):
    """ all_party_assortment takes a graph g (which has already been assigned P parties)
    and returns the party assortment of (average node assortment for party p)
    as a dictionary of P key-value pairs {p:A_p} """
    # Extracting the parties and assortments
    parties = set(nx.get_node_attributes(g,'party').values())
    return {p:party_assortment(g,p) for p in parties}

def poll_fraction(g,n):
    """ poll_fraction returns the poll (as a fraction) of node n in graph g """
    party = g.nodes[n]['party']
    delta = 1 # Since the node knows it will vote for its own party
    poll = list(g.neighbors(n)) +[n]
    poll_count = Counter([g.nodes[m]['party'] for m in poll])
    # for m in neighbours:
    #     if g.nodes[m]['party'] == party:
    #         delta+=1
    # delta = delta/(len(neighbours)+1)
    return poll_count[party]/len(poll)#delta # Precisely Delta_n as in Stewart et al

# ---------------- OTHER METRICS --------------------------
def dVS_gap(g,step=1):
    next_round_vote = np.zeros(len(g))
    parties = nx.get_node_attributes(g,'party')
    for n in g:
        poll = list(g.neighbors(n)) + [n] # list of nodes in the neighbourhood plus itself
        poll_counts = Counter([g.nodes[m]['party'] for m in poll])
        max_poll_count = max(poll_counts.values())
        if len(poll) <= 1:
            next_round_vote[n] = parties[n]
        elif max_poll_count == poll_counts[parties[n]]:
            next_round_vote[n] = parties[n]
        else:
            winners = [p for p in set(parties.values()) if poll_counts[p]==max_poll_count]
            next_round_vote[n] = np.random.choice(winners)
    if step <= 1:
        if len(set(parties.values())) == 1:
            return 0.5
        elif len(set(parties.values())) == 2:
            return len(next_round_vote[next_round_vote==1])/len(g) - 0.5
        else:
            return {p:v/len(g) for p,v in Counter(next_round_vote.values()).items()}
    else:
        h = g.copy()
        nx.set_node_attributes(h,{n:DG[n] for n in h},'party')
        return dVS_gap(h,step=step-1)

def wasted_vote(g,n):
    party = g.nodes[n]['party']
    neighbours = list(g.neighbors(n))
    frac = poll_fraction(g,n)
    if frac >= 1/2:
        A = 1
        B = 0
        for neigh in neighbours:
            if g.nodes[neigh]['party'] == party:
                A+=1
            else:
                B+=1
        return int(np.ceil((A-B)/2)-1)
    else:
        friends = 1
        for neigh in neighbours:
            if g.nodes[neigh]['party'] == party:
                friends += 1
        return friends

def efficiency_gap(g):
    wasted_votes = np.array([wasted_vote(g,n) for n in g])
    red_wv = [wasted_votes[n] for n in g if g.nodes[n]['party'] == 1]
    blu_wv = [wasted_votes[n] for n in g if g.nodes[n]['party'] == 0]
    return np.sum(red_wv) - np.sum(blu_wv)

# ----------------------- GRAPH GENERATION-----------------------------
def homophilic_relaxed_caveman_graph(l, k, p0, h, parties, seed=None,iterations=1):
    """Returns a homophilic relaxed caveman graph.

    A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are
    then randomly rewired with probability `p0*h` if both nodes have the
    same party. Else they are rewired with p0*(1-h). `parties` must be a
    dictionary, list or ndarray of size l*k

    Parameters
    ----------
    l : int
      Number of groups
    k : int
      Size of cliques
    p0 : float
      Probabilty of rewiring each edge.
    h : float
      Homophily
    parties : dict, list, ndarray
      party assignment
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    iterations : integer >= 1
        number of passes over the entire edge set when rewiring, classically 1

    Modification of nx.relaxed_caveman_graph source code
    """
    G = nx.caveman_graph(l, k)
    np.random.seed(seed)
    nx.set_node_attributes(G,parties,'party')
    nodes = list(G)
    x = np.random.choice(nodes,size=len(G.edges()))
    probs = np.random.rand(len(G.edges()))
    if type(h) != dict:
        h = {p:h for p in set(list(parties.values()))}
    for t in range(iterations):
        for i,(u, v) in enumerate(G.edges()):
            p_x = G.nodes[x[i]]['party']
            p_u = G.nodes[u]['party']
            if p_x == p_u:
                p = p0*h[p_u]
            else:
                p = p0*(1-h[p_u])
            if probs[i] < p:  # rewire the edge
                if G.has_edge(u, x[i]):
                    continue
                G.remove_edge(u, v)
                G.add_edge(u, x[i])
    return G


""" ----------------- Dynamics --------------- """

# Importing empirical data from Stewart et al
mat = sp.io.loadmat('dynamic_parameters.mat')
# Basically the heights of the bars from the empirical distributions, from which
    # we sample
par_dist = np.zeros((3,2,11))
# Distributions for seeing a win in the early and late phases
par_dist[0,0,:] = mat['Win1'].reshape(12)[:-1]
par_dist[0,1,:] = mat['Win2'].reshape(12)[:-1]
# Distributions for seeing a deadlock in the early and late phases
par_dist[1,0,:] = mat['Deadlock1'].reshape(12)[:-1]
par_dist[1,1,:] = mat['Deadlock2'].reshape(12)[:-1]
# Distributions for seeing a loss in the early and late phases
par_dist[2,0,:] = mat['Lose1'].reshape(12)[:-1]
par_dist[2,1,:] = mat['Lose2'].reshape(12)[:-1]

def dynamics(g,last=False,ass=False,full_ig=False):
    """ dynamics takes a graph g already with a party assignment and runs the
    dynamic model on it. By default returns the voting intetions at all times

    Parameters
    ----------
    g --> graph with numerical parties (float, int)
    last --> returns the final voting intentions only
    ass --> returns the influence assortment of both parties at all times
    full_ig --> returns influence gap AND voter skew at all times

    last overwrites ass which overwrites full_ig.
    """
    V = 0.6 # Super-majority
    T = 240 # Total length of game in seconds
    t_decision = 3.3 # The time between updates
    t = 0
    t_transition = 83 # Transition between early and late
    no_decisions = int(T//t_decision) # Number of updates

    # Initial poll fractions
    polls = np.array([poll_fraction(g,n) for n in g])
    # initial parties as an N-sized array
    initial_parties = np.array(list(nx.get_node_attributes(g,'party').values()),dtype=int)
    # set of parties present
    set_of_parties = set(nx.get_node_attributes(g,'party').values())

    # Voting intentions at time t
    intentions = np.zeros((no_decisions+1,len(g)))
    # Initial voting intention is the party assignment
    intentions[0,:] = initial_parties

    if full_ig:
        IG = np.zeros(no_decisions+1)
        IG[0] = influence_gap(g)

    if ass:
        assortment = np.zeros((no_decisions+1,2))
        assortment[0,:] = list(party_assortment(g).values())

    rand_numbers = np.random.rand(no_decisions+1,len(g))

    # For each person, sample once from each of the 6 distributions to get the
        # probability to stick during phase j {early,late} and seeing polls that
        # predict state i {win,deadlock,loss}
    pars = np.zeros((3,2,len(g)))
    for i in range(3):
        for j in range(2):
            # parameter : 0-1, in 11 bins
            pars[i,j,:] = np.random.choice(np.linspace(0,1,11),p=par_dist[i,j,:],size=len(g))

    phase = 0 # Early
    for t_index in range(1,no_decisions+1):
        t += t_decision
        if t > 83:
            phase = 1 # Late
        # Essentially saying which environment {i,j} a person n is in
        # the probability to stick to the INITIAL party
        pars_now = np.zeros(len(g))
        pars_now[polls>=V] = pars[0,phase,polls>=V]
        pars_now[(polls>=1-V)&(polls<V)] = pars[1,phase,(polls>=1-V)&(polls<V)]
        pars_now[polls<1-V] = pars[2,phase,polls<1-V]

        if full_ig:
            IG[t_index] = influence_gap(g)

        for n in g:
            # with prob p_ij stick to the INITIAL party
            if rand_numbers[t_index,n] <= pars_now[n]:
                intentions[t_index,n] = initial_parties[n]

            # with prob 1-p_ij swap to a plurality party in one's poll
            else:
                poll_n = list(g.neighbors(n))+[n]
                parties_poll_n = [g.nodes[m]['party'] for m in poll_n]
                count_poll_n = Counter(parties_poll_n)
                options_for_n = [p for p,v in count_poll_n.items() if v==max(count_poll_n.values())]
                intentions[t_index,n] = np.random.choice(options_for_n)
        parties_dict = {n:intentions[t_index,n] for n in g}
        nx.set_node_attributes(g,parties_dict,'party')
        #--------------- CORRECTION LINE 27/5/21-----------------
        polls = np.array([poll_fraction(g,n) for n in g])
        #--------------- END OF CORRECTION 27/5/21---------------
        if ass:
            assortment[t_index,:]=list(party_assortment(g).values())
    if last:
        return intentions[-1,:]
    elif ass:
        return assortment
    elif full_ig:
        return IG,np.sum(intentions,axis=1)/len(g) - 0.5
    else:
        return intentions


""" ----------------- Drawing functions --------------------"""

def draw_assortment(g,pos=None,
                    cmap=None,vmin=None,vmax=None,clabel=True,inf_bar=True):
    """ draw_assortment takes a graph g and draws it. Each node is labelled by its party and
    coloured by the influence assorment (ass) or party (par)"""
    if inf_bar:
        fig = plt.gcf()
        ax = fig.add_axes([0,0,0.85,1])
    else:
        ax = plt.gca()
    if pos == None:
        pos = nx.shell_layout(g)
    if cmap == None:
        cmap = 'bwr'
    color = np.array(list(nx.get_node_attributes(g,'party').values()))
    color[color==0] = -1
    node_inf = graph_assortment(g)
    color = color * (node_inf+1)
    clab = "Influence Assortment"
    if vmin == None:
        vmin = -2
    if vmax == None:
        vmax = 2
    nx.draw(g,pos=pos,
        node_color=list(color),cmap=cmap, node_size=500,vmin=vmin,vmax=vmax,ax=ax)
    if inf_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        caxb = plt.gcf().add_axes([0.85,0.1,0.035,0.8])
        caxr = plt.gcf().add_axes([0.85+0.035,0.1,0.035,0.8])
        cbarb=plt.colorbar(sm,cax=caxb,boundaries=np.linspace(-2,0,100),ticks=[-0,-0.5,-1,-1.5,-2])
        cbarb.ax.set_yticklabels([])
        cbarb.ax.tick_params(axis='y',direction='in')
        cbarb.ax.invert_yaxis()
        cbarr=plt.colorbar(sm,cax=caxr,boundaries=np.linspace(0,2,100),ticks=[0,0.5,1,1.5,2])
        cbarr.ax.set_yticklabels([-1,-0.5,0,0.5,1])
        cbarr.ax.tick_params(axis='y',direction='in')

def draw_influence_subplot(a,win,lose):
    size = len(win)
    a.scatter(range(size),win,c='red',marker='+',alpha=0.5)
    a.scatter(range(size),lose,c='blue',marker='x',alpha=0.5)
    a.plot([0,size-1],[np.mean(win)]*2,c='k',lw=4.5)
    a.plot([0,size-1],[np.mean(win)]*2,c='red',lw=3,label=r'$\mathcal{A}_W = $%.3f'%np.mean(win))
    a.plot([0,size-1],[np.mean(lose)]*2,c='k',lw=4.5)
    a.plot([0,size-1],[np.mean(lose)]*2,c='blue',lw=3,label=r'$\mathcal{A}_L = $%.3f'%np.mean(lose))
    a.legend(loc='lower right',framealpha=0.9)
    a.set_xticklabels([])
    a.tick_params(axis='x',which='both',bottom=False)
    a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

def draw_gap_subplot(a,win,lose):
    size = len(win)
    a.scatter(range(size),win-lose,c='k',marker='+',alpha=0.5)
    a.plot([0,size-1],[np.mean(win-lose)]*2,c='k',lw=3,label=r'$\mathcal{A}_W = $%.3f'%np.mean(win-lose))
    a.legend(loc='lower right',framealpha=0.9)
    a.set_xticklabels([])
    a.tick_params(axis='x',which='both',bottom=False)

def draw_ig_boxplot(df,cols=None,title=None,ax=None,mean=True):
    """ draw_ig_boxplot takes in a dataframe of influence gaps and plots it """
    if ax == None:
        ax=plt.gca()
    else:
        ax
    if cols == None:
        cols = list(df)
    if title == None:
        title="Effects of Party Assignment and Graph Strutcture"
    sns.boxplot(data=df[cols],palette='Set2')
    if mean:
        labels = ["%.3f"%np.mean(df[col]) for col in cols]
        pos = range(len(labels))
        medians = [np.median(df[col]) for col in cols]
        means = [np.mean(df[col]) for col in cols]
        for tick,label in zip(pos,ax.get_xticklabels()):
            ax.text(pos[tick], means[tick], labels[tick],
            horizontalalignment='center', size='x-small', color='k', weight='semibold')
    ax.set_ylabel("Influence Gap")
    ax.set_title(title)


""" --------------- Utility Functions -------------------"""

def PA_to_hex(node_list,N):
    parties = [str(int(n in node_list)) for n in range(N)]
    return hex(int(''.join(parties),2))

def PA_to_dec(node_list,N):
    parties = [str(int(n in node_list)) for n in range(N)]
    return (int(''.join(parties),2))

def hRC_wrapper(size=1000,pas=100,h_size=3,N=20):
    IG = np.zeros((size,pas,h_size))
    h = np.linspace(0.5,1,h_size)
    node_list = np.transpose(np.array([np.random.choice(range(N),size=(N//2),replace=False) for i in range(pas)]))
    if (len(np.unique([PA_to_dec(node_list[:,j],N) for j in range(pas)]))) == pas:
        for i in trange(size):
            g_hRC = [[homophilic_relaxed_caveman_graph(2,N//2,0.1,h[k],
                                                         {n:int(n in node_list[:,j]) for n in range(N)})
                     for j in range(pas)] for k in range(h_size)]
            for j in range(pas):
                for k in range(h_size):
                    IG[i,j,k] = np.abs(np.diff(list(party_assortment(g_hRC[k][j]).values()))[0])

    names = ['h', 'PA', 'graph']
    index = pd.MultiIndex.from_product([h,[PA_to_hex(node_list[:,j],N) for j in range(pas)],range(size)], names=names)
    df = pd.DataFrame({'IG': IG.flatten()}, index=index)['IG'].unstack(level='PA').sort_index()
    return df

def hex_to_PA(hx,N,party_dict=True):
    bitstr = bin(int(hx,16))[2:].zfill(N)
    out = {n:int(bitstr[n]) for n in range(20)}
    if party_dict:
        return out
    else:
        return np.array(list(out.keys()))[np.array(list(out.values()))==1]
