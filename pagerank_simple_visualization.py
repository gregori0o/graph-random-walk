import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx


def page_rank(G, d=0.95, tol=1e-2, max_iter=100):
    """Return the PageRank of the nodes in the graph.

    :param dict G: the graph
    :param float d: the damping factor
    :param flat tol: tolerance to determine algorithm convergence
    :param int max_iter: max number of iterations
    """
    nodes = G.nodes()
    matrix = nx.adjacency_matrix(G, nodelist=nodes)
    out_degree = matrix.sum(axis=0)
    weight = matrix / out_degree
    N = G.number_of_nodes()
    pr = np.ones(N).reshape(N, 1) * 1./N

    # need to repeat the initial step twice
    # for matplotlib animation
    yield nodes, pr, "init"
    yield nodes, pr, "init"

    for it in range(max_iter):
        old_pr = pr[:]
        pr = d * weight.dot(pr) + (1-d)/N
        yield nodes, pr, it
        err = np.absolute(pr - old_pr).sum()
        if err < tol:
            return pr


def update(r):
    res_nodes, res_values, it = r
    res_values = np.asarray(res_values).ravel()
    plt_nodes = nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        nodelist=res_nodes,
        node_color=res_values,
        alpha=1,
        node_size=700,
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=0.2
    )
    ax.axis("off")
    ax.set_title(f"Iteration {it}")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=14)
    return [plt_nodes, ]


G = nx.karate_club_graph()
pos = nx.circular_layout(G)

f, ax = plt.subplots()
ani = FuncAnimation(
    f,
    update,
    frames=page_rank(G),
    interval=1000,
    blit=True
)
f.suptitle(f"  Page Rank")
ani.save("gifs/graph_pr.gif", writer='imagemagick')