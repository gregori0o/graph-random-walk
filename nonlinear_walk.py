import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime


class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen


class NonlinearRandomWalk:
    """Nonlinear random walk algorithm.

    :param dict G: the graph
    """

    def __init__(self, G):
        self.G = G
        self.nodes = G.nodes()
        self.matrix = nx.adjacency_matrix(G, nodelist=self.nodes)
        self.N = G.number_of_nodes()
        self.pos = nx.spring_layout(G)

        self.ax = None
        self.alphas = None
        self.prs = None
        self.old_prs = None

    def simulate(self, alpha=1, d=1.0, tol=1e-2, max_iter=100):
        """Generator of the probabilities of the nodes in the graph for nonlinear random walk.

        :param float alpha: the alpha parameter of random walk
        :param float d: the damping factor
        :param flat tol: tolerance to determine algorithm convergence
        :param int max_iter: max number of iterations
        """

        pr = np.ones(self.N).reshape(self.N, 1) * 1./self.N

        # need to repeat the initial step twice
        # for matplotlib animation
        yield self.nodes, pr, "init"
        yield self.nodes, pr, "init"

        for it in range(max_iter):
            old_pr = pr[:]
            exp_pr = np.exp(alpha*pr)
            transition_matrix = self.matrix.T * exp_pr.reshape(-1, 1)
            transition_matrix = transition_matrix/transition_matrix.sum(axis=0)
            pr = d * (transition_matrix @ pr) + (1-d)/self.N
            yield self.nodes, pr, it
            err = np.absolute(pr - old_pr).sum()
            if err < tol:
                break
        return pr.reshape(-1), old_pr.reshape(-1)
    
    def _update(self, r):
        res_nodes, res_values, it = r
        res_values = np.asarray(res_values).ravel()
        plt_nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            ax=self.ax,
            nodelist=res_nodes,
            node_color=res_values,
            alpha=1,
            node_size=700,
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=0.2
        )
        self.ax.axis("off")
        self.ax.set_title(f"Iteration {it}")
        nx.draw_networkx_edges(self.G, self.pos, width=4)
        nx.draw_networkx_labels(self.G, self.pos, font_size=14)
        return [plt_nodes, ]
    
    def get_visualization(self, alpha=1, d=1.0, tol=1e-2, max_iter=100, filename=None):
        """Return the visualization of the probabilities of the nodes in the graph for nonlinear random walk.

        :param float alpha: the alpha parameter of random walk
        :param float d: the damping factor
        :param flat tol: tolerance to determine algorithm convergence
        :param int max_iter: max number of iterations
        :param str filename: the filename of the gif
        """
        f, self.ax = plt.subplots()
        ani = FuncAnimation(
            f,
            self._update,
            frames=self.simulate(alpha, d, tol, max_iter),
            interval=1000,
            blit=True
        )
        f.suptitle(f"  Nonlinear Random Walk")
        if filename is None:
            filename = f"nonlinear_random_walk-{datetime.now().isoformat()}.gif"
        ani.save(f"gifs/{filename}", writer='imagemagick')

    def calculate_for_alphas(self, min_alpha=-6, max_alpha=6, num_alpha=200, d=1.0, tol=1e-6, max_iter=1000):
        """Calculate the probabilities of the nodes in the graph for nonlinear random walk for different values of alpha.

        :param float min_alpha: the minimum value of alpha
        :param float max_alpha: the maximum value of alpha
        :param int num_alpha: the number of values of alpha
        :param float d: the damping factor
        :param flat tol: tolerance to determine algorithm convergence
        :param int max_iter: max number of iterations
        """
        self.alphas = np.linspace(min_alpha, max_alpha, num_alpha)
        self.prs = np.zeros((num_alpha, self.N), dtype=float)
        self.old_prs = np.full((num_alpha, self.N), fill_value=-1, dtype=float)
        for i, alpha in enumerate(self.alphas):
            generator = GeneratorWrapper(self.simulate(alpha, d, tol, max_iter))
            for _ in generator:
                pass
            pr, old_pr = generator.value
            self.prs[i, :] = pr
            if np.sum(np.abs(pr - old_pr)) > tol:
                self.old_prs[i, :] = old_pr

    def plot_for_alphas(self, nodes_to_plot=None, filename=None):
        """Plot the probabilities of the nodes in the graph for nonlinear random walk for different values of alpha.

        :param list nodes_to_plot: the list of indices of nodes to plot
        :param str filename: the filename of the plot
        """
        if self.alphas is None or self.prs is None:
            raise ValueError("Please run calculate_for_alphas() first.")

        if nodes_to_plot is None:
            nodes_to_plot = np.arange(self.N)

        plt.figure()
        for node in nodes_to_plot:
            mask = (self.old_prs[:, node] >= 0).reshape(-1)
            alphas = np.r_[self.alphas, self.alphas[mask]]
            prs = np.r_[self.prs[:, node], self.old_prs[mask, node]]

            plt.scatter(alphas, prs, label=f"{node} - {self.nodes[node]}", s=3)
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Probability")
        plt.title("Nonlinear Random Walk")
        if filename is None:
            filename = f"nonlinear_random_walk-{datetime.now().isoformat()}.png"
        plt.savefig(filename)
        plt.show()
