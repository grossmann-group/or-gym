import networkx as nx
import numpy as np
graph = nx.DiGraph()
graph.add_nodes_from([0], dist = 1,
                                dist_param = {'mu': 20})
graph.add_nodes_from([1], I0 = 100,
                                p = 2.000,
                                r = 1.500,
                                b = 0.100,
                                h = 0.150,
                                g = 0.000)
graph.add_nodes_from([2], I0 = 100,
                                C = 100,
                                p = 1.500,
                                r = 1.000,
                                # b = 0.075,
                                h = 0.100,
                                g = 0.000)
graph.add_nodes_from([3], I0 = 200,
                                C = 90,
                                p = 1.000,
                                r = 0.750,
                                # b = 0.050,
                                h = 0.050,
                                g = 0.000)
graph.add_nodes_from([4], I0 = np.Inf,
                                C = 80,
                                p = 0.750,
                                r = 0.500,
                                # b = 0.025,
                                h = 0.000,
                                g = 0.000)
graph.add_nodes_from([5])
graph.add_edges_from([(1,0),
                            (2,1,{'L': 3}),
                            (3,2,{'L': 5}),
                            (4,3,{'L': 10}),
                            (5,4,{'L': 0})])
