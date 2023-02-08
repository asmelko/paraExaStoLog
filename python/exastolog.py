from os.path import dirname, join
import numpy as np
import boolean
import scipy
import graphlib


class Model:

    def __init__(self, bnet_filename):
        self.model = self.readBNet(bnet_filename)

    def readBNet(self, filename):
        model = {}
        algebra = boolean.BooleanAlgebra()
        with open(filename, 'r') as f:
            lines = f.readlines()
            del lines[0]
            for line in lines:
                species, formula = [value.strip() for value in line.split(",")]
                b_formula = algebra.parse(formula).simplify()
                model.update({species: b_formula})

        return model


class TransitionTable:

    def __init__(self, model: Model):
        self.model = model.model

    def build_transition_table(self):
        nodes = list(self.model.keys())
        n = len(nodes)
        states_count = 2 ** n
        states = np.remainder(
            np.floor(
                np.multiply(
                    np.array([range(states_count)]).transpose(),
                    np.array(
                        [np.power([2.0]*n, np.array(range(0, -n, -1)))])
                )
            ), 2
        ).astype(bool)

        state_updates = np.array(
            [
                self.fcn_gen_node_update(self.model[node], states, nodes)
                for node in nodes
            ]
        ).transpose()

        trans_source = np.argwhere(
            state_updates != states)

        transitions_src = []
        transitions_dest = []
        for t in trans_source:
            transitions_src.append(t[0])
            if (state_updates[t[0], t[1]] == True):
                transitions_dest.append(t[0] + 2 ** t[1])
            else:
                transitions_dest.append(t[0] - 2 ** t[1])

        self.transition_table = scipy.sparse.csr_matrix(
            (
                [1]*len(transitions_dest),
                (transitions_dest,
                 transitions_src)
            ),
            shape=(states_count, states_count)
        )

    def fcn_gen_node_update(self, formula, list_binary_states, nodes):

        if isinstance(formula, boolean.boolean.Symbol):
            return list_binary_states[:, nodes.index(str(formula))]

        elif isinstance(formula, boolean.boolean.NOT):
            return np.logical_not(
                self.fcn_gen_node_update(
                    formula.args[0], list_binary_states, nodes)
            )

        elif isinstance(formula, boolean.boolean.OR):
            ret = self.fcn_gen_node_update(
                formula.args[0], list_binary_states, nodes)
            for i in range(1, len(formula.args)):
                ret = np.logical_or(ret,
                                    self.fcn_gen_node_update(
                                        formula.args[i], list_binary_states, nodes)
                                    )
            return ret

        elif isinstance(formula, boolean.boolean.AND):
            ret = self.fcn_gen_node_update(
                formula.args[0], list_binary_states, nodes)
            for i in range(1, len(formula.args)):
                ret = np.logical_and(ret,
                                     self.fcn_gen_node_update(
                                         formula.args[i], list_binary_states, nodes)
                                     )
            return ret

        else:
            print("Unknown boolean operator : %s" % type(formula))


class TransitionGraph:

    def __init__(self, table: TransitionTable):
        self.table = table.transition_table

    def create_metagraph(self, table):
        count, labels = scipy.sparse.csgraph.connected_components(
            table, directed=True, connection='strong')

        metagraph = {}

        sccomponents = []

        for i in range(count):
            sccomponents.append(set())
            metagraph.update({i: set()})

        for i in range(len(labels)):
            sccomponents[labels[i]].add(i)
            incoming = table[i].indices
            for inc in incoming:
                metagraph[labels[i]].add(labels[inc])

        return sccomponents, metagraph

    def build_adjacency_matrix(self, sccs, sorted_metavertices):

        sorted_vertices = []

        for v in sorted_metavertices:
            sorted_vertices = sorted_vertices + list(sccs[v])

        indices = []
        indptrs = [0]

        for v in sorted_vertices:
            incoming = self.table[v].indices

            indptrs.append(indptrs[-1] + len(incoming))

            for i in incoming:
                indices.append(sorted_vertices.index(i))

        sorted_table = scipy.sparse.csr_matrix(
            ([1] * len(indices), indices, indptrs), shape=(len(sorted_vertices), len(sorted_vertices)))

        print(self.table.todense())

        print(sorted_vertices)

        print(indptrs)

        print(sorted_table.todense())

        return sorted_table

    def sort(self):
        count, x = scipy.sparse.csgraph.connected_components(
            self.table, directed=True, connection='weak')

        # if count != 1:
        #     print("multiple graph components not yet supported")
        #     exit(0)

        sccs, metagraph = self.create_metagraph(self.table)

        ts = graphlib.TopologicalSorter(metagraph)
        sccs_ordering = list(ts.static_order())

        self.sorted_table = self.build_adjacency_matrix(
            sccs, sccs_ordering)

        def is_terminal(component):
            for i in reversed(sccs_ordering):
                if component in metagraph[i]:
                    return False
            return True

        self.terminal_offsets = []

        for i in range(len(sccs_ordering) - 1, -1, -1):

            if not is_terminal(sccs_ordering[i]):
                self.terminal_offsets.append(i+1)
                break

        for i in range(self.terminal_offsets[0], len(sccs_ordering)):
            self.terminal_offsets.append(
                self.terminal_offsets[-1] + len(sccs[sccs_ordering[i]]))


class Solution:

    def __init__(self, graph: TransitionGraph, nodes_count):
        self.table = graph.sorted_table
        self.offsets = graph.terminal_offsets
        self.n = nodes_count

    def create_kinetic(self, trans_table):
        uni_weights = 1/(2*self.n)
        K = uni_weights * trans_table
        return K - scipy.sparse.diags(np.array(K.sum(axis=0))[0])

    def compute_scc_kernel(self, terminal_index):

        state_start = self.offsets[terminal_index]
        state_end = self.offset[terminal_index + 1]

        scc_transition_table = self.table[state_start:
                                          state_end][state_start:state_end]

        K = self.create_kinetic(scc_transition_table).todense()

        sign = np.empty((K.shape[0]))
        sign[::2] = 1
        sign[1::2] = -1

        return scipy.linalg.inv(K)[:, 0] * sign * scipy.linalg.det(K)

    def compute_column_nullspace(self):

        terminal_start = self.offsets[0]

        data = []
        indices = []
        indptrs = [0] * (terminal_start + 1)

        for i in range(len(self.offsets) - 1):
            scc_size = self.offsets[i+1] - self.offsets[i]
            indptrs.append(indptrs[-1] + scc_size)

            indices += [self.offsets[i+1] - terminal_start - 1] * scc_size
            if scc_size == 1:
                data.append(1)
            else:
                data += self.compute_scc_kernel(i)

        return scipy.sparse.csr_matrix((data, indices, indptrs), shape=(self.table.shape[0], self.table.shape[0] - terminal_start))

    def compute_row_nullspace(self, column_nullspace):

        terminal_start = self.offsets[0]

        U = column_nullspace[terminal_start:, :].transpose()
        U.data = np.array([1] * len(U.data))

        print(U.todense())

        K = self.create_kinetic(self.table)
        N = K[:terminal_start, :terminal_start]
        B = K[terminal_start:, :terminal_start]

        print(K.todense())
        print(N.todense())
        print(B.todense())

        X = -U @ B @ scipy.sparse.linalg.inv(N)

        return scipy.sparse.hstack((X, U))

    def compute_nullspaces(self):
        R = self.compute_column_nullspace()
        print(R.todense())
        L = self.compute_row_nullspace(R)
        print(L.todense())

        print((L @ R).todense())

        return L, R


model = Model(join(dirname(__file__), "../data/toy-paper.bnet"))
table = TransitionTable(model)
table.build_transition_table()
graph = TransitionGraph(table)
graph.sort()
solution = Solution(graph, len(model.model.keys()))
solution.compute_nullspaces()
