import pandas as pd
from os.path import dirname, join
import numpy as np
import boolean
import scipy


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


class InitialState:

    def __init__(self, model: Model, fixed_nodes=[], fixed_values=[], fixed_probability=1):
        nodes = list(model.model.keys())
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

        fixed_indices = [nodes.index(f) for f in fixed_nodes]
        fixed_state = [v == 1 for v in fixed_values]

        fixed_state_mask = np.sum(
            np.array(fixed_state) == states[:, fixed_indices], axis=1) == len(fixed_nodes)

        fixed_state_indices_count = np.sum(fixed_state_mask)
        nonfixed_state_indices_count = states_count - fixed_state_indices_count

        self.x_0 = np.empty((states_count))

        self.x_0[fixed_state_mask] = fixed_probability / \
            fixed_state_indices_count

        self.x_0[~fixed_state_mask] = (
            1 - fixed_probability) / nonfixed_state_indices_count


class TransitionGraph:

    def __init__(self, table: TransitionTable, initial_state: InitialState):
        self.table = table.transition_table
        self.sorted_subgraphs = []
        self.initial_state = initial_state.x_0

    def toposort(self, table):
        ordering = np.empty((table.shape[0]), dtype=int)
        terminals_idx = None
        fill_idx = table.shape[0]
        while table.shape[0] != 0:
            mask = np.array(table.sum(axis=0) == 0)[0]
            table = table[~mask, :][:, ~mask]

            indices = np.argwhere(mask).flatten()
            fill_idx -= indices.shape[0]
            ordering[fill_idx: fill_idx + indices.shape[0]] = indices

            if terminals_idx is None:
                terminals_idx = fill_idx

        return ordering, terminals_idx

    def create_metagraph(self, table):
        count, labels = scipy.sparse.csgraph.connected_components(
            table, directed=True, connection='strong')

        sccomponents = []

        for i in range(count):
            sccomponents.append(set())

        for i in range(len(labels)):
            sccomponents[labels[i]].add(i)

        # All SCCs are single vertices => metagraph = graph
        if count == table.shape[0]:
            return sccomponents, table

        rows, columns = table.nonzero()

        sccs_transitions_mask = labels[rows] != labels[columns]

        row_transitions = rows[sccs_transitions_mask]
        col_transitions = columns[sccs_transitions_mask]

        metagraph = scipy.sparse.csr_matrix(
            ([1] * len(row_transitions),
             (labels[row_transitions], labels[col_transitions])),
            shape=(count, count)
        )

        metagraph.data = np.ones((metagraph.data.shape))

        return sccomponents, metagraph

    def build_sorted_transition_table(self, original_table, sccs, sorted_metavertices):

        sorted_vertices = []

        for v in sorted_metavertices:
            sorted_vertices = sorted_vertices + list(sccs[v])

        return original_table[sorted_vertices, :][:, sorted_vertices], sorted_vertices

    def sort(self):
        count, disconnected_subgraphs = scipy.sparse.csgraph.connected_components(
            self.table, directed=True, connection='weak')

        for i in range(count):
            subgraph_indices = np.argwhere(
                disconnected_subgraphs == i).flatten()

            # whole component is within empty part of initial state => skip
            if np.all(self.initial_state[subgraph_indices] == 0):
                continue

            subtable = self.table[subgraph_indices, :][:, subgraph_indices]

            subnodes_count = len(subgraph_indices)

            if (subnodes_count == 1):
                self.sorted_subgraphs.append((1, [[0]], [0, 1], [0]))
                continue

            sccs, metagraph = self.create_metagraph(subtable)

            if len(sccs) == 1:
                self.sorted_subgraphs.append(
                    (subgraph_indices, subtable, [0, subnodes_count], list(range(subnodes_count))))
                continue

            sccs_ordering, terminal_start_idx = self.toposort(metagraph)

            sorted_subtable, sorted_subvertices = self.build_sorted_transition_table(subtable,
                                                                                     sccs, sccs_ordering)

            terminal_offsets = [0]

            for i in range(terminal_start_idx, len(sccs_ordering)):
                terminal_offsets.append(
                    terminal_offsets[-1] + len(sccs[sccs_ordering[i]]))

            terminal_offsets = [
                x + (subnodes_count - terminal_offsets[-1]) for x in terminal_offsets]

            self.sorted_subgraphs.append(
                (subgraph_indices, sorted_subtable, terminal_offsets, sorted_subvertices))


class Solution:

    def __init__(self, graph: TransitionGraph, initial_state: InitialState, nodes_count):
        self.subgraphs = graph.sorted_subgraphs
        self.initial_state = initial_state.x_0
        self.n = nodes_count

    def create_kinetic(self, trans_table):
        uni_weights = 1/(2*self.n)
        K = uni_weights * trans_table
        return K - scipy.sparse.diags(np.array(K.sum(axis=0))[0])

    def compute_scc_kernel(self, table, terminal_offsets, terminal_index):

        state_start = terminal_offsets[terminal_index]
        state_end = terminal_offsets[terminal_index + 1]

        scc_transition_table = table[state_start:
                                     state_end, state_start:state_end]

        nodes_count = scc_transition_table.shape[0]

        K = self.create_kinetic(scc_transition_table).todense()
        K = np.delete(K, 0, axis=0)

        kernel = np.empty((nodes_count))

        sign = -1 ** (nodes_count - 1)
        for i in range(nodes_count):
            minor = np.delete(K, i, axis=1)
            kernel[i] = sign * scipy.linalg.det(minor)
            sign *= -1

        return kernel / np.sum(kernel)

    def compute_column_nullspace(self, table, terminal_offsets):

        terminals_count = len(terminal_offsets) - 1
        terminal_start = terminal_offsets[0]

        data = np.empty((table.shape[0] - terminal_start))
        indices = []
        indptrs = [0] * (terminal_start + 1)

        for i in range(terminals_count):
            scc_size = terminal_offsets[i+1] - terminal_offsets[i]

            for _ in range(scc_size):
                indptrs.append(indptrs[-1] + 1)
            indices += [i] * scc_size
            if scc_size == 1:
                data[i] = 1
            else:
                data[i: scc_size] = self.compute_scc_kernel(
                    table, terminal_offsets, i)

        return scipy.sparse.csr_matrix((data, indices, indptrs), shape=(table.shape[0], terminals_count))

    def compute_row_nullspace(self, table, terminal_offsets, column_nullspace):

        terminal_start = terminal_offsets[0]

        U = column_nullspace[terminal_start:, :].transpose()
        U.data = np.array([1] * len(U.data))

        if terminal_start == 0:
            return U

        K = self.create_kinetic(table)
        N = K[:terminal_start, :][:, :terminal_start]
        B = K[terminal_start:, :][:, :terminal_start]

        A = -U @ B
        Bb = N

        scipy.sparse.linalg.use_solver(
            useUmfpack=True, assumeSortedIndices=False)
        X = scipy.sparse.linalg.spsolve(
            Bb.conj().T, A.conj().T,
            use_umfpack=True).conj().T

        return scipy.sparse.hstack((X, U))

    def compute_final_states(self):

        for subgraph_indices, subtable, terminal_offsets, sorted_subvertices in self.subgraphs:

            # just a single disconnected state -> probability of final state does not change
            if len(subgraph_indices) == 1:
                continue

            R = self.compute_column_nullspace(subtable, terminal_offsets)
            L = self.compute_row_nullspace(subtable, terminal_offsets, R)

            self.initial_state[subgraph_indices[sorted_subvertices]
                               ] = R @ L @ self.initial_state[subgraph_indices[sorted_subvertices]]

        return self.initial_state


# model = Model(join(dirname(__file__), "../data/toy.bnet"))
# table = TransitionTable(model)
# table.build_transition_table()
# graph = TransitionGraph(table)
# graph.sort()
# initial_state = InitialState(model, ['A','C', 'D'], [0, 0, 0])
# solution = Solution(graph, initial_state, len(model.model.keys()))
# x_star = solution.compute_final_states()

# model = Model(join(dirname(__file__), "../data/toy2.bnet"))
# table = TransitionTable(model)
# table.build_transition_table()
# initial_state = InitialState(model, ['A','B', 'C'], [0, 0, 0])
# graph = TransitionGraph(table, initial_state)
# graph.sort()
# solution = Solution(graph, initial_state, len(model.model.keys()))
# x_star = solution.compute_final_states()

# model = Model(join(dirname(__file__), "../data/toy3.bnet"))
# table = TransitionTable(model)
# table.build_transition_table()
# graph = TransitionGraph(table)
# graph.sort()
# initial_state = InitialState(model, ['A','B'], [0, 0])
# solution = Solution(graph, initial_state, len(model.model.keys()))
# x_star = solution.compute_final_states()

# model = Model(join(dirname(__file__), "../data/EMT_cohen_ModNet.bnet"))
# table = TransitionTable(model)
# table.build_transition_table()
# graph = TransitionGraph(table)
# graph.sort()
# initial_state = InitialState(model, ['ECMicroenv','DNAdamage','Metastasis','Migration','Invasion','EMT','Apoptosis','Notch_pthw','p53'], [1, 1, 0, 0, 0, 0, 0, 1, 0])
# solution = Solution(graph, initial_state, len(model.model.keys()))
# x_star = solution.compute_final_states()

# model = Model(join(dirname(__file__), "../data/mammalian_cc.bnet"))
# table = TransitionTable(model)
# table.build_transition_table()
# graph = TransitionGraph(table)
# graph.sort()
# initial_state = InitialState(model, ['CycE','CycA','CycB','Cdh1','Rb_b1','Rb_b2','p27_b1','p27_b2'], [0, 0, 0, 1, 1, 1, 1, 1])
# solution = Solution(graph, initial_state, len(model.model.keys()))
# x_star = solution.compute_final_states()

model = Model(join(dirname(__file__), "../data/krasmodel15vars.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(
    model, ['cc', 'KRAS', 'DSB', 'cell_death'], [1, 1, 1, 0])
graph = TransitionGraph(table, initial_state)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()


probs = np.zeros((len(x_star.nonzero()[0])))
states = []

model.nodes = list(model.model.keys())

for i, stateval in enumerate(x_star.nonzero()[0]):

    binstate = np.zeros((len(model.nodes)))
    c = len(model.nodes)-1
    t_stateval = stateval

    while t_stateval > 0:
        binstate[c] = t_stateval % 2
        t_stateval = t_stateval // 2
        c -= 1

    inds_states, = np.where(np.flip(binstate))

    if len(inds_states) > 0:
        t_state = [model.nodes[ind] for ind in inds_states]
        states.append(" -- ".join(t_state))

    else:
        states.append("<nil>")

    probs[i] = x_star[stateval]

last_states_probtraj = pd.DataFrame([probs], columns=states)
last_states_probtraj.sort_index(axis=1, inplace=True)

print(last_states_probtraj)
