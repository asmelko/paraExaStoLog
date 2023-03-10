import numpy as np
import sortednp
import pyeda.inter
import scipy

# sympy import takes forever
# but for some cases, to_dnf method performs better than the one of pyeda
# this flag therefore enables sympy use instead of pyeda
USE_SYMPY = False


class Model:

    def __init__(self, bnet_filename):
        self.model = self.readBNet(bnet_filename)

    def readBNet(self, filename):
        model = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            del lines[0]
            for line in lines:
                species, formula = [value.strip() for value in line.split(",")]
                model.update({species: formula})

        return model


class TransitionTable:

    def __init__(self, model: Model):
        self.model = model.model

    def to_dnf(self, formula_str):

        if not USE_SYMPY:
            formula = pyeda.inter.expr(
                formula_str.replace('!', '~')).simplify()
            formula = formula.to_dnf()
            _, _, clauses_tmp = formula.encode_dnf()

            clauses = []
            for clause in clauses_tmp:
                clause_map = {}
                for arg in clause:
                    clause_map.update(
                        {str(formula.inputs[abs(arg) - 1]): True if arg > 0 else False})

                clauses.append(clause_map)

            return clauses
        else:
            import sympy

            formula = sympy.parsing.sympy_parser.parse_expr(
                formula_str.replace('!', '~')).simplify()

            dnf = sympy.logic.boolalg.to_dnf(formula)

            if not isinstance(dnf, sympy.logic.boolalg.Or):
                clauses_tmp = dnf,
            else:
                clauses_tmp = dnf.args

            clauses = []
            for clause in clauses_tmp:
                if not isinstance(clause, sympy.logic.boolalg.And):
                    args = clause,
                else:
                    args = clause.args

                clause_map = {}
                for arg in args:
                    clause_map.update(
                        {str(arg.free_symbols.pop()): not isinstance(arg, sympy.logic.boolalg.Not)})

                clauses.append(clause_map)

            return clauses

    def get_node_transitions(self, node_idx, nodes, node_values):
        def get_transitions(formula, up_down):
            node_transitions = [np.empty((0), dtype=int)]

            for clause in formula:
                # no change
                if nodes[node_idx] in clause.keys() and clause[nodes[node_idx]] == up_down:
                    continue

                contributing_nodes_idxs = []
                contributing_nodes_vals = []
                for arg in clause.keys():
                    contributing_nodes_idxs.append(nodes.index(arg))
                    contributing_nodes_vals.append(clause[arg])

                if nodes[node_idx] not in clause.keys():
                    contributing_nodes_idxs.append(node_idx)
                    contributing_nodes_vals.append(not up_down)

                static_val = np.sum(
                    node_values[contributing_nodes_idxs] * contributing_nodes_vals)

                not_contributing_count = len(
                    nodes) - len(contributing_nodes_idxs)
                clause_transitions = np.empty(
                    (2 ** not_contributing_count), dtype=int)
                clause_transitions[0] = static_val

                states_len = 1
                for not_contributing_idx in range(len(nodes)):
                    if not_contributing_idx in contributing_nodes_idxs:
                        continue

                    clause_transitions[states_len: 2 *
                                       states_len] = clause_transitions[: states_len] + 2 ** not_contributing_idx

                    states_len *= 2

                node_transitions.append(clause_transitions)

            node_transitions = sortednp.kway_merge(
                *node_transitions, duplicates=sortednp.DROP)

            return node_transitions

        formula = self.model[nodes[node_idx]]

        dnf_clauses = self.to_dnf(formula)
        up_transitions_src = get_transitions(dnf_clauses, True)

        dnf_clauses = self.to_dnf(f"!({formula})")
        down_transitions_src = get_transitions(dnf_clauses, False)

        up_trans_src_len = up_transitions_src.shape[0]
        transitions_src = np.concatenate(
            [up_transitions_src, down_transitions_src])

        transitions_dst = np.empty((transitions_src.shape[0]))
        transitions_dst[:up_trans_src_len] = transitions_src[:
                                                             up_trans_src_len] + 2 ** node_idx
        transitions_dst[up_trans_src_len:] = transitions_src[up_trans_src_len:] - 2 ** node_idx

        return transitions_src, transitions_dst

    def build_transition_table(self):
        nodes = list(self.model.keys())
        n = len(nodes)
        states_count = 2 ** n

        node_values = 2 ** np.arange(n, dtype=int)

        transitions_src = []
        transitions_dst = []

        for node_idx in range(n):
            node_tr_src, node_tr_dst = self.get_node_transitions(
                node_idx, nodes, node_values)

            transitions_src.append(node_tr_src)
            transitions_dst.append(node_tr_dst)

        transitions_src = np.concatenate(transitions_src)
        transitions_dst = np.concatenate(transitions_dst)

        self.transition_table = scipy.sparse.csr_matrix(
            (
                np.ones((len(transitions_dst))),
                (transitions_dst,
                 transitions_src)
            ),
            shape=(states_count, states_count)
        )


class InitialState:

    def __init__(self, model: Model, fixed_nodes=[], fixed_values=[], fixed_probability=1):
        nodes = list(model.model.keys())
        n = len(nodes)
        states_count = 2 ** n
        states = np.empty((states_count, n), dtype=bool)
        for i in range(n):
            a = np.full((2 ** (i + 1)), True)
            a[:2 ** i] = False

            states[:, i] = np.tile(a, (2 ** (n - i - 1)))

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
        orig_table = table
        ordering = np.empty((table.shape[0]), dtype=int)
        mask = np.full((table.shape[0]), False)
        terminals_idx = None
        fill_idx = table.shape[0]
        while table.shape[0] != 0:
            new_mask = np.array(table.sum(axis=0) == 0).flatten()

            indices = np.argwhere(new_mask ^ mask).flatten()
            fill_idx -= indices.shape[0]
            ordering[fill_idx: fill_idx + indices.shape[0]] = indices

            mask = new_mask
            table = orig_table[~mask, :]

            if terminals_idx is None:
                terminals_idx = fill_idx

        return ordering, terminals_idx

    def create_metagraph(self, table):
        count, labels = scipy.sparse.csgraph.connected_components(
            table, directed=True, connection='strong')

        # All SCCs are single vertices => metagraph = graph
        if count == table.shape[0]:
            sccomponents = np.arange(count, dtype=int).reshape((count, 1))
            return sccomponents, table

        sccomponents = np.empty(shape=(count), dtype=object)

        sort_idx = np.argsort(labels)
        labels_sorted = labels[sort_idx]
        unq_first = np.concatenate(
            ([True], labels_sorted[1:] != labels_sorted[:-1]))
        unq_count = np.diff(np.nonzero(unq_first)[0])
        sccomponents = np.array(
            np.split(sort_idx, np.cumsum(unq_count)), dtype=object)

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

        sorted_vertices = np.concatenate(sccs[sorted_metavertices])

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
                data[i: i + scc_size] = self.compute_scc_kernel(
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

        X = -U @ B

        scipy.sparse.linalg.use_solver(
            useUmfpack=True, assumeSortedIndices=False)
        X = scipy.sparse.linalg.spsolve(N.conj().transpose(
        ), X.conj().transpose(), use_umfpack=True).conj().transpose()

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


def state_to_human_readable(state, node_names):
    nonzero_indices = state.nonzero()[0]
    probabilities = np.empty((nonzero_indices.shape[0], 2), dtype=object)
    probabilities[:, 0] = state[nonzero_indices]

    for i, idx in enumerate(nonzero_indices):

        present_nodes = []

        node_i = 0
        while idx != 0:
            if idx % 2:
                present_nodes.append(node_i)
            idx = idx >> 1
            node_i += 1

        if len(present_nodes) > 0:
            nodes = [node_names[ind] for ind in present_nodes]
            probabilities[i, 1] = " -- ".join(nodes)

        else:
            probabilities[i, 1] = "<nil>"

    return probabilities
