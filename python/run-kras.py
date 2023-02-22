from exastolog import *
from os.path import dirname, join

model = Model(join(dirname(__file__), "../data/krasmodel15vars.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(
    model, ['cc', 'KRAS', 'DSB', 'cell_death'], [1, 1, 1, 0])
graph = TransitionGraph(table, initial_state)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()
p = state_to_human_readable(x_star, list(model.model.keys()))
print(p)
