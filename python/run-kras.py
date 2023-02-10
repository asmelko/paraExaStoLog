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
print(state_to_df(x_star, list(model.model.keys())))
