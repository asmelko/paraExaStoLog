from exastolog import *
from os.path import dirname, join

model = Model(join(dirname(__file__), "../data/mammalian_cc.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(model, ['CycE', 'CycA', 'CycB', 'Cdh1',
                             'Rb_b1', 'Rb_b2', 'p27_b1', 'p27_b2'], [0, 0, 0, 1, 1, 1, 1, 1])
graph = TransitionGraph(table, initial_state)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()
print(state_to_df(x_star, list(model.model.keys())))
