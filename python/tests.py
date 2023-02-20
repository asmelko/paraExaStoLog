from exastolog import *
from os.path import dirname, join


model = Model(join(dirname(__file__), "../data/toy.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(model, ['A', 'C', 'D'], [0, 0, 0])
graph = TransitionGraph(table, initial_state, model)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()
print(state_to_df(x_star, list(model.model.keys())))


model = Model(join(dirname(__file__), "../data/toy2.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(model, ['A', 'B', 'C'], [0, 0, 0])
graph = TransitionGraph(table, initial_state, model)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()
print(state_to_df(x_star, list(model.model.keys())))


model = Model(join(dirname(__file__), "../data/toy3.bnet"))
table = TransitionTable(model)
table.build_transition_table()
initial_state = InitialState(model, ['A', 'B'], [0, 0])
graph = TransitionGraph(table, initial_state, model)
graph.sort()
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()
print(state_to_df(x_star, list(model.model.keys())))
