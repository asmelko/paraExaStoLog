from exastolog import *
import glob

for f in glob.glob("data/database-small/*"):
    model = Model(f)
    table = TransitionTable(model)
    table.build_transition_table()
    initial_state = InitialState(model)
    graph = TransitionGraph(table, initial_state)
    graph.sort()
    solution = Solution(graph, initial_state, len(model.model.keys()))
    x_star = solution.compute_final_states()
    print(state_to_df(x_star, list(model.model.keys())))