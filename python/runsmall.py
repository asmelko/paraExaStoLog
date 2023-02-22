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
    p = state_to_human_readable(x_star, list(model.model.keys()))
    print(p)