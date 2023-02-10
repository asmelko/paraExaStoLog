from exastolog import *
from os.path import dirname, join

model = Model(join(dirname(__file__), "../data/EMT_cohen_ModNet.bnet"))
table = TransitionTable(model)
table.build_transition_table()                                                                                      # 18%
initial_state = InitialState(model, ['ECMicroenv', 'DNAdamage', 'Metastasis', 'Migration',                          
                             'Invasion', 'EMT', 'Apoptosis', 'Notch_pthw', 'p53'], [1, 1, 0, 0, 0, 0, 0, 1, 0])     #  8%
graph = TransitionGraph(table, initial_state)
graph.sort()                                                                                                        # 40%
solution = Solution(graph, initial_state, len(model.model.keys()))
x_star = solution.compute_final_states()                                                                            # 28%
print(state_to_df(x_star, list(model.model.keys())))
