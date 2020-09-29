
# GRU params
input_length = 250      # the size of each embedded tweet (the size of the input vector)
hidden_length = 300     # the size of the hidden vectors of the GRUs
output_dim_rumors = 3   # output size for rumor detection (True rumor, False rumor, Unverified)
output_dim_stances = 4  # output size for stance classification (Support, Deny, Query, Comment)

task_stances_no = 1
task_rumors_no = 2
