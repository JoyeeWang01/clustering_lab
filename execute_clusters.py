from clusters import *

best_distance_for_countries()
best_k_for_countries()
compare_bisecting()
best_inter_distance_for_countries()

data_result = data_preparation("dataset.csv")
rownames1 = data_result[0]
rownames2 = data_result[1]
colnames = data_result[2]
data = data_result[3]
clust= hcluster(data, max_dis, euclidean)
cut_hierarchical(clust, 2)
heat_map(data, hierarchical, "hierarchical_heat_map.png")
result = bisectkcluster(data, euclidean, 6)
heat_map(data, result[0], "bisecting_heat_map.png")

output_result_and_label()
to_json()
