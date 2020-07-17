from graph_classification_kernels.rw_kernel import RWKernel
from graph_classification_kernels.sp_kernel import SPKernel
from graph_classification_kernels.wl_kernel import WLKernel

from kegg.pathway import Pathway
import random
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


suffixes = ['00010', '00020', '00030', '00040', '00051',
            '00052',          '00061', '00071', '00130',
                     '00220', '00230', '00240', '00250',
            '00260', '00270', '00280',          '00310']

human_prefix = 'hsa'
ecoli_prefix = 'eco'

url_base = "http://rest.kegg.jp/get/{}/kgml"

human_pathways = [Pathway.from_url(url_base.format(human_prefix + suffix)) for suffix in suffixes]
ecoli_pathways = [Pathway.from_url(url_base.format(ecoli_prefix + suffix)) for suffix in suffixes]

for pathway in human_pathways:
    pathway.re_root(["name"])

for pathway in ecoli_pathways:
    pathway.re_root(["name"])

human_ind = range(0, len(human_pathways))
ecoli_ind = range(0, len(ecoli_pathways))
rw_no_smoothing = []
rw_smoothing = []
sp_no_smoothing = []
sp_smoothing = []
wl_no_smoothing = []
wl_smoothing = []

no_walks = 100
wl_loops = 5
i = 0
for ind in human_ind:
    this_rw_no_smoothing = []
    this_rw_smoothing = []
    this_sp_no_smoothing = []
    this_sp_smoothing = []
    this_wl_no_smoothing = []
    this_wl_smoothing = []
    for ind_2 in human_ind[i:]:
        print(ind, ind_2)
        path_1 = human_pathways[ind].root_graph
        path_2 = human_pathways[ind_2].root_graph

        rw_kernel = RWKernel(path_1, path_2)
        rw_kernel.populate_samples(no_walks)
        q_1 = rw_kernel.q_g_1
        q_2 = rw_kernel.q_g_2
        this_rw_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        rw_kernel.set_up_neural_network()
        rw_kernel.adjust_network(4)
        this_rw_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), rw_kernel.M),
                q_2
            ))
        )

        sp_kernel = SPKernel(path_1, path_2)
        sp_kernel.populate_samples()
        q_1 = sp_kernel.q_g_1
        q_2 = sp_kernel.q_g_2
        this_sp_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        sp_kernel.set_up_neural_network()
        sp_kernel.adjust_network(1)
        this_sp_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), sp_kernel.M),
                q_2
            ))
        )

        wl_kernel = WLKernel(path_1, path_2)
        wl_kernel.loop(wl_loops)
        q_1 = wl_kernel.q_g_1
        q_2 = wl_kernel.q_g_2
        this_wl_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        wl_kernel.set_up_neural_network()
        wl_kernel.adjust_network(2)
        this_wl_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), wl_kernel.M),
                q_2
            ))
        )

    for ind_2 in ecoli_ind:
        print(ind, ind_2)
        path_1 = human_pathways[ind].root_graph
        path_2 = ecoli_pathways[ind_2].root_graph
        rw_kernel = RWKernel(path_1, path_2)
        rw_kernel.populate_samples(no_walks)
        q_1 = rw_kernel.q_g_1
        q_2 = rw_kernel.q_g_2

        this_rw_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        rw_kernel.set_up_neural_network()
        rw_kernel.adjust_network(4)
        this_rw_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), rw_kernel.M),
                q_2
            ))
        )

        sp_kernel = SPKernel(path_1, path_2)
        sp_kernel.populate_samples()
        q_1 = sp_kernel.q_g_1
        q_2 = sp_kernel.q_g_2
        this_sp_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        sp_kernel.set_up_neural_network()
        sp_kernel.adjust_network(1)
        this_sp_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), sp_kernel.M),
                q_2
            ))
        )

        wl_kernel = WLKernel(path_1, path_2)
        wl_kernel.loop(wl_loops)
        q_1 = wl_kernel.q_g_1
        q_2 = wl_kernel.q_g_2
        this_wl_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        wl_kernel.set_up_neural_network()
        wl_kernel.adjust_network(2)
        this_wl_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), wl_kernel.M),
                q_2
            ))
        )

    rw_no_smoothing.append([0.0] * i + this_rw_no_smoothing)
    rw_smoothing.append([0.0] * i + this_rw_smoothing)
    sp_no_smoothing.append([0.0] * i + this_sp_no_smoothing)
    sp_smoothing.append([0.0] * i + this_sp_smoothing)
    wl_no_smoothing.append([0.0] * i + this_wl_no_smoothing)
    wl_smoothing.append([0.0] * i + this_wl_smoothing)
    i += 1

for ind in ecoli_ind:
    this_rw_no_smoothing = []
    this_rw_smoothing = []
    this_sp_no_smoothing = []
    this_sp_smoothing = []
    this_wl_no_smoothing = []
    this_wl_smoothing = []
    for ind_2 in ecoli_ind[max(i-17, 0):]:
        print(ind, ind_2)
        path_1 = ecoli_pathways[ind].root_graph
        path_2 = ecoli_pathways[ind_2].root_graph

        rw_kernel = RWKernel(path_1, path_2)
        rw_kernel.populate_samples(no_walks)
        q_1 = rw_kernel.q_g_1
        q_2 = rw_kernel.q_g_2
        this_rw_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        rw_kernel.set_up_neural_network()
        rw_kernel.adjust_network(4)
        this_rw_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), rw_kernel.M),
                q_2
            ))
        )

        sp_kernel = SPKernel(path_1, path_2)
        sp_kernel.populate_samples()
        q_1 = sp_kernel.q_g_1
        q_2 = sp_kernel.q_g_2
        this_sp_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        sp_kernel.set_up_neural_network()
        sp_kernel.adjust_network(1)
        this_sp_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), sp_kernel.M),
                q_2
            ))
        )

        wl_kernel = WLKernel(path_1, path_2)
        wl_kernel.loop(wl_loops)
        q_1 = wl_kernel.q_g_1
        q_2 = wl_kernel.q_g_2
        this_wl_no_smoothing.append(float(np.dot(np.transpose(q_1), q_2)))
        wl_kernel.set_up_neural_network()
        wl_kernel.adjust_network(2)
        this_wl_smoothing.append(
            float(np.dot(
                np.dot(np.transpose(q_1), wl_kernel.M),
                q_2
            ))
        )

    rw_no_smoothing.append([0.0] * i + this_rw_no_smoothing)
    rw_smoothing.append([0.0] * i + this_rw_smoothing)
    sp_no_smoothing.append([0.0] * i + this_sp_no_smoothing)
    sp_smoothing.append([0.0] * i + this_sp_smoothing)
    wl_no_smoothing.append([0.0] * i + this_wl_no_smoothing)
    wl_smoothing.append([0.0] * i + this_wl_smoothing)
    i += 1

names = [human_prefix + suffixes[i] for i in human_ind] + [ecoli_prefix + suffixes[i] for i in ecoli_ind]

rw_no_smoothing_arr = np.array(rw_no_smoothing)
rw_smoothing_arr = np.array(rw_smoothing)
sp_no_smoothing_arr = np.array(sp_no_smoothing)
sp_smoothing_arr = np.array(sp_smoothing)
wl_no_smoothing_arr = np.array(wl_no_smoothing)
wl_smoothing_arr = np.array(wl_smoothing)

i_lower = np.tril_indices(rw_no_smoothing_arr.shape[0], -1)

rw_no_smoothing_arr[i_lower] = rw_no_smoothing_arr.T[i_lower]
rw_smoothing_arr[i_lower] = rw_smoothing_arr.T[i_lower]

sp_no_smoothing_arr[i_lower] = sp_no_smoothing_arr.T[i_lower]
sp_smoothing_arr[i_lower] = sp_smoothing_arr.T[i_lower]

wl_no_smoothing_arr[i_lower] = wl_no_smoothing_arr.T[i_lower]
wl_smoothing_arr[i_lower] = wl_smoothing_arr.T[i_lower]

rw_df_no_smoothing = pd.DataFrame(rw_no_smoothing_arr, index=names, columns=names)
rw_df_smoothing = pd.DataFrame(rw_smoothing_arr, index=names, columns=names)
sp_df_no_smoothing = pd.DataFrame(sp_no_smoothing_arr, index=names, columns=names)
sp_df_smoothing = pd.DataFrame(sp_smoothing_arr, index=names, columns=names)
wl_df_no_smoothing = pd.DataFrame(wl_no_smoothing_arr, index=names, columns=names)
wl_df_smoothing = pd.DataFrame(wl_smoothing_arr, index=names, columns=names)

print(rw_df_no_smoothing)
print(rw_df_smoothing)

print(sp_df_no_smoothing)
print(sp_df_smoothing)

print(wl_df_no_smoothing)
print(wl_df_smoothing)

rw_df_no_smoothing.to_csv( "try_2_rw_no_smoothing.csv")
rw_df_smoothing.to_csv(    "try_2_rw_smoothing.csv")
sp_df_no_smoothing.to_csv( "try_2_sp_no_smoothing.csv")
sp_df_smoothing.to_csv(    "try_2_sp_smoothing.csv")
wl_df_no_smoothing.to_csv( "try_2_wl_no_smoothing.csv")
wl_df_smoothing.to_csv(    "try_2_wl_smoothing.csv")
