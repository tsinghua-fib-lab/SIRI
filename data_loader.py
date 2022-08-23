import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from pandera.typing import DataFrame
import numpy as np
from importlib.machinery import SourceFileLoader
import pandas as pd

path = './utils.py'
utils = SourceFileLoader('utils', path).load_module()

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_IDs: List[str],
                 o2d2flow: Dict,
                 oa2features: Dict,
                 oa2centroid: Dict,
                 dim_dests: int,
                 frac_true_dest: float, 
                 model: str,
                 historyflow: DataFrame, 
                 treatmentdict: Dict,
                 trainstate: int, 
                 d_set: List,
                 flow_df: DataFrame
                ) -> None:
        'Initialization'
        self.o2d2flow = o2d2flow
        self.oa2features = oa2features
        self.oa2centroid = oa2centroid
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest
        self.model = model
        self.historyflow = historyflow.copy(deep=True)
        self.d_set = d_set
        self.flow_df = flow_df.copy(deep=True)
        self.oa2pop, self.filtered_oa = self.popcal(self.flow_df, d_set)
        self.list_IDs = sorted(list(set(list_IDs) - set(self.filtered_oa)))
        np.random.shuffle(self.list_IDs)
        self.historyallo = self.allocal(self.historyflow, self.d_set)
        self.treatmentdict = treatmentdict

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def sampled_origin(self):
        return self.list_IDs
    
    def popcal(self, flow_df, d_set):
        def whether_in_d_set(m, d_set):
            if m in set(d_set):
                return 1
            else:
                return 0
        o_set = set(flow_df['workplace'])
        flow_df['workplace'] = flow_df['workplace'].map(lambda x: whether_in_d_set(x, d_set))
        flow_df = flow_df[flow_df['workplace'] == 1]
        outflow = flow_df[['residence', 'commuters']].groupby(['residence'], as_index=False).agg(sum)
        outflow.columns = ['residence', 'outflow']
        oa2pop = dict(zip(list(outflow['residence']), list(outflow['outflow'])))
        def filter(o_set, o1_set):
            return set(o_set) - set(o1_set)   #找出在d_set中outflow为0的点
        filtered_oa = filter(o_set, oa2pop.keys())     
        print(filtered_oa)
        return oa2pop, filtered_oa
    
    def allocal(self, historyflow, d_set):
        # 计算2019年各个pair的比例
        def whether_in_d_set(m, d_set):
            if m in set(d_set):
                return 1
            else:
                return 0
        historyflow['workplace'] = historyflow['workplace'].map(lambda x: whether_in_d_set(x, d_set))
        tmp = historyflow[historyflow['workplace'] == 1].copy(deep=True)
        tmp = tmp.groupby(['residence'], as_index=False).agg(sum)
        keys = list(tmp['residence'])
        values = list(tmp['commuters'])
        outflow_dict = dict(zip(keys, values))
        historyflow['outflow'] = historyflow['residence'].apply(lambda x: outflow_dict[x] if x in outflow_dict else 0)
        historyflow = historyflow[historyflow['outflow']!=0]
        historyflow['ratio'] = historyflow['commuters']/historyflow['outflow']
        keys = list(historyflow['odpair'])
        values = list(historyflow['ratio'])
        historyallo = dict(zip(keys, values))
        return historyallo
        
    def get_history(self, oa_origin, oa_destination):
        historyallo = self.historyallo
        key = str((str(oa_origin), str(oa_destination)))
        try:
            return historyallo[key]
        except KeyError:
            return 0

    def get_features(self, oa_origin, oa_destination):
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        treatmentdict = self.treatmentdict
        dist_od = utils.earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
        historyallo = self.get_history(oa_origin, oa_destination)  
        return [np.log(self.oa2pop[oa_origin])] + oa2features[oa_origin] + oa2features[oa_destination] + [historyallo] + [dist_od] 

    def get_treatment(self, oa_origin, oa_destination):
        oa2treatment = self.treatmentdict[oa_origin] + self.treatmentdict[oa_destination]
        return oa2treatment

    def get_flow(self, oa_origin, oa_destination):
        o2d2flow = self.o2d2flow
        try:
            return o2d2flow[oa_origin][oa_destination]
        except KeyError:
            return 0

    def get_X_T(self, origin_locs, dest_locs):
        X, Tr, T = [], [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            Tr += [[]]
            T += [[]]
            for j in dest_locs:
                X[-1] += [self.get_features(i, j)]
                Tr[-1] += [self.get_treatment(i, j)]
                T[-1] += [self.get_flow(i, j)]
        # 把Treatment和flow归一化了
        teX = torch.from_numpy(np.array(X)).float()
        teTr = torch.from_numpy(np.array(Tr)).float()
        teT = torch.from_numpy(np.array(T)).float()    #归一化了的flow
        return teX, teTr, teT

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Select sample (tile)
        sampled_origins = [self.list_IDs[index]]
        sampled_dests = self.d_set
        sampled_trX, sampled_tR, sampled_t = self.get_X_T(sampled_origins, sampled_dests)
        return sampled_trX, sampled_tR, sampled_t, sampled_origins

class VSR_Dataset(object):
    def __init__(self,
                 d_set: List,
                 flow_df: DataFrame,
                 oa2features: Dict,
                 od2flow: Dict, 
                 oa2centroid: Dict,
                 historyflow: DataFrame, 
                 treatment_dict: Dict
                ) -> None:
        super(VSR_Dataset, self).__init__()
        self.d_set = d_set
        self.flow_df = flow_df.copy(deep=True)
        self.oa2pop, self.filtered_oa = self.popcal(self.flow_df, self.d_set)
        self.oa2features = oa2features
        self.od2flow = od2flow
        self.oa2centroid = oa2centroid
        self.treatment_dict = treatment_dict
        self.odpair_list = self.get_odpairs()
        self.train_dataset = self.get_input(list(range(len(self.odpair_list))))

    def popcal(self, flow_df, d_set):
        def whether_in_d_set(m, d_set):
            if m in set(d_set):
                return 1
            else:
                return 0
        o_set = set(flow_df['workplace'])
        flow_df['workplace'] = flow_df['workplace'].map(lambda x: whether_in_d_set(x, d_set))
        flow_df = flow_df[flow_df['workplace'] == 1]
        outflow = flow_df[['residence', 'commuters']].groupby(['residence'], as_index=False).agg(sum)
        outflow.columns = ['residence', 'outflow']
        oa2pop = dict(zip(list(outflow['residence']), list(outflow['outflow'])))
        def filter(o_set, o1_set):
            return set(o_set) - set(o1_set)   #找出在d_set中outflow为0的点
        filtered_oa = filter(o_set, oa2pop.keys())     
        print(filtered_oa)
        return oa2pop, filtered_oa

    def get_treatment(self, oa_origin, oa_destination):
        oa2treatment = self.treatment_dict[oa_origin] + self.treatment_dict[oa_destination]
        return oa2treatment

    def get_X_T(self, odpairs):    
        T_ = []
        for odpair in odpairs:
            i = odpair[0]
            j = odpair[1]
            T=self.get_treatment(i, j)
            T_.append(T)
        T_ = np.array(T_)
        return T_

    def get_input(self, ids):
        sampled_ods = self.odpair_list[ids]
        sampled_trT = self.get_X_T(sampled_ods)
        return sampled_trT

    def get_odpairs(self):
        total_data = list(range(len(self.od2flow.keys())))
        np.random.shuffle(total_data)
        odpair_list = np.array(list(self.od2flow.keys()))[total_data]
        odpair_list = np.array([pair for pair in odpair_list if pair[0] in set(self.d_set) and pair[1] in set(self.d_set) and pair[0] not in self.filtered_oa])
        return odpair_list

    def get_data(self):
        return self.train_dataset, self.odpair_list
    
