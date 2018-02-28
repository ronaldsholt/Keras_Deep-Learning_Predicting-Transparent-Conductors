#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:09:08 2018

@author: RSH
"""


import warnings
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ga_cols = []
al_cols = []
o_cols = []
in_cols = []
for i in range(6):
    ga_cols.append("Ga_" + str(i))

for i in range(6):
    al_cols.append("Al_" + str(i))

for i in range(6):
    o_cols.append("O_" + str(i))

for i in range(6):
    in_cols.append("In_" + str(i))

ga_df = pd.DataFrame(columns=ga_cols)


def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float), x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)


def load_train():
    train_name = 'train_with_coords.csv'
    if os.path.exists(train_name):
        return pd.read_csv(train_name, index_col=0)
    train = pd.read_csv("../df_train.csv")

    ga_df = pd.DataFrame(columns=ga_cols)
    al_df = pd.DataFrame(columns=al_cols)
    o_df = pd.DataFrame(columns=o_cols)
    in_df = pd.DataFrame(columns=in_cols)
    for i in train.id.values:
        fn = "../Data/train/{}/geometry.xyz".format(i)
        train_xyz, train_lat = get_xyz_data(fn)

        ga_list = []
        al_list = []
        o_list = []
        in_list = []

        for li in train_xyz:
            try:
                if li[1] == "Ga":
                    ga_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "Al":
                    al_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "In":
                    in_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "O":
                    o_list.append(li[0])
            except:
                pass

        temp_ga = transform(ga_list, ga_cols, i)
        temp_al = transform(al_list, al_cols, i)
        temp_o = transform(o_list, o_cols, i)
        temp_in = transform(in_list, in_cols, i)

        ga_df = pd.concat([ga_df, temp_ga])
        al_df = pd.concat([al_df, temp_al])
        o_df = pd.concat([o_df, temp_o])
        in_df = pd.concat([in_df, temp_in])


    ga_df["id"] = ga_df.index
    al_df["id"] = al_df.index
    o_df["id"] = o_df.index
    in_df["id"] = in_df.index
    train = pd.merge(train, ga_df, on=["id"], how="left")
    train = pd.merge(train, al_df, on=["id"], how="left")
    train = pd.merge(train, o_df, on=["id"], how="left")
    train = pd.merge(train, in_df, on=["id"], how="left")
    train.to_csv(train_name)
    return train


def load_test():
    test_name = 'test_with_coords.csv'
    if os.path.exists(test_name):
        return pd.read_csv(test_name, index_col=0)

    test = pd.read_csv("../df_test.csv")
    ga_df = pd.DataFrame(columns=ga_cols)
    al_df = pd.DataFrame(columns=al_cols)
    o_df = pd.DataFrame(columns=o_cols)
    in_df = pd.DataFrame(columns=in_cols)
    for i in test.id.values:
        fn = "../Data/test/{}/geometry.xyz".format(i)
        train_xyz, train_lat = get_xyz_data(fn)

        ga_list = []
        al_list = []
        o_list = []
        in_list = []

        for li in train_xyz:
            try:
                if li[1] == "Ga":
                    ga_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "Al":
                    al_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "In":
                    in_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "O":
                    o_list.append(li[0])
            except:
                pass
        temp_ga = transform(ga_list, ga_cols, i)
        temp_al = transform(al_list, al_cols, i)
        temp_o = transform(o_list, o_cols, i)
        temp_in = transform(in_list, in_cols, i)
        ga_df = pd.concat([ga_df, temp_ga])
        al_df = pd.concat([al_df, temp_al])
        o_df = pd.concat([o_df, temp_o])
        in_df = pd.concat([in_df, temp_in])

    ga_df["id"] = ga_df.index
    al_df["id"] = al_df.index
    o_df["id"] = o_df.index
    in_df["id"] = in_df.index

    test = pd.merge(test, ga_df, on=["id"], how="left")
    test = pd.merge(test, al_df, on=["id"], how="left")
    test = pd.merge(test, o_df, on=["id"], how="left")
    test = pd.merge(test, in_df, on=["id"], how="left")
    test.to_csv(test_name)

    return test


def transform(input_list, cols, index_n):
    try:
        model = PCA(n_components=2)
        temp_ga = model.fit_transform(np.array(input_list).transpose())
        tmp = [item for sublist in temp_ga for item in sublist]
    except:
        tmp = [0, 0, 0, 0, 0, 0]
    tmp = pd.DataFrame(tmp).transpose()
    tmp.columns = cols
    tmp.index = np.array([index_n])
    return tmp
