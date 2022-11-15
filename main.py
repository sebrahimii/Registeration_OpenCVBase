#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:18:07 2022

@author: sami
"""
from register_cv2 import *

if __name__=="__main__":
    #load data 
    data = load('/home/sami/data/PROG/1-Code/-1-TORNGATS/DATA/Registerationdata_firstEXP/slab_registeration.h5')
    data = data['unreg']
    _regs, _tmats = [], []
    data_c = np.asarray([contrast_correction(img) for img in tqdm(data) ])
    for i in tqdm(range(len(data_c))):
        if i ==0 :
            _regs.append(data_c[0])
            _reg, _mat = register_cv2(data_c[0], data_c[0], 
                                    number_of_iter = 5000, term_eps = 1e-10,
                                    wmode='translation')
        else:
            
            _reg, _mat = register_cv2(_regs[i-1], data_c[i], 
                                    number_of_iter = 5000, term_eps = 1e-10,
                                    wmode='translation' )
            _regs.append(_reg)
        _tmats.append(_mat)

    _regs = np.asarray(_regs)
    _tmats = np.asarray(_tmats)
    _sub_mat, mov_mat =compare_tmats(_tmats, comp_with='previous', flatten = True, range_value=1)
    new_seq = register_selectedIndex(data,mov_mat,wmode ='euclidean' , 
                                     gr_base = False  ,cont_corr_base = False,
                                     n_iter= 5000, t_eps = 1e-10, 
                                     show_overlay= False)