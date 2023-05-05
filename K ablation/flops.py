# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:54:53 2023

@author: AmayaGS
"""

$ from fvcore.nn import FlopCountAnalysis
$ flops = FlopCountAnalysis(model, input)
$ flops.total()
274656
$ flops.by_operator()
Counter({'conv': 194616, 'addmm': 80040})
$ flops.by_module()
Counter({'': 274656, 'conv1': 48600,
         'conv2': 146016, 'fc1': 69120,
         'fc2': 10080, 'fc3': 840})
$ flops.by_module_and_operator()
{'': Counter({'conv': 194616, 'addmm': 80040}),
 'conv1': Counter({'conv': 48600}),
 'conv2': Counter({'conv': 146016}),
 'fc1': Counter({'addmm': 69120}),
 'fc2': Counter({'addmm': 10080}),
 'fc3': Counter({'addmm': 840})}