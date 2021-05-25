#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:52:39 2021

@author: jackchen
"""

from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

output_file("error.html")

groups= ['A', 'B', 'C', 'D']
counts = [5, 3, 4, 2]
error = [0.8, 0.4, 0.4, 0.3]
upper = [x+e for x,e in zip(counts, error) ]
lower = [x-e for x,e in zip(counts, error) ]

source = ColumnDataSource(data=dict(groups=groups, counts=counts, upper=upper, lower=lower))

p = figure(x_range=groups, plot_height=350, toolbar_location=None, title="Values", y_range=(0,7))
p.vbar(x='groups', top='counts', width=0.9, source=source, legend="groups",
       line_color='white', fill_color=factor_cmap('groups', palette=["#962980","#295f96","#29966c","#968529"],
                                                  factors=groups))

p.add_layout(
    Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay")
)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"

show(p)
