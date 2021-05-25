#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:35:37 2021

@author: jackchen
"""

import seaborn as sns
import pandas as pd

penguins = sns.load_dataset("penguins")
penguins.head()

# sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")

