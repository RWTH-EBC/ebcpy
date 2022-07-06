```python
"""
Goals of this part of the examples:
1. Learn how to create a custom `Optimizer` class
2. Learn the different optimizer frameworks
3. Learn the usage of `StatisticsAnalyzer`
4. Understand the motivation behing `AixCaliBuHA`
"""
```
 Start by importing all relevant packages
```python
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmetrics
```
 Imports from ebcpy
```python
from ebcpy.optimization import Optimizer
from ebcpy.utils.statistics_analyzer import StatisticsAnalyzer


statistical_measure = "MAE"
with_plot = True


```
 ######################### Class definition ##########################
 To create a custom optimizer, one needs to inherit from the Optimizer
```python
class PolynomalFitOptimizer(Optimizer):
    
```
