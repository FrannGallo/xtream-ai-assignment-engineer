# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:40:00 2024

@author: fran_
"""

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    carat:  float 
    cut:    int
    color:  int
    clarity:    int 
    depth:  float 
    table:  float 
    x:  float
    y:  float
    z:  float

