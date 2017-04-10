#! /usr/bin/env python
#################################################################################
#     File Name           :     ./util.py
#     Created By          :     yang
#     Creation Date       :     [2017-02-16 22:32]
#     Last Modified       :     [2017-03-18 00:13]
#     Description         :     util functions 
#################################################################################
import datetime

def remove_space(s):
    return "_".join(s.strip().split())

def get_date():
    current_time = datetime.datetime.now()
    return remove_space(str(current_time))
