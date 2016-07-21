# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/adaline.py
'''
### Helper functions for termination conditions

def Counter(max_iterations):
    i = max_iterations
    while True:
        yield i
        i -= 1

def max_iters(max_iterations):
    '''Assumes max_iterations is a Natural Integer.'''
    counter = Counter(max_iterations)
    return lambda: not next(counter)