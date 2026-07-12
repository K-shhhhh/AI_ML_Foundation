# plotting a line graph using .show on a python script to see the difference between jupyter notebooks 

import matplotlib.pyplot as plt

X = [1,2,3,4]
Y = [5,6,7,8]

plt.plot(X,Y) # like in jupyter notebooks, only this function will not plot the graph, it will just make it 

plt.show() # in python scripts we need to use .show to actually plot the graph, also unlike in jupyter notebooks when we hover over the graph
           # that was plotted via this method, we will see coordinates of each point on the bottom right unlike in jupyter notebooks
           # also we can zoom into a specific part of the graph
           # we can save that graph as an .png image 

# therefore, the plt.show() is very useful