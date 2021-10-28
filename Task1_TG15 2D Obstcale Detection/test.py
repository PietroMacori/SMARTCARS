# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# rectangle = [(0,0),(0,1),(1,1),(1,0)]
# x_values = [2, 3]

# y_values = [4, 5]


# fig, ax = plt.subplots()
# ax.plot(x_values, y_values, color='r')
# plt.show()
import pickle
file = open('myFile', 'rb')
new_d = pickle.load(file)
file.close()

#print dictionary object loaded from file
print(new_d[0][100])


# ax.add_patch(rectangle, fc='none', ec='orangered')

# automin, automax = plt.xlim()
# plt.xlim(automin-0.5, automax+0.5)
# automin, automax = plt.ylim()
# plt.ylim(automin-0.5, automax+0.5)
# plt.show()
# import numpy as np
# x = [1,22,2]
# y = [2,4,3]

# a = np.array([x, y]).T
# print(tuple(np.array([x[0],1])))
# print(a.shape)