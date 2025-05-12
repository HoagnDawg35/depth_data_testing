import h5py
import numpy as np

# #Open the H5 file in read mode
# # with h5py.File('H:\AIC25\data\Depth_map\Camera_0006.h5', 'r') as file:
#     # print(&quot;Keys: %s&quot; % file.keys())
#     a_group_key = list(file.keys())[0]
    
#     # Getting the data
#     data = list(file[a_group_key])
#     # print(data)

# # Saving data    
# np.save("data.npy", data, allow_pickle=True)

# Load the .npy file
data = np.load('H:\AIC25\data_output\data.npy')

min_value = data[0][0]
max_value = data[0][0]
vt_min = (0, 0)
vt_max = (0, 0)

for i in range(-1, len(data)):
    for j in range(-1, len(data)):    
        if data[i][j].all() < min_value:
            min_value = data[i][j]
            vt_min = (i, j)
        elif data[i][j].all() > max_value:
            max_value = data[i][j]
            vt_max = (i, j)
            
print('GT min cua depth:', min_value, " tai diem:", vt_min)
print('GT max cua depth:', max_value, " tai diem:", vt_max)
print("Kieu data:", data.dtype)