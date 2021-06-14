'''
A set of 14 vectors was provided to us representing the keypoints on the car.
In this system, x axis was along the width, y-axis was along the length and z axis was along the height.
This takes the co-ordinates as input and tells the avg width, length and height for the same.
'''

# 14x3 matrix representing (x,y,z) of the keypoints
# 2.5447 -3.7577 -1.5125
# -3.0188 -3.8300 -1.5128
# 2.2950 3.4357 -1.2746
# -2.8187 3.3599 -1.2430
# 1.8002 -5.7926 -0.2371
# -2.2685 -5.8769 -0.2381
# 1.5809 5.3567 0.2227
# -2.1554 5.3089 0.2352
# 2.6917 -1.6341 0.8413
# -3.1261 -1.7055 0.8464
# 1.6331 -0.9118 1.9708
# -2.0508 -0.9471 1.9701
# 1.4829 2.6154 2.0137
# -1.9433 2.6160 2.0160

x = [2.5447, -3.0188, 2.2950, -2.8187, 1.8002, -2.2685, 1.5809, -2.1554, 2.6917, -3.1261, 1.6331, -2.0508, 1.4829, -1.9433]
max_x = max(x)
min_x = min(x)
x_length = max_x-min_x
print("\nMin coordinate along x is "+str(min_x))
print("Max coordinate along x is "+str(max_x))
print("Length along x is "+str(x_length))

y = [-3.7577, -3.8300, 3.4357, 3.3599, -5.7926, -5.8769, 5.3567, 5.3089, -1.6341, -1.7055, -0.9118, -0.9471, 2.6154, 2.6160]
max_y = max(y)
min_y = min(y)
y_length = max_y-min_y
print("\nMin coordinate along y is "+str(min_y))
print("Max coordinate along y is "+str(max_y))
print("Length along y is "+str(y_length))

z = [-1.5125, -1.5128, -1.2746, -1.2430, -0.2371, -0.2381, 0.2227, 0.2352, 0.8413, 0.8464, 1.9708, 1.9701, 2.0137, 2.0160]
max_z = max(z)
min_z = min(z)
z_length = max_z-min_z
print("\nMin coordinate along z is "+str(min_z))
print("Max coordinate along z is "+str(max_z))
print("Length along z is "+str(z_length))
print()

avg_width = 1.6362
avg_length = 3.8600
avg_height = 1.5208

x_scaling_factor = avg_width/x_length
y_scaling_factor = avg_length/y_length
z_scaling_factor = avg_height/z_length

print("Scaling factors are", x_scaling_factor, y_scaling_factor, z_scaling_factor, '\n')
