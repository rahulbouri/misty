import pyvista as pv

# Load the .ply file
point_cloud = pv.read('/home/rahul/Desktop/misty_github/Examples/pointcloud.ply')

# Create a PyVista plotter and add the point cloud
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, point_size=5, render_points_as_spheres=True, color="blue")

# Customize the plot (optional)
plotter.set_background("white")
plotter.show()
# import numpy as np
# from plyfile import PlyData, PlyElement

# def read_ply(filename):
#     """ read XYZ point cloud from filename PLY file """
#     plydata = PlyData.read(filename)
#     x = np.asarray(plydata.elements[0].data['x'])
#     y = np.asarray(plydata.elements[0].data['y'])
#     z = np.asarray(plydata.elements[0].data['z'])
#     return np.stack([x,y,z], axis=1)

# coors =  read_ply("/home/rahul/Desktop/misty_github/Examples/pointCloudDeepLearning.ply")

# print(coors.shape)