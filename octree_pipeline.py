"""
	Author : Dr.-Ing. Omar ELSAYED
	Date: 20.04.2023
 	This file is only for demonstration and educational uses
	To use it, you have to mention/cite the source properly with the name
	For more info write me : omarabozaid@aucegypt.edu
"""

import vtk
import numpy as np
import os
from rtree import index
import argparse
import json
import sys
import IO as IO 

class AdaptiveOctreeSTL:
	def __init__(self, stl_file, max_cell_size, min_cell_size):
		print("CREATING OCTREE AND PROXIMITY FIELD")
		self.stl_file = stl_file
		
		self.max_cell_size = max_cell_size
		self.min_cell_size = min_cell_size
		
		
		self.reader = vtk.vtkSTLReader()
		self.octree = vtk.vtkOctreePointLocator()
		self.cells = vtk.vtkCellArray()
		self.points = vtk.vtkPoints()
		self.unstructuredGrid = vtk.vtkUnstructuredGrid()
		self.proximityArray = vtk.vtkFloatArray()
		self.proximityArray.SetName("Proximity")
		
		self.pointIndexLookup = {}
		self.proximityLookup = {}
		
		self.p = index.Property()
		self.p.dimension = 3  
		self.range_tree = index.Index(properties=self.p)
			
	def read_and_prepare_stl(self):
		self.stl_polydata = IO.input_output.load_stl_as_vtk(self.stl_file)
		cell_index = 0
		polys = self.stl_polydata.GetPolys() 
		points = self.stl_polydata.GetPoints()  
		
		idList = vtk.vtkIdList()
		polys.InitTraversal()

		while polys.GetNextCell(idList):
			triangle_points =np.asarray( [points.GetPoint(idList.GetId(i)) for i in range(idList.GetNumberOfIds())])
			bbox = self.calculate_bbox(triangle_points)
			self.range_tree.insert(cell_index, bbox)
			cell_index += 1
		self.distanceFilter = vtk.vtkImplicitPolyDataDistance()
		self.distanceFilter.SetInput(self.stl_polydata)

	def calculate_bbox(self, points):
		min_x, max_x = min(points[:, 0]), max(points[:, 0])
		min_y, max_y = min(points[:, 1]), max(points[:, 1])
		min_z, max_z = min(points[:, 2]), max(points[:, 2])
		return (min_x, min_y, min_z, max_x, max_y, max_z)
	
	def get_or_insert_point_index(self, point):
		point_key = tuple(point)
		if point_key not in self.pointIndexLookup:
			point_id = self.points.InsertNextPoint(point)
			self.pointIndexLookup[point_key] = point_id
			proximity = self.distanceFilter.EvaluateFunction(point)
			self.proximityArray.InsertNextValue(proximity)
			self.proximityLookup[point_key] = proximity
		return self.pointIndexLookup[point_key]

	def add_hexahedron_cell(self, bounds):
		p = [
			[bounds[0], bounds[2], bounds[4]],
			[bounds[1], bounds[2], bounds[4]],
			[bounds[1], bounds[3], bounds[4]],
			[bounds[0], bounds[3], bounds[4]],
			[bounds[0], bounds[2], bounds[5]],
			[bounds[1], bounds[2], bounds[5]],
			[bounds[1], bounds[3], bounds[5]],
			[bounds[0], bounds[3], bounds[5]],
		]
		pointIds = [self.get_or_insert_point_index(point) for point in p]
		hexa = vtk.vtkHexahedron()
		for i, pid in enumerate(pointIds):
			hexa.GetPointIds().SetId(i, pid)
		self.cells.InsertNextCell(hexa)
	
	def should_divide_cell(self, bounds, size, divide_via_r_tree=True):
		if size <= self.min_cell_size:
			return False 
		
		corner_distances = []
		for x in [bounds[0], bounds[1]]:
			for y in [bounds[2], bounds[3]]:
				for z in [bounds[4], bounds[5]]:
					point_key = (x, y, z)
					if point_key in self.proximityLookup:
						proximity = self.proximityLookup[point_key]
					else:
						proximity = self.distanceFilter.EvaluateFunction([x, y, z])
						self.proximityLookup[point_key] = proximity
					corner_distances.append(proximity)
					
		if divide_via_r_tree:
			center = [
				0.5*(bounds[0]+bounds[1]),0.5*(bounds[2]+bounds[3]),0.5*(bounds[4]+bounds[5])
			]
			delta = bounds[1]-bounds[0]
			ordered_bounds = [
       							center[0]-delta , center[1]-delta , center[2]-delta,
              					center[0]+delta , center[1]+delta , center[2]+delta
                   			]
   
			candidates = list(self.range_tree.intersection(ordered_bounds))
			return len(candidates)>0
		
	def create_adaptive_octree(self, bounds, depth=0):
		cell_size = max([bounds[i+1] - bounds[i] for i in range(0, 6, 2)])
		if cell_size > self.max_cell_size or self.should_divide_cell(bounds, cell_size):
			new_bounds = self.calculate_subdivision_bounds(bounds)
			for sub_bounds in new_bounds:
				self.create_adaptive_octree(sub_bounds, depth+1)
		else:
			self.add_hexahedron_cell(bounds)

	def calculate_subdivision_bounds(self, bounds):
		x_mid = (bounds[0] + bounds[1]) / 2
		y_mid = (bounds[2] + bounds[3]) / 2
		z_mid = (bounds[4] + bounds[5]) / 2

		return [
			(bounds[0], x_mid, bounds[2], y_mid, bounds[4], z_mid),
			(x_mid, bounds[1], bounds[2], y_mid, bounds[4], z_mid),
			(bounds[0], x_mid, y_mid, bounds[3], bounds[4], z_mid),
			(x_mid, bounds[1], y_mid, bounds[3], bounds[4], z_mid),
			(bounds[0], x_mid, bounds[2], y_mid, z_mid, bounds[5]),
			(x_mid, bounds[1], bounds[2], y_mid, z_mid, bounds[5]),
			(bounds[0], x_mid, y_mid, bounds[3], z_mid, bounds[5]),
			(x_mid, bounds[1], y_mid, bounds[3], z_mid, bounds[5]),
		]

	def write_octree_to_vtk(self, output_file):
		IO.input_output.write_octree_to_vtk(self,output_file)

def main(config_file):
    # Read configuration from JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)

    stl_file = config.get("stl_file")
    max_cell_size = config.get("max_cell_size", 1.0)
    min_cell_size = config.get("min_cell_size", 0.001)
    coeff = config.get("coeff", 1.25)

    if not stl_file:
        print("Error: STL file name must be provided in the configuration.")
        sys.exit(1)

    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "base_adaptive_octree.vtu")

    converter = AdaptiveOctreeSTL(stl_file, max_cell_size, min_cell_size)
    converter.read_and_prepare_stl()

    bounds = converter.stl_polydata.GetBounds()
    max_bound = np.max([bd for bd in bounds])
    min_bound = np.min([bd for bd in bounds])

    if max_bound == 0:
        max_bound = -1 * min_bound

    if min_bound == 0:
        min_bound = -1 * max_bound

    new_bounds = [min_bound, max_bound, min_bound, max_bound, min_bound, max_bound]
    scaled_bounds = tuple(bd * coeff for bd in new_bounds)

    converter.create_adaptive_octree(scaled_bounds)
    converter.write_octree_to_vtk(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an STL file to create an adaptive octree.")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file")

    args = parser.parse_args()

    if not args.config_file:
        print("Error: Configuration file must be provided.")
        sys.exit(1)

    main(args.config_file)