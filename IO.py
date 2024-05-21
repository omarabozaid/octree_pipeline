"""
    Class to interface Input Output of polydata and vtk objects
    author : Dr.-Ing. Omar ELSAYED
    Date: 20.04.2023
"""

import vtk

class input_output:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load_stl_as_vtk(file_path):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader.GetOutput()

    @staticmethod
    def write_octree_to_vtk(octree,output_file):
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        unstructuredGrid.SetPoints(octree.points)
        unstructuredGrid.SetCells(vtk.VTK_HEXAHEDRON, octree.cells)
        unstructuredGrid.GetPointData().AddArray(octree.proximityArray)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(unstructuredGrid)
        writer.SetDataModeToAscii()
        writer.Write()