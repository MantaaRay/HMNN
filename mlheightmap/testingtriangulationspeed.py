from delaunay.delaunay import delaunay
from delaunay.quadedge.mesh import Mesh
from delaunay.quadedge.point import Vertex
from scipy.spatial import Delaunay
import numpy as np
import time
from random import uniform

def generate_random_points(num_points):
    return [Vertex(uniform(0, 4000), uniform(0, 4000)) for _ in range(num_points)], np.array([[uniform(0, 4000), uniform(0, 4000)] for _ in range(num_points)])

def perform_delaunay_triangulation_delaunay_module(points):
    N = len(points)
    m = Mesh()
    m.loadVertices(points)
    end = N - 1
    return delaunay(m, 0, end)

def perform_delaunay_triangulation_scipy(points):
    return Delaunay(points)

def test_delaunay_speed(num_points):
    points_delaunay_module, points_scipy = generate_random_points(num_points)

    start_time = time.time()
    # triangulation_delaunay_module = perform_delaunay_triangulation_delaunay_module(points_delaunay_module)
    end_time = time.time()
    elapsed_time_delaunay_module = end_time - start_time

    start_time = time.time()
    triangulation_scipy = perform_delaunay_triangulation_scipy(points_scipy)
    end_time = time.time()
    elapsed_time_scipy = end_time - start_time

    print(f"Time taken for Delaunay triangulation of {num_points} points using delaunay module: {elapsed_time_delaunay_module:.4f} seconds")
    print(f"Time taken for Delaunay triangulation of {num_points} points using SciPy: {elapsed_time_scipy:.4f} seconds")

# Example usage:
num_points = 1000000
test_delaunay_speed(num_points)
