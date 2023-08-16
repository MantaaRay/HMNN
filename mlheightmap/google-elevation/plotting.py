from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import geopandas as gpd

class ElevationPlotter:
    def __init__(self, gdf_land, random_points_on_land, center_lat, center_lon):
        self.gdf_land = gdf_land
        self.random_points_on_land = random_points_on_land
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoomed_in = False
        self.fig, self.ax = plt.subplots()
        self.plot_land_and_points()
        self.create_controls()

    def plot_land_and_points(self):
        self.ax.clear()
        self.gdf_land.plot(ax=self.ax, color='lightgrey')
        gdf_points = gpd.GeoDataFrame(geometry=self.random_points_on_land)
        gdf_points.plot(ax=self.ax, marker='o', color='red', markersize=5)
        if self.zoomed_in:
            self.ax.set_xlim(self.center_lon - self.zoom_radius.val, self.center_lon + self.zoom_radius.val)
            self.ax.set_ylim(self.center_lat - self.zoom_radius.val, self.center_lat + self.zoom_radius.val)
        plt.draw()

    def toggle_view(self, event):
        self.zoomed_in = not self.zoomed_in
        self.update_view()

    def update_radius(self, val):
        self.update_view()

    def update_view(self):
        self.plot_land_and_points()

    def create_controls(self):
        button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        button = Button(button_ax, 'Toggle View')
        button.on_clicked(self.toggle_view)

        slider_ax = plt.axes([0.2, 0.05, 0.5, 0.03])
        self.zoom_radius = Slider(slider_ax, 'Zoom Radius', 0.1, 10, valinit=1)
        self.zoom_radius.on_changed(self.update_radius)

        plt.show()


