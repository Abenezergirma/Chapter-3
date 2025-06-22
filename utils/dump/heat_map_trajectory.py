import folium
import numpy as np
import pickle
import os
from branca.element import Element
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from folium import Icon, Marker

# Define trajectory endpoints
trajectory_names = {
    'Long': ([0, 0], [2083.4, 503.7]),
    'Infeasible': ([1041.7, 251.85], [2083.4, 503.7]),
    'Short': ([1041.7, 251.85], [946.2, 502.36]),
    'EM1': ([1041.7, 251.85], [260.425, -237.0375]),
    'EM2': ([1041.7, 251.85], [1822.975, 740.7375]),
}

folder_base = "new_voltages/voltage NODE"
voltage_data = {}
time_interval = 0.1
speed = 3

# Load voltage trajectories
for name in trajectory_names:
    file_path = os.path.join(folder_base, name, "saved_actual_voltage_trajectories.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list) and "voltage" in data[0]:
                voltage_data[name] = np.array(data[0]["voltage"])
            else:
                print(f"⚠️ Unexpected format in {file_path}")
    else:
        print(f"⚠️ File not found: {file_path}")

# Reference geolocation
initial_lat = 33 + 8/60 + 48/3600
initial_lon = -(96 + 48/60 + 22/3600)

def convert_to_latlon(base_lat, base_lon, dx, dy):
    meters_per_deg_lat = 111000
    meters_per_deg_lon = 111000 * abs(np.cos(np.radians(base_lat)))
    return base_lat + dy / meters_per_deg_lat, base_lon + dx / meters_per_deg_lon

# Marker helper
def add_custom_marker(latlon, label, icon_name, color):
    folium.Marker(
        location=latlon,
        tooltip=label,
        icon=Icon(icon=icon_name, prefix='fa', color=color)
    ).add_to(m)

# Initialize map
m = folium.Map(location=[initial_lat, initial_lon], zoom_start=13, tiles=None)

# Add Stadia tile layer
folium.TileLayer(
    tiles='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png?api_key=14c8fe0b-87de-4c52-a635-c8b1b8541114',
    attr='Map tiles by Stamen Design via Stadia Maps',
    name='Stamen Terrain',
    overlay=False,
    control=True
).add_to(m)

# Normalize voltage for color mapping
all_voltages = np.concatenate(list(voltage_data.values()))
vmin, vmax = np.min(all_voltages), np.max(all_voltages)
turbo_cmap = plt.get_cmap('turbo').reversed()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Plot each trajectory
for name, (start_xy, end_xy) in trajectory_names.items():
    if name not in voltage_data:
        continue

    voltage = voltage_data[name]
    n = len(voltage)
    direction = np.array(end_xy) - np.array(start_xy)
    unit_vector = direction / np.linalg.norm(direction)
    positions = [np.array(start_xy) + i * speed * time_interval * unit_vector for i in range(n)]
    latlon = [convert_to_latlon(initial_lat, initial_lon, x, y) for x, y in positions]



    # # Colored voltage segments
    for i in range(n - 1):
        color = mcolors.rgb2hex(turbo_cmap(norm(voltage[i])))
        folium.PolyLine(
            [latlon[i], latlon[i + 1]],
            color=color,
            weight=5,
            opacity=0.9
        ).add_to(m)



    start_latlon = convert_to_latlon(initial_lat, initial_lon, *start_xy)
    # Override end_latlon with the last point in actual voltage trajectory
    positions = [np.array(start_xy) + i * speed * time_interval * unit_vector for i in range(n)]
    latlon = [convert_to_latlon(initial_lat, initial_lon, x, y) for x, y in positions]
    end_latlon = latlon[-1]  # Ensures visual match with end of voltage line
        # # Black outline + label
    # if name == "Long" or name =="Infeasible":
    #     folium.PolyLine(
    #         latlon,
    #         color='black',
    #         weight=5,
    #         opacity=0.9,
    #         tooltip=f"Trajectory: {name}"
    #     ).add_to(m)


    # Long flight gets warehouse/residential and mid-flight incident marker
    if name == "Long":
        add_custom_marker(start_latlon, "Warehouse (Start)", "home", "darkblue")
        
        
        # Midpoint of Long trajectory = unexpected incident
        mid_xy = [(start_xy[0] + end_xy[0]) / 2, (start_xy[1] + end_xy[1]) / 2]
        mid_latlon = convert_to_latlon(initial_lat, initial_lon, *mid_xy)
        add_custom_marker(mid_latlon, "Unexpected Event Mid-Flight", "exclamation-triangle", "red")
    
    elif name == "Infeasible":
        add_custom_marker(end_latlon, "Residential (End)", "user", "cadetblue")

    # All other flights go to emergency landings
    else:
        # add_custom_marker(start_latlon, f"{name} (Start at Midpoint)", "plus-square", "orange")
        add_custom_marker(end_latlon, f"{name} Emergency Landing", "plus-square", "orange")



# Create vertical colorbar gradient
num_stops = 20
gradient_colors = [mcolors.rgb2hex(turbo_cmap(i / (num_stops - 1))) for i in range(num_stops)]
gradient_css = ', '.join([f'{c} {i * 100 // (num_stops - 1)}%' for i, c in enumerate(gradient_colors)])

vmin = vmin -2
vmax = vmax - 2
legend_html = f"""
<div style="
    position: absolute;
    top: 340px;
    right: 540px;
    z-index: 9999;
    background-color: white;
    padding: 10px 15px;
    border: 2px solid gray;
    font-size: 16px;
    font-family: 'CMU Serif', 'Georgia', serif;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    width: 320px;
">
    <b style="font-size: 17px;"> </b>
    <div style="
        margin-top: 10px;
        width: 100%;
        height: 18px;
        background: linear-gradient(to right, {gradient_css});
        border: 1px solid #aaa;
    "></div>
    <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: 5px;">
        <span>{vmin:.2f}</span>
        <span>{vmin + 0.25 * (vmax - vmin):.2f}</span>
        <span>{(vmin + vmax)/2:.2f}</span>
        <span>{vmin + 0.75 * (vmax - vmin):.2f}</span>
        <span>{vmax:.2f}</span>
    </div>
</div>
"""


m.get_root().html.add_child(Element(legend_html))

# Save map
m.save("map_with_Traj.html")
print("✅ Map saved as No Trajs.html")
