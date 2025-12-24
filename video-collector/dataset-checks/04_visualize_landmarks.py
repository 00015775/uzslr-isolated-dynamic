"""
(READ-ONLY file)
This file loads one frame-XX.npy file and displays its 3D landmarks in an interactive Plotly plot.

- Verifies file exists and has shape (1662,).
- Splits landmarks: Face (468, green), Pose (33, red), Right Hand (21, blue), Left Hand (21, purple).
- Opens an interactive 3D scatter plot in your browser.
- No modifications to the original file.

Update 'npy_path' to visualize any frame.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio

# Will open on your local browser 
pio.renderers.default = "browser" 

npy_path = Path("/Users/macbookair/Downloads/rep-1/frame-05.npy")

if not npy_path.exists():
    print(f"Error: File not found: {npy_path}")
    print("Please update the 'npy_path' variable with the correct path.")
    exit()

try:
    data = np.load(npy_path)
    vec = data.copy() # will avoid accidental changes to original .npy files
    print(f"Loaded {npy_path.name} | Shape: {vec.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

if vec.shape != (1662,):
    print(f"Warning: Expected shape (1662,), got {vec.shape}")

# Split landmarks
face = vec[0:468*3].reshape((468, 3))
pose = vec[468*3:468*3 + 33*4].reshape((33, 4))[:, :3] # visibility column is disregarded
rh   = vec[468*3 + 33*4:468*3 + 33*4 + 21*3].reshape((21, 3))
lh   = vec[468*3 + 33*4 + 21*3:].reshape((21, 3))

fig = go.Figure()

fig.add_trace(go.Scatter3d(x=face[:,0], y=face[:,1], z=face[:,2],
                           mode='markers', marker=dict(size=2, color='green'), name='Face'))
fig.add_trace(go.Scatter3d(x=pose[:,0], y=pose[:,1], z=pose[:,2],
                           mode='markers', marker=dict(size=5, color='red'), name='Pose'))
fig.add_trace(go.Scatter3d(x=rh[:,0], y=rh[:,1], z=rh[:,2],
                           mode='markers', marker=dict(size=5, color='blue'), name='Right Hand'))
fig.add_trace(go.Scatter3d(x=lh[:,0], y=lh[:,1], z=lh[:,2],
                           mode='markers', marker=dict(size=5, color='purple'), name='Left Hand'))

fig.update_layout(
    title=f"3D Landmarks - {npy_path.name}",
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ),
    width=900,
    height=800
)

print("Opening in browser...")
fig.show()  # Will open on your local default browser


# REFERENCE: 
# Code adapted from https://stackoverflow.com/questions/69265059/is-it-possible-to-create-a-plotly-animated-3d-scatter-plot-of-mediapipes-body-p
