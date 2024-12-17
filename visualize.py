import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

def animate_trajectory(csv_file):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Ensure all required columns are present
    required_columns = [
        "frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
        "x", "y", "z", "q_x", "q_y", "q_z", "q_w"
    ]

    if not all(column in data.columns for column in required_columns):
        raise ValueError("CSV file is missing required columns")

    # Extract position data
    x = data["x"]
    z = -1*data["y"]
    y = data["z"]

    # Extract quaternion data
    q_x = data["q_x"]
    q_y = data["q_y"]
    q_z = data["q_z"]
    q_w = data["q_w"]

    timestamps = data["timestamp"]

    # Setup the figure and axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Position Trajectory Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialize the plot elements
    trajectory, = ax.plot([], [], [], label="Trajectory", lw=2)
    point, = ax.plot([], [], [], 'ro', label="Current Position")
    arrow = None

    def init():
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_zlim([z.min(), z.max()])
        trajectory.set_data([], [])
        trajectory.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return trajectory, point

    def update(frame):
        nonlocal arrow
        if arrow:
            arrow.remove()  # Remove the previous arrow
        
        trajectory.set_data(x[:frame], y[:frame])
        trajectory.set_3d_properties(z[:frame])
        point.set_data(x[frame:frame+1], y[frame:frame+1])
        point.set_3d_properties(z[frame:frame+1])

        # Compute orientation arrow from quaternion
        rotation = R.from_quat([q_x[frame], q_y[frame], q_z[frame], q_w[frame]])
        direction = rotation.apply([-1, 0, 0])  # Arrow points in the X-axis direction in local frame
        direction2 = rotation.apply([0, 1, 0])  # Arrow points in the X-axis direction in local frame
        direction3 = rotation.apply([0, 0, 1])  # Arrow points in the X-axis direction in local frame
        scale = 0.2  # Scale factor to make the arrow smaller
        arrow = ax.quiver(
            x[frame], y[frame], z[frame],
            direction[0] * scale, direction[1] * scale, direction[2] * scale,
            color='blue', label="Orientation"
        )
        # arrow = ax.quiver(
        #     x[frame], y[frame], z[frame],
        #     direction2[0] * scale, direction2[1] * scale, direction2[2] * scale,
        #     color='red', label="Orientation"
        # )
        # arrow = ax.quiver(
        #     x[frame], y[frame], z[frame],
        #     direction3[0] * scale, direction3[1] * scale, direction3[2] * scale,
        #     color='pink', label="Orientation"
        # )

        return trajectory, point

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=False, interval=16)

    ax.legend()
    plt.show()

# Example usage
# Replace 'imu_data.csv' with the path to your CSV file
csv_file = 'data/camera_trajectory.csv'
animate_trajectory(csv_file)
