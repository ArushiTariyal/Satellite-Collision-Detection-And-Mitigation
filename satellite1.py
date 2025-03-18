# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import plotly.graph_objects as go  # For creating interactive 3D plots
import streamlit as st  # For building the web application interface
from skyfield.api import load  # For astronomical calculations and satellite data
from scipy.spatial.distance import euclidean  # For calculating Euclidean distance
import requests  # For making HTTP requests to fetch data
import time  # For handling time-related operations

# Step 1: Fetch Real-Time TLE Data
def fetch_tle():
    # URL to fetch the Two-Line Element (TLE) data for active satellites
    url = 'https://celestrak.org/NORAD/elements/stations.txt'
    # Send a GET request to the URL to fetch the data
    response = requests.get(url)
    # Write the fetched TLE data to a local file
    with open('active_satellites.txt', 'w') as file:
        file.write(response.text)
    # Display a success message in the Streamlit app
    st.success("TLE Data Fetched Successfully")

# Step 2: Convert TLE to Cartesian Coordinates
def read_tle(file_path):
    # Load the TLE data from the file using Skyfield
    satellites = load.tle_file(file_path)
    # Create a timescale object for time-related calculations
    ts = load.timescale()
    # Initialize an empty list to store satellite data
    data = []
    # Loop through each satellite in the TLE data
    for sat in satellites:
        # Get the current time
        t = ts.now()
        # Calculate the satellite's position in kilometers
        position = sat.at(t).position.km
        # Calculate the satellite's velocity in kilometers per second
        velocity = sat.at(t).velocity.km_per_s

        # Check for zero or invalid positions/velocities
        if np.all(position == 0) or np.all(velocity == 0):
            st.warning(f"Invalid data for satellite {sat.name}. Skipping.")
            continue  # Skip this satellite

        # Append the satellite's data to the list
        data.append(
            [sat.name, t.utc_iso(), position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]])
    # Convert the list of satellite data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['name', 'timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    # Return the DataFrame and the list of satellite objects
    return df, satellites

# Step 3: Collision Detection using Improved Algorithm
def detect_collisions(data, threshold):
    # Initialize an empty list to store collision data
    collisions = []
    # Get the number of satellites in the data
    n = len(data)
    # Loop through each pair of satellites to check for collisions
    for i in range(n):
        for j in range(i + 1, n):
            # Get the positions of the two satellites
            pos1 = np.array(data.iloc[i][['x', 'y', 'z']])
            pos2 = np.array(data.iloc[j][['x', 'y', 'z']])

            # Debugging: Print positions being compared
            print(f"Comparing {data.iloc[i]['name']} (Position: {pos1}) and {data.iloc[j]['name']} (Position: {pos2})")

            # Check if positions are identical (distance = 0)
            if np.array_equal(pos1, pos2):
                #st.warning(f"Identical positions for satellites {data.iloc[i]['name']} and {data.iloc[j]['name']}. Skipping.")
                continue  # Skip this pair

            # Calculate the Euclidean distance between the two satellites
            distance = euclidean(pos1, pos2) * 100  # Convert to kilometers
            print(f"Distance between {data.iloc[i]['name']} and {data.iloc[j]['name']}: {distance:.2f} km")

            # If the distance is less than the threshold, consider it a collision
            if distance < threshold:
                # Append the collision details to the list
                collisions.append((data.iloc[i]['name'], data.iloc[j]['name'], data.iloc[i]['timestamp'], distance))
    # Return the list of collisions
    return collisions

# Step 4: Mitigation Suggestions
def suggest_mitigation(distance):
    # Provide mitigation suggestions based on the distance between satellites
    if distance < 1.0:
        return "Immediate Evasive Maneuver Required"
    elif distance < 3.0:
        return "Prepare for Evasive Action"
    else:
        return "Monitor Closely"

# Step 5: Visualization using Plotly
def plot_satellites(data, collisions):
    # Add a 'risk_level' column to the DataFrame and set it to 'Low' by default
    data['risk_level'] = 'Low'
    # Update the risk level for satellites involved in collisions
    for sat1, sat2, timestamp, dist in collisions:
        data.loc[data['name'].isin([sat1, sat2]), 'risk_level'] = 'High' if dist < 1.0 else 'Medium'

    # Create a color map for different risk levels
    color_map = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    # Map the risk levels to colors
    colors = data['risk_level'].map(color_map)

    # Create a 3D scatter plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=data['x'], y=data['y'], z=data['z'], mode='markers',
        marker=dict(size=6, color=colors, opacity=0.8),
        name='Satellites'
    ))

    # Add lines to indicate potential collisions
    for sat1, sat2, timestamp, dist in collisions:
        sat1_data = data[data['name'] == sat1]
        sat2_data = data[data['name'] == sat2]
        fig.add_trace(go.Scatter3d(
            x=[sat1_data['x'].values[0], sat2_data['x'].values[0]],
            y=[sat1_data['y'].values[0], sat2_data['y'].values[0]],
            z=[sat1_data['z'].values[0], sat2_data['z'].values[0]],
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Collision: {sat1} & {sat2}'
        ))
        # Display mitigation suggestions for each collision
        mitigation = suggest_mitigation(dist)
        st.write(
            f"Collision Risk: {sat1} and {sat2} at {timestamp} with Distance {dist:.2f} km | Mitigation: {mitigation}")

    # Update the layout of the 3D plot
    fig.update_layout(title='Satellite Trajectories in 3D',
                      scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'))
    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

# Step 6: Plotting the Trajectory of Selected Satellite with Increased Variance
def plot_trajectory(selected_satellite, satellites, collisions):
    # Create a timescale object for time-related calculations
    ts = load.timescale()
    # Initialize lists to store original and new trajectory data
    original_x, original_y, original_z = [], [], []
    new_x, new_y, new_z = [], [], []

    # Get the selected satellite data
    for sat in satellites:
        if sat.name == selected_satellite:
            # Check if the selected satellite is involved in any collision
            collision_involved = any(selected_satellite in (sat1, sat2) for sat1, sat2, _, _ in collisions)

            # Simulate 100 points for the trajectory
            for i in range(100):
                # Increment time in steps (e.g., every 10 minutes)
                t = ts.utc(2025, 1, 1, 0, 0) + i * 10
                # Calculate the satellite's position at the current time
                position = sat.at(t).position.km

                # Check for zero or invalid positions
                if np.all(position == 0):
                    st.warning(f"Invalid position data for satellite {sat.name} at time {t}. Skipping.")
                    continue  # Skip this time step

                # Append the original position to the lists
                original_x.append(position[0])
                original_y.append(position[1])
                original_z.append(position[2])

                # If the satellite is involved in a collision, adjust its velocity to avoid collision
                if collision_involved:
                    # Apply a larger delta-v to increase the variance in the new trajectory
                    delta_v = np.array([0.1, 0.1, 0.1])  # Increased delta-v (km/s)
                    # Adjust position based on delta-v
                    new_position = position + delta_v * i
                    # Append the new position to the lists
                    new_x.append(new_position[0])
                    new_y.append(new_position[1])
                    new_z.append(new_position[2])
                else:
                    # If no collision, keep the original trajectory
                    new_x.append(position[0])
                    new_y.append(position[1])
                    new_z.append(position[2])

    # Plot the original and new trajectories
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=original_x, y=original_y, z=original_z, mode='lines',
        line=dict(color='blue', width=4),
        name=f'{selected_satellite} - Original Trajectory'
    ))
    fig.add_trace(go.Scatter3d(
        x=new_x, y=new_y, z=new_z, mode='lines',
        line=dict(color='orange', width=4),
        name=f'{selected_satellite} - New Trajectory'
    ))

    # Update the layout of the 3D plot
    fig.update_layout(title=f"Satellite Trajectories for {selected_satellite}",
                      scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'))
    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

# Streamlit UI
st.title("Real-Time Satellite Collision Detection")

# Fetching the TLE data
st.button("Fetch Latest TLE Data", on_click=fetch_tle)
# Read the TLE data and convert it to a DataFrame
df, satellites = read_tle('active_satellites.txt')
# Save the DataFrame to a CSV file
df.to_csv('satellite_data.csv', index=False)
# Display a success message in the Streamlit app
st.success("Data saved to satellite_data.csv")

# Display satellite names in the sidebar
st.sidebar.title("Satellite List")
for name in df['name'].unique():
    st.sidebar.write(name)

# Select a satellite to show the trajectory
satellite_names = df['name'].unique()
selected_satellite = st.selectbox('Select a Satellite', satellite_names)

# Collision detection logic
threshold = st.slider("Set Collision Detection Threshold (km)", 0.5, 10.0, 5.0)
collisions = detect_collisions(df, threshold)

# Show collision report
st.write("## Collision Report")
if collisions:
    st.write(f"Detected {len(collisions)} possible collisions.")
else:
    st.write("No Collisions Detected")

# Plot original and new trajectories for the selected satellite
plot_trajectory(selected_satellite, satellites, collisions)

# Plot collision detection results
plot_satellites(df, collisions)

# Auto-refresh every 3600 seconds (this is for updating data periodically)
while True:
    time.sleep(3600)
    df, satellites = read_tle('active_satellites.txt')
    collisions = detect_collisions(df, threshold)
    plot_satellites(df, collisions)