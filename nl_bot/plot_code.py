import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Read JSON data from stdin
input_data = sys.stdin.read()
data = json.loads(input_data)

# Extract forecasts
forecasts = data['forecasts'][0]

# Extract time and temperature data
times = [datetime.strptime(forecast['time'], '%Y-%m-%dT%H:%M:%S%z') for forecast in forecasts]
temperatures = [forecast['temperature_2m'] for forecast in forecasts]

# Plot the temperature data
plt.figure(figsize=(10, 5))
plt.plot(times, temperatures, marker='o', linestyle='-')
plt.title('Temperature Forecast for the Next Week')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a file
plt.savefig('output.png')