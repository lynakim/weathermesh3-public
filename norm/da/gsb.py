import os 

os.environ['WB_CLIENT_ID'] = 'mission_analysis'
os.environ['WB_API_KEY'] = '3c2e2cb4b2a7b98efc42a9ba89c325b'

from windborne import poll_observations

poll_observations(
    start_time='2024-12-18_00:00',
    output_format='csv'  # or 'little_r'
)