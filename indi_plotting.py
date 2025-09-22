import pandas as pd
import matplotlib.pyplot as plt
import sys

# filepath = 'var/logs/20250917-125522.csv'  #current acceleration is in the feedback loop
# filepath = 'var/logs/20250917-130011.csv' #current acceleration is not in the feedback loop
filepath = 'var/logs/20250917-164049.csv' 




try:
    # Load CSV file with headers. Must update the csv being logged
    df = pd.read_csv(filepath)   
except FileNotFoundError:
    print(f"Error: Filepath '{filepath}' does not exist. Please update filepath.")
    sys.exit(1) 



# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['rate_sp_measure.p'], label='rate_sp_measure.p', linewidth=2)
plt.plot(df['time'], df['rate_sp_measure.q'], label='rate_sp_measure.q', linewidth=2)
plt.plot(df['time'], df['rate_sp_measure.r'], label='rate_sp_measure.r', linewidth=2)
# # plt.plot(df['time'], df['pos_x_ref'], label='Position Ref', linewidth=2)
# plt.plot(df['time'], df['pos_x_actual'], label='Position Actual', linewidth=2)
# plt.plot(df['time'], df['vel_x_ref'], label='Velocity Ref', linewidth=2)
# plt.plot(df['time'], df['vel_x_actual'], label='Velocity Actual', linewidth=2)
plt.plot(df['time'], df['acc_x_ref'], label='Acceleration Ref', linewidth=2)
plt.plot(df['time'], df['acc_x_actual'], label='Acceleration Actual', linewidth=2)
# plt.plot(df['time'], df['rate_p'], label='Rate p', linewidth=2)
plt.plot(df['time'], df['rate_q'], label='Rate q', linewidth=2)
plt.plot(df['time'], df['att_phi'], label='Att phi', linewidth=2)
plt.plot(df['time'], df['att_theta'], label='Att theta', linewidth=2)
# plt.plot(df['time'], df['rate_r'], label='Rate r', linewidth=2)
# plt.plot(df['time'], df['T_calculated'], label='T calculated', linewidth=2)
plt.plot(df['time'], df['roll_rate_cmd'], label='Commanded roll rate', linewidth=2)
plt.plot(df['time'], df['pitch_rate_cmd'], label='Commanded pitch rate', linewidth=2)
# plt.plot(df['time'], df['dcmd[0]'], label='dcmd[0]', linewidth=2)
# plt.plot(df['time'], df['dcmd[1]'], label='dcmd[1]', linewidth=2)
# plt.plot(df['time'], df['dcmd[2]'], label='dcmd[2]', linewidth=2)
# plt.plot(df['time'], df['qi'], label='q.i', linewidth=2)
# plt.plot(df['time'], df['qx'], label='q.x', linewidth=2)
# plt.plot(df['time'], df['qy'], label='q.y', linewidth=2)
# plt.plot(df['time'], df['qz'], label='q.z', linewidth=2)



plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Logging')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()