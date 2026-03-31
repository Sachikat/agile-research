import pandas as pd
import matplotlib.pyplot as plt


#Index(['wb', 'time_abs', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz', 'species', 'moth',
#    'trial', 'wblen', 'phase', 'time', 'muscle', 'wbfreq', 'fx_pc1',
#    'fx_pc2', 'fx_pc3', 'fx_pc4', 'fx_pc5', 'fx_pc6', 'fx_pc7', 'fx_pc8',
#    'fx_pc9', 'fx_pc10', 'fy_pc1', 'fy_pc2', 'fy_pc3', 'fy_pc4', 'fy_pc5',
#    'fy_pc6', 'fy_pc7', 'fy_pc8', 'fy_pc9', 'fy_pc10', 'fz_pc1', 'fz_pc2',
#    'fz_pc3', 'fz_pc4', 'fz_pc5', 'fz_pc6', 'fz_pc7', 'fz_pc8', 'fz_pc9',
#    'fz_pc10', 'tx_pc1', 'tx_pc2', 'tx_pc3', 'tx_pc4', 'tx_pc5', 'tx_pc6',
#    'tx_pc7', 'tx_pc8', 'tx_pc9', 'tx_pc10', 'ty_pc1', 'ty_pc2', 'ty_pc3',
#    'ty_pc4', 'ty_pc5', 'ty_pc6', 'ty_pc7', 'ty_pc8', 'ty_pc9', 'ty_pc10',
#    'tz_pc1', 'tz_pc2', 'tz_pc3', 'tz_pc4', 'tz_pc5', 'tz_pc6', 'tz_pc7',
#    'tz_pc8', 'tz_pc9', 'tz_pc10'],
#   dtype='str')

df = pd.read_csv('preprocessedCache.csv')

# print(df)

var = 'tz'
avg_phase = df.groupby(['species','phase'])[var].mean().reset_index()

plt.figure(figsize=(16,10))

for sp in avg_phase['species'].unique():
    sub = avg_phase[avg_phase['species'] == sp]
    plt.plot(sub['phase'], sub[var], linewidth=0.5,  label=sp)

plt.xlabel('Phase (0–1)')
plt.ylabel(f'Mean {var.upper()}')
plt.title('Force vs Wingbeat Phase')

plt.legend(fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'phase_plot_{var}.png', dpi=300)
#plt.show()


# var = 'fz'

# muscles = df['muscle'].unique()
# species_list = df['species'].unique()

# for m in muscles:
    
#     muscle_data = df[df['muscle'] == m]
    
#     avg_phase = muscle_data.groupby(['species','phase'])[var].mean().reset_index()
#     avg_phase = avg_phase.sort_values('phase')
    
#     for sp in species_list:
#         sub = avg_phase[avg_phase['species'] == sp]

#     plt.figure(figsize=(16,10))
#     plt.plot(sub['phase'], sub[var], linewidth=0.5, label=sp)
#     plt.title(f'{var.upper()} vs Phase — Muscle: {m} - Species {sp}', fontsize=16)
#     plt.xlabel('Phase', fontsize=14)
#     plt.ylabel(f'Mean {var.upper()}', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'{var.upper()} vs Phase — Muscle: {m} - Species {sp}' + ".png", dpi=300)
#     plt.close()
#     #plt.show()

# var = 'fz'

# muscles = df['muscle'].unique()
# species_list = df['species'].unique()

# for m in muscles:
    
#     muscle_data = df[df['muscle'] == m]
    
#     avg_phase = muscle_data.groupby(['species','phase'])[var].mean().reset_index()
#     avg_phase = avg_phase.sort_values('phase')
    
#     for sp in species_list:
#         sub = avg_phase[avg_phase['species'] == sp]
#         plt.figure(figsize=(16,10))
#         plt.plot(sub['phase'], sub[var], linewidth=0.5, label=sp)
#         plt.title(f'{var.upper()} vs Phase — Muscle: {m} - Species {sp}', fontsize=16)
#         plt.xlabel('Phase', fontsize=14)
#         plt.ylabel(f'Mean {var.upper()}', fontsize=14)
#         plt.legend(fontsize=10)
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(f'{var.upper()} vs Phase — Muscle: {m} - Species {sp}' + ".png", dpi=300)
#         plt.close()
#         #plt.show()





