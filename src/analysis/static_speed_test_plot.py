import matplotlib.pyplot as plt

bmssp_times = [0.000100, 0.000679, 0.004566, 0.022052, 0.161711, 0.638586, 3.079402]
dijkstra_times = [0.000070, 0.000517, 0.004540, 0.021895, 0.099225, 0.413363, 2.037235]

iterations = len(bmssp_times)
starting_satellites = 8
satellite_counts = [starting_satellites * (4 ** i) for i in range(iterations)]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(iterations)
plt.bar(index, bmssp_times, bar_width, label='Bmssp', color='blue')
plt.bar([i + bar_width for i in index], dijkstra_times, bar_width, label='Dijkstra', color='orange')
plt.xlabel('Number of Satellites')
plt.ylabel('Average Computation Time (seconds)')
plt.title('Static Speed Test: Bmssp vs Dijkstra')
plt.xticks([i + bar_width / 2 for i in index], satellite_counts)
plt.legend()
plt.tight_layout()
plt.savefig('static_speed_test.png')
plt.show() 


plt.figure(figsize=(10, 6))
plt.bar(index, bmssp_times, bar_width, label='Bmssp', color='blue')
plt.bar([i + bar_width for i in index], dijkstra_times, bar_width, label='Dijkstra', color='orange')
plt.xlabel('Number of Satellites')
plt.ylabel('Average Computation Time (seconds)')
plt.title('Static Speed Test: Bmssp vs Dijkstra (Log Scale)')
plt.xticks([i + bar_width / 2 for i in index], satellite_counts)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('static_speed_test_log.png')
plt.show()  