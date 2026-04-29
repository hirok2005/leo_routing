import matplotlib.pyplot as plt

sizes_bmssp = (
    (144.9, ), # 8
    (144.9, ), # 32
    (145, 146, 146, ), # 128
    (145.6, 146.6, 146.6, ), # 512
    (151.2, 151.2, 151.2, ), # 2048
    (166.7, 166.7, 166.8), # 8192
    (233.1, 233.3, 233.1), # 32768
)

sizes_dijkstra = (
    (144.9, ), # 8
    (144.9, 145, 144.9, ), # 32
    (145, ), # 128
    (146.6, ), # 512
    (151.2, 152.2, 151.2, ), # 2048
    (166.7, ), # 8192
    (233.1, ) # 32768
)

constellation_sizes = [8, 32, 128, 512, 2048, 8192, 32768]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(constellation_sizes))
avg_bmssp = [sum(sizes) / len(sizes) for sizes in sizes_bmssp]
avg_dijkstra = [sum(sizes) / len(sizes) for sizes in sizes_dijkstra]
plt.bar(index, avg_bmssp, bar_width, label='Bmssp', color='blue')
plt.bar([i + bar_width for i in index], avg_dijkstra, bar_width, label='Dijkstra', color='orange')
plt.xlabel('Number of Satellites')
plt.ylabel('Average Memory Usage (MB)')
plt.title('Memory Usage: Bmssp vs Dijkstra')
plt.xticks([i + bar_width / 2 for i in index], constellation_sizes)
plt.legend()
plt.tight_layout()
plt.savefig('memory_usage.png')
plt.show()