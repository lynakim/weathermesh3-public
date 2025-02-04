import matplotlib.pyplot as plt


matemem = [3.50,21.82,44.01]
sysram = [7.55,35.24,68.54]
vram = [11.14,12.96,13.62]
timesteps = ['[0,6]','[0,6,24,48,53]','[0,6,24,48,53,144]']
n = len(timesteps)

slope = (sysram[-1]-sysram[0])/(matemem[-1]-matemem[0])
intercept = sysram[0]-slope*matemem[0]

print(f"y = {slope:.3f}x + {intercept:.3f}")


plt.plot(matemem,sysram,'black')
for i in range(n):
    plt.plot(matemem[i],sysram[i],'o',label=timesteps[i])


plt.xlabel('Matepoint Memory (GB)')
plt.ylabel('System RAM (GB)')
plt.legend()
plt.grid()
plt.savefig('mateplot.png')