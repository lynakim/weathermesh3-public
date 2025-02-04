from utils import *

def plot_hour_vs_day(dates):
    plt.figure(figsize=(10,5))
    days = [get_date(d) for d in dates]
    days = [datetime(d.year,d.month,d.day) for d in days]
    hours = [get_date(d).hour for d in dates]
    plt.scatter(days,hours,s=2)
    plt.yticks([0,6,12,18,24])
    plt.ylim(-0.5,23.5); plt.ylabel("z");
    plt.grid(); plt.tight_layout()
    plt.savefig("ohp.png")
