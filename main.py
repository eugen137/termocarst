import numpy as np

from src.Zone import Zone

temperature = np.loadtxt("./data/ЕС_Катанга/temperature.txt")
precipitation = np.loadtxt("./data/ЕС_Катанга/precipitation.txt")
square = np.loadtxt("./data/ЕС_Катанга/square.txt")
year_with_square = np.loadtxt("./data/ЕС_Катанга/year_with_square.txt")
years = np.loadtxt("./data/ЕС_Катанга/years.txt", dtype=int)

z = Zone(square, year_with_square, temperature, precipitation, years, alpha=np.array([-2.7, 10]),
         beta=np.array([-1.9, 10]), ksi=np.array([-0.05, 0.05]))

# z.params_gaps()
z.theta_calc()
z.modeling(200)
print(z.restored_square)
z.draw()

