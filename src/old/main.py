import numpy as np

from src.old.Zone import Zone

temperature = np.loadtxt("../../data/ЕС_Катанга/temperature.txt")
precipitation = np.loadtxt("../../data/ЕС_Катанга/precipitation.txt")
square = np.loadtxt("../../data/ЕС_Катанга/square.txt")
year_with_square = np.loadtxt("../../data/ЕС_Катанга/year_with_square.txt")
years = np.loadtxt("../../data/ЕС_Катанга/years.txt", dtype=int)
print(years)

z = Zone(square, year_with_square, temperature, precipitation, years)
z.theta_calc()

