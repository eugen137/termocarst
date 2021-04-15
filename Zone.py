import numpy as np


class Zone:
    def __init__(self, square: np.ndarray, years_square :np.ndarray, temperature :np.ndarray,
                 recipients :np.ndarray, years :np.ndarray):
        self.square = square
        self.years_square = years_square
        self.temperature = temperature
        self.recipients = recipients
        self.years = years
        y = years_square - years.min()
        self.norm_recipients = (recipients - recipients.min()) / \
                               (recipients.max() - recipients.min())
        self.norm_temp = (temperature - temperature.min()) / \
                               (temperature.max() - temperature.min())
        self.norm_square = (square - square.min()) / \
                         (square.max() - square.min())
        self.temp_y = self.norm_recipients[y]
        self.recip_y = self.norm_temp[y]

    def (self):
