# A wrapper for integrating with the tarot 18 simulation. Only simulates a single step according to some current passed. Output voltages are saved in the filename given.

from battery_electrochem_TarotT18 import BatteryElectroChem as Battery

class TarotBattery():
    def __init__(self, timeStep, filename):
        self.battery = Battery()
        # initialize to tarot. The zeros mean nothing. See line 233 of battery_electromchem_TarotT18 file
        self.x = self.battery.initialize(0, 0)
        self.t = 0
        self.timeStep = timeStep
        self.f = open(filename, 'w')

    # Step only a single timestep. 
    def step(self, currentNeeded): 
        self.t += self.timeStep
        self.x = self.battery.next_state(self.x, currentNeeded, self.timeStep)
        output = self.battery.output(self.x)['v']
        self.f.write(str(output) + "\n")
        return output


