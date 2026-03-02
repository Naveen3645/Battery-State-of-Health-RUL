import scipy.io
import numpy as np


def extract_features(file_path, battery_name):

    data = scipy.io.loadmat(file_path)
    battery = data[battery_name][0][0]
    cycles = battery['cycle'][0]

    all_cycle_data = []

    for i in range(len(cycles)):

        if cycles[i]['type'][0] == 'discharge':

            if 'data' in cycles[i].dtype.names and len(cycles[i]['data']) > 0:

                data_block = cycles[i]['data'][0][0]

                if 'Capacity' in data_block.dtype.names:

                    capacity_array = data_block['Capacity']

                    if capacity_array.size > 0:

                        capacity = capacity_array[0][0]

                        voltage = data_block['Voltage_measured'][0]
                        current = data_block['Current_measured'][0]
                        temperature = data_block['Temperature_measured'][0]
                        time = data_block['Time'][0]

                        mean_voltage = np.mean(voltage)
                        mean_current = np.mean(current)
                        mean_temperature = np.mean(temperature)
                        discharge_duration = time[-1] - time[0]

                        rated_capacity = 2.0
                        soh = (capacity / rated_capacity) * 100

                        all_cycle_data.append([
                            i,
                            capacity,
                            mean_voltage,
                            mean_current,
                            mean_temperature,
                            discharge_duration,
                            soh
                        ])

    return all_cycle_data