import numpy as np
from collections import deque, namedtuple
import random


class env_2agent:
    def __init__(self):
        self.n = 2000
        self.packets = np.zeros(self.n)
        self.dis = None
        self.energyData = []
        self.energy_consumption = []
        self.success = []
        self.seed = 0
        self.energyDataall = None
        self.channel = np.zeros(self.n)
        self.num_subchannels = 500
        self.max_steps = 100  # Fixed episode length
        self.current_step = 0  # Step counter

        # Define observation and action specs
        ObservationInfo = namedtuple('rlNumericSpec', ['shape'])
        self.ObservationInfo = [
            ObservationInfo(shape=(1, 9)),
            ObservationInfo(shape=(1, 9)),
        ]

        values1 = np.arange(0.005, 0.105, 0.005)  # 0.005:0.005:0.1
        values2= np.arange(1, 21, 1)
        values3 = np.arange(0, 55, 5)  # 0:5:50

        FiniteSetSpec = namedtuple('rlFiniteSetSpec', ['values'])
        self.ActionInfo = [
            FiniteSetSpec(values=values2),  # 1:20
            FiniteSetSpec(values=values3)
        ]

    def reset(self):
        self.packets = np.zeros(self.n)
        InitialState = {'State': [np.zeros(9), np.zeros(9)]}
        InitialObservation = InitialState['State']

        # For simulation
        if hasattr(self, 'energyData') and len(self.energyData) > 0:
            if self.energyDataall is None:
                self.energyDataall = np.array([])
            self.energyDataall = np.append(self.energyDataall, np.sum(self.energyData))

        # Generating devices in each region uniformly
        total_devices = 2000
        regions = 5
        region_bounds = np.linspace(0, 500, regions + 1)

        # Pre-allocate array to store distances
        self.dis = np.zeros(total_devices)

        # Assign each device to a random distance within the defined regions
        for i in range(total_devices):
            region_idx = random.randint(0, regions - 1)  # 0-based index
            self.dis[i] = region_bounds[region_idx] + random.random() * \
                          (region_bounds[region_idx + 1] - region_bounds[region_idx])

        # Sort distances and assign channels
        sorted_indices = np.argsort(self.dis)
        subchannel_assignment = np.mod(np.arange(self.n), self.num_subchannels) + 1

        for i in range(self.n):
            self.channel[sorted_indices[i]] = subchannel_assignment[i]

        self.seed = 0
        self.current_step = 0  # Reset step counter
        return InitialObservation, InitialState

    def step(self, action):
        # Parameters
        noise_dbm = -120
        noise = 10 ** (noise_dbm / 10)
        gamma_db = 0
        gamma = 10 ** (gamma_db / 10)
        alpha = 4  # path-loss exponent
        ms = 10 ** -3
        Slot = 1 * ms  # slot Time_s length
        N_RAR = 40  # RAR window size
        Pr = 80 * (10 ** -3)  # receive power
        Energy = 1e-140  # base energy
        rngs = np.array([0, 100, 200, 300, 400, 500])

        # Generate active devices
        self.seed += 1
        np.random.seed(self.seed)
        arrival = np.random.rand()
        ir = 0 + arrival * 0.1
        device_status = np.zeros(self.n)
        device_status[self.packets > 0] = -1
        new_packets = np.random.poisson(ir, 2000)
        num_new_packet = np.sum(new_packets > 0)

        # Update packets
        self.packets += new_packets

        # Find active devices
        active_devices = np.where(self.packets > 0)[0]
        num_active = len(active_devices)
        distance=self.dis.copy()
        distance[self.packets==0] = 0
        active_distance = distance
        collisioned_CTU = 0
        unused_CTU = 0

        actives = np.zeros(5)

        for i in range(5):
            active_indices = np.where(self.packets > 0)[0]  # Get indices of active packets
            active_distances = self.dis[active_indices]  # Get corresponding distances

            # Filter distances within the range
            filtered_actives = active_distances[(active_distances >= rngs[i]) & (active_distances < rngs[i + 1])]

            actives[i] = len(filtered_actives)

        # Generate channel gains
        channel_fading = np.random.exponential(1, 2000)
        channel_gain = channel_fading * (self.dis ** -alpha)
        channel_gain[self.packets == 0] = 0

        current_channel = self.channel.copy()
        current_channel[self.packets == 0] = 0
        r=0.04
        # Process actions
        c, rr = action
        print(c, rr)
        # Generate power levels
        minValue = 1e-12
        maxValue = 1
        pp = []
        x = 0

        while True:
            f_x = (np.exp(c * x) - 1) / (np.exp(c) - 1)
            current_value = minValue + (maxValue - minValue) * f_x

            if current_value >= maxValue:
                break

            pp.append(current_value)
            x += r

        if r == 1:
            pp = [maxValue]

        # Assign power levels based on regions
        region_bounds = np.linspace(0, 500, rr + 1)
        power_level = np.zeros(self.n)

        if len(pp) >= rr and rr > 1:
            # Split power pool into regions
            base_size = len(pp) // rr
            remainder = len(pp) % rr
            part_sizes = [base_size] * (rr - remainder) + [base_size + 1] * remainder
            pool_parts = []
            start = 0

            for size in part_sizes:
                pool_parts.append(pp[start:start + size])
                start += size

            # Assign power levels based on region
            for i in active_devices:
                distance = self.dis[i]
                region_index = np.sum(distance > region_bounds[:-1]) - 1

                if 0 <= region_index < len(pool_parts):
                    power_level[i] = np.random.choice(pool_parts[region_index])
                else:
                    power_level[i] = minValue
        else:
            random_indices = np.random.randint(0, len(pp), size=self.n)
            power_level = np.array([pp[i] for i in random_indices])

        power_level[self.packets == 0] = 0
        power_r = power_level * channel_gain
        power_T = power_level.copy()

        # Energy calculations
        Energy += np.sum(power_T[self.packets > 0]) * 2 * Slot
        Energy_pl = np.sum(power_T[self.packets > 0])

        # Energy per power level
        eng = np.zeros(10)
        active = np.zeros(10)
        for i in range(10):
            powerlevel_value = (i + 1) / 10
            mask = (self.packets > 0) & (np.abs(power_level - powerlevel_value) < 1e-9)
            eng[i] = 1e-10 + np.sum(power_level[mask]) * 2 * Slot
            active[i] = np.sum(mask)

        active_half = [
            np.sum(self.dis[self.packets > 0] < 250),
            np.sum(self.dis[self.packets > 0] > 250)
        ]

        Time_s = 2 * Slot

        # SINR calculations
        SINR = np.zeros(self.n)
        channel_powers = [[] for _ in range(self.num_subchannels)]
        channel_r = [[] for _ in range(self.num_subchannels)]
        channel_indices = [[] for _ in range(self.num_subchannels)]

        for k in range(self.num_subchannels):
            devices_on_channel = np.where(current_channel == k + 1)[0]
            if len(devices_on_channel) == 0:
                unused_CTU += len(pp)
                continue

            channel_powers[k] = power_level[devices_on_channel]
            channel_r[k] = power_r[devices_on_channel]
            channel_indices[k] = devices_on_channel

            for power_val in np.unique(power_level):
                count = np.sum(power_level[devices_on_channel] == power_val)
                if count == 0:
                    unused_CTU += 1
                elif count > 1:
                    collisioned_CTU += 1

        # Calculate SINR per channel
        for k in range(self.num_subchannels):
            if len(channel_r[k]) == 0:
                continue

            sort_idx = np.argsort(channel_r[k])[::-1]
            sorted_power_r = channel_r[k][sort_idx]
            sorted_indices = channel_indices[k][sort_idx]

            for ch in range(len(sorted_power_r)):
                interference = np.sum(sorted_power_r[ch + 1:])
                SINR_val = sorted_power_r[ch] / (interference + noise)
                SINR[sorted_indices[ch]] = SINR_val
                if SINR_val < gamma:
                    break

        # Success metrics
        xi = 64 * 2000  # Total preamble space
        self.preambles = np.random.randint(1, xi + 1, size=self.n)
        self.preambles[self.packets == 0] = 0  # Clear inactive devices
        SINR[self.preambles == 0] = 0
        succ_index = np.where(SINR >= gamma)[0]
        num_succ = len(succ_index)

        # Update system state
        self.preambles[SINR < gamma] = 0
        Energy += (len(succ_index) * (N_RAR / 2) * Slot * Pr +
                   (len(np.where(self.preambles > 0)[0]) - len(succ_index)) * N_RAR * Pr * Slot)
        Time_s += (len(succ_index) * (N_RAR / 2) * Slot +
                   (len(np.where(self.preambles > 0)[0]) - len(succ_index)) * N_RAR * Slot)

        self.packets[succ_index] -= 1
        Energy = np.sum(power_level)
        energy_Efficiency = num_succ / max(Energy, 1e-10)
        success_probability = num_succ / max(num_active, 1)

        # Failure analysis
        fail_dist1 = np.zeros(5)

        for i in range(5):
            active_indices = np.where(self.packets > 0)[0]  # Get indices of active packets
            active_distances = self.dis[active_indices]  # Get corresponding distances

            # Filter distances within the range
            fail_dis1 = active_distances[(active_distances >= rngs[i]) & (active_distances < rngs[i + 1])]

            fail_dist1[i] = len(fail_dis1)

        # Update metrics
        self.energyData.append(energy_Efficiency)
        self.energy_consumption.append(Energy_pl)
        num_succ = max(1, num_succ)
        self.success.append(num_succ)

        # Prepare observations and rewards
        obs = [
            np.concatenate([action, [num_succ, Energy_pl], fail_dist1]),
            np.concatenate([action, [num_succ, Energy_pl], fail_dist1]),
        ]

        reward = [num_succ/Energy for _ in range(2)]

        # Increment step counter
        self.current_step += 1

        # Determine termination
        done = self.current_step >= self.max_steps

        return obs, reward, done