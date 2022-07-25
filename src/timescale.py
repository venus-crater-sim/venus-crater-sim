import random


RESURFACING_AREAS = [0.1, 0.05, 0.01, 0.007, 0.001, 0.0001]

SIM_LENGTH = 4500  # My


resurfacing_event_areas = []
resurfacing_event_years = []
while sum(resurfacing_event_areas) < 1:
    resurfacing_event_areas.append(random.choice(RESURFACING_AREAS))
    resurfacing_event_years.append(random.uniform(0, SIM_LENGTH))
    resurfacing_event_years = list(sorted(resurfacing_event_years))

# Consider trimming vs. leaving excess area (>100%)
print(resurfacing_event_areas)
print(resurfacing_event_years)

# Each impact period ends with a resurfacing event
# Add start and end of simulation
impact_periods = [(resurfacing_event_years[i], resurfacing_event_years[i + 1])
                  for i in range(len(resurfacing_event_years) - 1)]

# Rate curves --- add later --- assume crater formation rate is constant for now

# Want to find the scaling factor X so that the total craters at end of era checks out
# Initialize number of craters N to 0
# For each event area E_a and impact period length T_i,
# Add X * T_i (or the integral of the rate curve) to N
# Multiply N by (1 - E_a)

num_craters = 0  # This is the wrong variable name
for impact_period, resurfaced_area in zip(impact_periods, resurfacing_event_areas):
    (period_start, period_end) = impact_period
    num_craters += period_end - period_start
    num_craters *= (1 - resurfaced_area)

scaling_factor = 1000 / num_craters
print(num_craters, scaling_factor)
