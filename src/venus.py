#!/usr/bin/env python
# coding: utf-8

"""Hello Future Tutor! This is my code so far. Background is that only data on
Venus is radar imagery of its surface, but we don't know what happened for it
to look like it does today over its 4.5Byr history. Astrophysicists usually
measure surface evolution by observing the craters and to what degree they are
destroyed by thermal processes like volcanic explosions. This is what I'm
attempting to do.

My model is essentially observing two constant processes, crater creation (when
asteroids hit) and crater destruction (when a resurfacing event occurs and
cover craters with lava/rock). The constant input values are from NASA,
where I'm working. The model goes kinda year-by-year like this; at the
beginning of history, x number of craters were emplaced. Then at time 2, x
number of resurfacing events occurred that could either leave craters pristine
as before, modified, or destroyed. Now, reclassify the craters emplaced prior
to the event as one of those 3 categories based on the distance between it and
resurfacing event/the radius of the impacted area, and move on to time 3.
Repeat until 4.5Byr is over.

this is very basic and rough draft.Not really doing anything that helpful to
NASA yet.

goal is to have results that could be compared to what we have observed on
Venus; the 2 observational constraints include (1) Venus has ~1000 craters (not
including destroyed ones because we cant see them), (2) only ~175 craters are
modified craters.

I'm now trying to change my model to be more efficient, or simply perfroming
the same task in a better way. This could mean splitting into time bins instead
of year-by-year. I also want to incorporate some type of geometric series so
that we can always end up with 1000 craters on the planet, but the amount
emplaced throughout the history differs by test run, depending on how many were
destroyed.
"""

from __future__ import division
from collections import Counter
import math
import random

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


num_simulations = 1
Venus_radius = 6051.8  # km
Venus_SurfaceArea = 4 * math.pi * (Venus_radius**2)  # km^2
num_impacts = 1000

RESURFACING_AREAS = [0.1, 0.05, 0.01, 0.007, 0.001, 0.0001]

SIM_LENGTH = 1000  # yrs


def total_resurfaced_area(resurfaced_proportion):
    """Calculate the total resurfaced area from the resurfaced proportion."""
    return resurfaced_proportion * Venus_SurfaceArea


def resurfacing_event_radius(resurfaced_proportion):
    """Calculate the radius of a given resurfacing event from the resurfaced
    proportion.
    """
    return (resurfaced_proportion * Venus_SurfaceArea / math.pi) ** 0.5


def sample_location():
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    xx = Venus_radius * np.sin(phi) * np.cos(theta)
    yy = Venus_radius * np.sin(phi) * np.sin(theta)
    zz = Venus_radius * np.cos(phi)

    return [xx, yy, zz]


resurfacing_event_areas = []
resurfacing_event_years = []
resurfacing_event_locations = []
while sum(resurfacing_event_areas) < 1:
    resurfacing_event_areas.append(random.choice(RESURFACING_AREAS))
    # Truncating vs. scaling vs. leaving excess area (>100%)
    resurfacing_event_years.append(random.uniform(0, SIM_LENGTH))
    resurfacing_event_locations.append(sample_location())

resurfacing_event_years = list(sorted(resurfacing_event_years))


# Each impact period ends with a resurfacing event
resurfacing_event_years = [0.] + resurfacing_event_years
resurfacing_event_areas = [0.] + resurfacing_event_areas
resurfacing_event_locations = [sample_location()] + resurfacing_event_locations

resurfacing_event_years.append(SIM_LENGTH)
resurfacing_event_areas.append(0)
resurfacing_event_locations.append(sample_location())


events_df = pd.DataFrame(np.column_stack([resurfacing_event_years, resurfacing_event_areas, resurfacing_event_locations]), columns=["year", "area", "x", "y", "z"])
events_df["r"] = resurfacing_event_radius(events_df["area"])

events_df["Lon"] = np.degrees(np.arctan2(events_df["y"], events_df["x"]))
events_df["Lat"] = np.degrees(np.arcsin(events_df["z"] / Venus_radius))

events_df = events_df.astype(float)



impact_periods = [(resurfacing_event_years[i], resurfacing_event_years[i + 1])
                  for i in range(len(resurfacing_event_years) - 1)]


# Rate curves --- add later --- assume crater formation rate is constant for now
def impact_rate(time):
    return 1


def definite_integral(f, a, b, n=64):
    """Return the definite integral of f on [a, b] approximated by the
    trapezoidal rule with n points."""
    return np.trapz([f(x) for x in np.linspace(a, b, n)], dx=(b - a)/n)


# Want to find the scaling factor X so that the total craters at end of era checks out
# Initialize number of craters N to 0
# For each event area E_a and impact period length T_i,
# Add X * T_i (or the integral of the rate curve) to N
# Multiply N by (1 - E_a)
unscaled_mean = 0.
for impact_period, resurfaced_area in zip(impact_periods, resurfacing_event_areas):
    (period_start, period_end) = impact_period
    unscaled_mean += definite_integral(impact_rate, period_start, period_end)
    unscaled_mean *= (1 - resurfaced_area)

scaling_factor = num_impacts / unscaled_mean


##radius in the paper is 15km, but im working with sphere of radius 1, so 15km/Venus circumference is this
radius_craters = 15  # km


def crater_classify(crater, event):
    # Great circle distance between impact center and event center
    [xi, yi, zi] = crater[1:4]
    [xj, yj, zj] = event[2:5]

    magi = math.sqrt(xi**2 + yi**2 + zi**2)
    magj = math.sqrt(xj**2 + yj**2 + zj**2)

    dot = (xi * xj) + (yi * yj) + (zi * zj)
    theta = np.arccos(dot / (magi * magj))  # radians

    distance = Venus_radius * theta

    event_r = event[5]

    crater_class = crater[5]
    crater_r = crater[4]

    max_distance = event_r + crater_r
    min_distance = event_r - crater_r

    if crater_class == "pristine":
        if distance > max_distance:
            return "pristine"

        if max_distance >= distance >= min_distance:
            return "modified"

        if min_distance > distance:
            return "destroyed"
    elif crater_class == "modified":
        if min_distance > distance:
            return "destroyed"

        return "modified"
    elif crater_class == "destroyed":
        return "destroyed"


"""SIMULATION"""


def sim_yearbyyear():
    ##num_impacts is set down to 0 since it was used as 1000 in rate determination
    craters = []

    rng = np.random.default_rng()
    for year in range(SIM_LENGTH):
        # Test to see if an impact occurred during this year
        mean_impacts = scaling_factor * definite_integral(impact_rate, year, year + 1)
        impacts_occurred = rng.poisson(mean_impacts)

        # this tests to see if crater occurred in one time increment
        for impact in range(impacts_occurred):
            impactlocation = sample_location()

            crater = [year, *impactlocation, radius_craters, "pristine"]
            craters.append(crater)

        resurfacing_event = None
        for event in events_df.itertuples():
            if int(event.year) == year:
                resurfacing_event = event
                break

        if resurfacing_event:
            for idx, crater in enumerate(craters):
                # TODO This is gross, fix with dataclasses
                crater[5] = crater_classify(crater, resurfacing_event)

    craters_df = pd.DataFrame(craters, columns=["year", "x", "y", "z", "r", "classification"])

    craters_df["Lon"] = np.degrees(np.arctan2(craters_df["y"], craters_df["x"]))
    craters_df["Lat"] = np.degrees(np.arcsin(craters_df["z"] / Venus_radius))

    return craters_df


def sim_timebins():
    craters = []

    # For each impact period (t_i, t_{i + 1})
    # Calculate the integral over period of r(t)
    # Generate that many craters all at once
    # Simulate the resurfacing event at t_{i + 1}

    for idx, period in enumerate(impact_periods):
        (period_start, period_end) = period
        expected_craters = scaling_factor * definite_integral(impact_rate, period_start, period_end)
        for _ in range(int(round(expected_craters))):
            [impact_x, impact_y, impact_z] = sample_location()

            impact_date = random.uniform(period_start, period_end)
            crater = [impact_date, impact_x, impact_y, impact_z, radius_craters, "pristine"]
            craters.append(crater)

        resurfacing_event = events_df.iloc[idx].array
        for crater in craters:
            crater[5] = crater_classify(crater, resurfacing_event)

    craters_df = pd.DataFrame(craters, columns=["year", "x", "y", "z", "r", "classification"])

    craters_df["Lon"] = np.degrees(np.arctan2(craters_df["y"], craters_df["x"]))
    craters_df["Lat"] = np.degrees(np.arcsin(craters_df["z"] / Venus_radius))

    return craters_df


# craters_df = sim_yearbyyear()
craters_df = sim_timebins()

prisdf = craters_df[craters_df["classification"] == "pristine"]
modifydf = craters_df[craters_df["classification"] == "modified"]
destroydf = craters_df[craters_df["classification"] == "destroyed"]

priscraterLon = prisdf["Lon"].to_numpy()
priscraterLat = prisdf["Lat"].to_numpy()
modcraterLon = modifydf["Lon"].to_numpy()
modcraterLat = modifydf["Lat"].to_numpy()
destroycraterLon = destroydf["Lon"].to_numpy()
destroycraterLat = destroydf["Lat"].to_numpy()

eventLon = events_df["Lon"].to_numpy()
eventLat = events_df["Lat"].to_numpy()
eventR = events_df["r"].to_numpy()

counts = Counter(craters_df["classification"])
print(counts)


"""PLOTTING/VISUALIZATION"""

#%% Previous Mollweides

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=0))
ax = plt.axes(projection=ccrs.Mollweide())
plt.title("Surface At End of Simulation")

plt.scatter(
    x=prisdf["Lon"],
    y=prisdf["Lat"],
    color="red",
    s=15,
    alpha=1,
    transform=ccrs.PlateCarree(),
)


plt.scatter(
    x=events_df["Lon"],
    y=events_df["Lat"],
    color="blue",
    s=events_df["r"],
    alpha=0.2,
    transform=ccrs.PlateCarree(),
)
plt.scatter(
    x=modifydf["Lon"],
    y=modifydf["Lat"],
    color="green",
    s=40,
    alpha=1,
    transform=ccrs.PlateCarree(),
)

plt.scatter(
    x=destroydf["Lon"],
    y=destroydf["Lat"],
    color="purple",
    s=15,
    alpha=1,
    transform=ccrs.PlateCarree(),
)

def plot_craters(
    priscraterLon,
    priscraterLat,
    modcraterLon,
    modcraterLat,
    destroycraterLon,
    destroycraterLat,
    eventLon,
    eventLat,
    eventR,
    org=0,
    title="Mollweide projection",
    projection="mollweide",
):
    ##RA, Dec are arrays of the same length.
    ##RA takes values in [0,360), Dec in [-90,90], which represent angles in degrees.
    # org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    # title is the title of the figure.
    # projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + org, 360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection, facecolor="LightCyan")
    ax.scatter(
        np.radians(priscraterLon), np.radians(priscraterLat), color="purple", s=15
    )  # convert degrees to radians
    ax.scatter(np.radians(modcraterLon), np.radians(modcraterLat), color="green", s=50)
    ax.scatter(
        np.radians(destroycraterLon), np.radians(destroycraterLat), color="blue", s=40
    )

    ax.scatter(
        np.radians(eventLon), np.radians(eventLat), color="red", s=eventR, alpha=0.5
    )
    ax.set_xticklabels(tick_labels)  # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("Lon")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Lat")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)
    

plot_craters(
    priscraterLon,
    priscraterLat,
    modcraterLon,
    modcraterLat,
    destroycraterLon,
    destroycraterLat,
    eventLon,
    eventLat,
    eventR,
)

plt.show()
