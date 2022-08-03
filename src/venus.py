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
from dataclasses import dataclass
from itertools import groupby
import math
import random

import cartopy.crs as ccrs
from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VENUS_RADIUS = 6051.8  # km
VENUS_SURFACE_AREA = 4 * math.pi * (VENUS_RADIUS**2)  # km^2
NUM_IMPACTS = 1000

RESURFACING_AREAS = [0.1, 0.05, 0.01, 0.007, 0.001, 0.0001]

SIM_LENGTH = 4.5 * 10 ** 9 # yrs


@dataclass
class ResurfacingEvent:
    year: float
    loc: [float, float, float]
    area: float

    def radius(self):
        """Calculate the radius of a given resurfacing event from the
        resurfaced proportion."""
        return math.sqrt(self.area * VENUS_SURFACE_AREA / math.pi)

    def longitude(self):
        [x, y, z] = self.loc
        return np.degrees(np.arctan2(y, x))

    def latitude(self):
        [x, y, z] = self.loc
        return np.degrees(np.arcsin(z / VENUS_RADIUS))


@dataclass
class Crater:
    year: float
    loc: [float, float, float]
    radius: float
    classification: str

    def longitude(self):
        [x, y, z] = self.loc
        return np.degrees(np.arctan2(y, x))

    def latitude(self):
        [x, y, z] = self.loc
        return np.degrees(np.arcsin(z / VENUS_RADIUS))


def sample_location():
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    x = VENUS_RADIUS * np.sin(phi) * np.cos(theta)
    y = VENUS_RADIUS * np.sin(phi) * np.sin(theta)
    z = VENUS_RADIUS * np.cos(phi)

    return [x, y, z]


# Rate curves --- add later --- assume crater formation rate is constant for now
def impact_rate(time):
    return 1


def definite_integral(f, a, b, n=64):
    """Return the definite integral of f on [a, b] approximated by the
    trapezoidal rule with n points."""
    return np.trapz([f(x) for x in np.linspace(a, b, n)], dx=(b - a) / n)


radius_craters = 100  # km


def crater_classify(crater, event):
    # Great circle distance between impact center and event center
    dot = np.dot(crater.loc, event.loc)
    theta = np.arccos(dot / (VENUS_RADIUS ** 2))  # radians

    distance = VENUS_RADIUS * theta

    max_distance = event.radius() + crater.radius
    min_distance = event.radius() - crater.radius

    if crater.classification == "pristine":
        if distance > max_distance:
            return "pristine"

        if max_distance >= distance >= min_distance:
            return "modified"

        if min_distance > distance:
            return "destroyed"
    elif crater.classification == "modified":
        if min_distance > distance:
            return "destroyed"

        return "modified"

    # crater.classification == "destroyed"
    return "destroyed"


def crater_contour(crater, num_points=20):
    [x, y, z] = crater.loc
    r = crater.radius

    alpha = r / VENUS_RADIUS

    points = np.array([[VENUS_RADIUS, longitude, alpha]
                       for longitude in np.linspace(0, 2 * math.pi, num_points)])

    points_rect = np.array([[VENUS_RADIUS * math.cos(longitude) * math.sin(alpha),
                             VENUS_RADIUS * math.sin(longitude) * math.sin(alpha),
                             VENUS_RADIUS * math.cos(alpha)]
                            for [rho, longitude, alpha] in points])

    longitude = theta = np.radians(crater.lon)
    latitude = np.radians(crater.lat)
    colatitude = phi = math.pi / 2 - latitude
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)

    rotation_matrix = np.array([[ct, -st * cp, st * sp],
                                [st, ct * cp,  -ct * sp],
                                [0,  sp,       cp]])

    rim_rect = np.transpose(np.matmul(rotation_matrix, np.transpose(points_rect)))
    rim_lon = np.arctan2(rim_rect[:, 1], rim_rect[:, 0])
    rim_lat = np.arcsin(rim_rect[:, 2] / VENUS_RADIUS)

    return np.column_stack([rim_lon, rim_lat])


def split_contour(contour):
    arcs = [np.array(list(group)) for key, group in groupby(contour, lambda coord: coord[0] >= 0)]


    if len(arcs) < 3:
        return [contour]

    if abs(arcs[0][-1][0]) < 1:
        return [contour]


    split = [np.row_stack([arcs[2], arcs[0]]), arcs[1]]
    print(split)
    return split



"""
def sim_yearbyyear():
    craters = []

    rng = np.random.default_rng()
    for year in range(SIM_LENGTH):
        # Test to see if an impact occurred during this year
        mean_impacts = scaling_factor * definite_integral(
            impact_rate, year, year + 1
        )
        impacts_occurred = rng.poisson(mean_impacts)

        # this tests to see if crater occurred in one time increment
        for _ in range(impacts_occurred):
            crater = Crater(
                year=year,
                loc=sample_location(),
                radius=radius_craters,
                classification="pristine",
            )
            craters.append(crater)

        resurfacing_events = events_df.where(int(events_df["year"]) == year)
        for event in resurfacing_events:
            for crater in craters:
                crater.classification = crater_classify(crater, event)

    craters_df = pd.DataFrame(craters)
    return craters_df
"""


def sim_timebins(resurfacing_events):
    craters = []

    impact_periods = [
        (resurfacing_events[i].year, resurfacing_events[i + 1].year)
        for i in range(len(resurfacing_events) - 1)
    ]

    # Want to find the scaling factor X so that the total craters at end of era checks out
    # Initialize number of craters N to 0
    # For each event area E_a and impact period length T_i,
    # Add X * T_i (or the integral of the rate curve) to N
    # Multiply N by (1 - E_a)
    unscaled_mean = 0.0
    for impact_period, event in zip(impact_periods, resurfacing_events):
        (period_start, period_end) = impact_period
        unscaled_mean += definite_integral(
            impact_rate, period_start, period_end
        )
        unscaled_mean *= 1 - event.area

    scaling_factor = NUM_IMPACTS / unscaled_mean

    # For each impact period (t_i, t_{i + 1})
    # Calculate the integral over period of r(t)
    # Generate that many craters all at once
    # Simulate the resurfacing event at t_{i + 1}
    for idx, period in enumerate(impact_periods):
        (period_start, period_end) = period
        expected_craters = scaling_factor * definite_integral(
            impact_rate, period_start, period_end
        )
        for _ in range(int(round(expected_craters))):
            # Change radius_craters to a value generated based on the impact date
            # Possibly sample a normal distribution with mean R_max * (1 - (impact_date / SIM_LENGTH))   or something
            # Could possibly interpolate between the points (0, R_max) and (SIM_LENGTH, R_min)
            # Choose standard deviation freely

            # linearly interpolate between radius(0) = R_max and radius(SIM_LENGTH) = R_min
            R_max = 500 # km
            R_min = 30  # km

            impact_date = random.uniform(period_start, period_end)
            percent_elapsed = impact_date / SIM_LENGTH
            mean_radius = (1 - percent_elapsed) * R_max + percent_elapsed * R_min
            radius = random.normalvariate(mean_radius, 5)

            if radius < 0:
                radius *= -1

            crater = Crater(
                year=impact_date,
                loc=sample_location(),
                radius=radius,
                classification="pristine",
            )
            craters.append(crater)

        resurfacing_event = resurfacing_events[idx]
        for crater in craters:
            crater.classification = crater_classify(crater, resurfacing_event)

    craters_df = pd.DataFrame(craters)
    craters_df["lon"] = [crater.longitude() for crater in craters]
    craters_df["lat"] = [crater.latitude() for crater in craters]
    return craters_df


def plot_craters(
    craters_df,
    events_df,
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

    pristine = craters_df[craters_df["classification"] == "pristine"]
    modified = craters_df[craters_df["classification"] == "modified"]
    destroyed = craters_df[craters_df["classification"] == "destroyed"]

    pristine_contours = [crater_contour(pristine, num_points=100)
                         for pristine in pristine.itertuples()]

    pristine_contours = [arc
                         for contour in pristine_contours
                         for arc in split_contour(contour)]

    
    def alpha(crater, max_alpha=1, min_alpha=0.5):
        impact_date = crater.year
        percent_elapsed = impact_date / SIM_LENGTH
        return (1 - percent_elapsed) * min_alpha + percent_elapsed * max_alpha


    pristine_collection = collections.PolyCollection(pristine_contours)

    pristine_colors = [(1, 0, 1, alpha(crater)) for crater in pristine.itertuples()]
    pristine_collection.set_color(pristine_colors)

    ax.add_collection(pristine_collection)

    modified_contours = [crater_contour(modified, num_points=100)
                         for modified in modified.itertuples()]

    modified_contours = [arc
                         for contour in modified_contours
                         for arc in split_contour(contour)]

    modified_collection = collections.PolyCollection(modified_contours)
    modified_colors = [(0, 1, 0, alpha(crater)) for crater in modified.itertuples()]
    modified_collection.set_color(modified_colors)
    ax.add_collection(modified_collection)

    destroyed_contours = [crater_contour(destroyed, num_points=100)
                          for destroyed in destroyed.itertuples()]

    destroyed_contours = [arc
                          for contour in destroyed_contours
                          for arc in split_contour(contour)]

    destroyed_collection = collections.PolyCollection(destroyed_contours)
    destroyed_colors = [(0, 0, 1, alpha(crater)) for crater in destroyed.itertuples()]
    destroyed_collection.set_color(destroyed_colors)
    ax.add_collection(destroyed_collection)

    event_contours = [crater_contour(event, num_points=100)
                      for event in events_df.itertuples()]

    event_contours = [arc
                      for contour in event_contours
                      for arc in split_contour(contour)]

    event_collection = collections.PolyCollection(event_contours)
    event_collection.set_color("red")
    event_collection.set_alpha(0.25)
    ax.add_collection(event_collection)

    """
    ax.scatter(
        np.radians(pristine["lon"]),
        np.radians(pristine["lat"]),
        color="purple",
        s=15,
    )
    ax.scatter(
        np.radians(modified["lon"]),
        np.radians(modified["lat"]),
        color="green",
        s=50,
    )
    ax.scatter(
        np.radians(destroyed["lon"]),
        np.radians(destroyed["lat"]),
        color="blue",
        s=40,
    )

    ax.scatter(
        np.radians(events_df["lon"]),
        np.radians(events_df["lat"]),
        color="red",
        s=events_df["radius"],
        alpha=0.5,
    )
    """

    ax.set_xticklabels(tick_labels)  # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("Lon")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Lat")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)


def main():
    resurfacing_events: list[ResurfacingEvent] = []
    while sum(ev.area for ev in resurfacing_events) < 1:
        this_event = ResurfacingEvent(
            year=random.uniform(0, SIM_LENGTH),
            loc=sample_location(),
            area=random.choice(RESURFACING_AREAS),
        )

        resurfacing_events.append(this_event)

    resurfacing_events = list(
        sorted(resurfacing_events, key=lambda ev: ev.year)
    )

    sim_beginning = ResurfacingEvent(year=0, loc=[0, 0, 0], area=0)
    sim_end = ResurfacingEvent(year=SIM_LENGTH, loc=[0, 0, 0], area=0)

    resurfacing_events = [sim_beginning] + resurfacing_events + [sim_end]

    events_df = pd.DataFrame(resurfacing_events)
    events_df["radius"] = [event.radius() for event in resurfacing_events]
    events_df["lon"] = [event.longitude() for event in resurfacing_events]
    events_df["lat"] = [event.latitude() for event in resurfacing_events]

    # craters_df = sim_yearbyyear()
    craters_df = sim_timebins(resurfacing_events)

    counts = Counter(craters_df["classification"])
    print(counts)

    plot_craters(craters_df, events_df)
    plt.show()


if __name__ == "__main__":
    main()
