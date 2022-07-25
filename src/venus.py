#!/usr/bin/env python
# coding: utf-8

# In[4]:

#### Hello Future Tutor! This is my code so far. Background is that only data on Venus
#### is radar imagery of its surface, but we don't know what happened for it to look like
### it does today over its 4.5Byr history. Astrophysicists usually measure surface evolution
#### by observing the craters and to what degree they are destroyed by thermal processes
#### like volcanic explosions. This is what I'm attempting to do.

#### My model is essentially observing two constant processes, crater creation (when asteroids hit)
#### and crater destruction (when a resurfacing event occurs and cover craters with lava/rock).
#### The constant input values are from NASA, where I'm working. The model goes kinda year-by-year like this;
###  at the beginning of history, x number of craters were emplaced. Then at time 2, x number
### of resurfacing events occurred that could either leave craters pristine as before, modified, or destroyed.
### Now, reclassify the craters emplaced prior to the event as one of those 3 categories based on the distance between
### it and resurfacing event/the radius of the impacted area, and move on to time 3. Repeat until 4.5Byr is over.

### this is very basic and rough draft.Not really doing anything that helpful to NASA yet.

### goal is to have results that could be compared to what we have observed on Venus; the 2 observational
### constraints include (1) Venus has ~1000 craters (not including destroyed ones because we cant see them),
### (2) only ~175 craters are modified craters.

### I'm now trying to change my model to be more efficient, or simply perfroming the same task in a better way.
###This could mean splitting into time bins instead of year-by-year. I also want to incorporate some type of
### geometric series so that we can always end up with 1000 craters on the planet, but the amount emplaced throughout
### the history differs by test run, depending on how many were destroyed.


from __future__ import division
import random
import matplotlib.pyplot as plt
import math
import numpy as np

num_simulations = 1
max_num_years = 45000000  # years
Venus_radius = 6051.8  # km
Venus_SurfaceArea = 4 * math.pi * (Venus_radius**2)  # km^2
num_impacts = 1000

# randomly select the number of expected resurfacing events for each test run
num_expectedresurfacing_list = [2, 4, 5, 10, 100, 143, 1000, 10000]
num_expectedresurfacing = random.choice(num_expectedresurfacing_list)

resurfacing_rates = [0.1, 0.05, 0.01, 0.007, 0.001, 0.0001]

##incremental resurfacing area percentages, global area is the surface area of sphere of r=1
resurfacingarea = [
    (0.1 * Venus_SurfaceArea),
    (0.05 * Venus_SurfaceArea),
    (0.01 * Venus_SurfaceArea),
    (0.007 * Venus_SurfaceArea),
    (0.001 * Venus_SurfaceArea),
    (0.0001 * Venus_SurfaceArea),
]
##deriving list of the possible radii for resurfacing, necessary for crater classification
radius_resurfacing_list = [
    math.sqrt((0.1 * Venus_SurfaceArea) / math.pi),
    math.sqrt((0.05 * Venus_SurfaceArea) / math.pi),
    math.sqrt((0.01 * Venus_SurfaceArea) / math.pi),
    math.sqrt((0.007 * Venus_SurfaceArea) / math.pi),
    math.sqrt((0.001 * Venus_SurfaceArea) / math.pi),
    math.sqrt((0.0001 * Venus_SurfaceArea) / math.pi),
]

##randomly selecting radius of resurfacing event from the list
rand_idx = random.randint(0, len(radius_resurfacing_list) - 1)
radius_resurfacing = radius_resurfacing_list[rand_idx]


##radius in the paper is 15km, but im working with sphere of radius 1, so 15km/Venus circumference is this
radius_craters = 15  # km

##function to find the rates of impact
def impact_rate(num_impacts, max_num_years):
    return num_impacts / max_num_years


##gives probability of impact event ocurring in a given year
def impact(ratecrater):
    return random.uniform(0, 1) <= ratecrater


##function defining rate of resurfacing
def resurfacing_rate(num_expectedresurfacing, max_num_years):
    # this is selecting from the random # of expected resuirfacing event list
    rand_idx2 = random.randint(0, len(num_expectedresurfacing_list) - 1)
    num_expectedresurfacing = num_expectedresurfacing_list[rand_idx2]
    # this gives the probability of a resurfaicng event occurring in a given year
    rateresurfacing = num_expectedresurfacing / (max_num_years)
    print("Number of Expected Resurfacing Events " + str(num_expectedresurfacing))
    return rateresurfacing


def resurfacing(rateresurfacing):
    randomnum = random.uniform(0, 1)
    if randomnum <= rateresurfacing:
        return True
    else:
        return False


def sample_craters():
    vec = []
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    xx = Venus_radius * np.sin(phi) * np.cos(theta)
    yy = Venus_radius * np.sin(phi) * np.sin(theta)
    zz = Venus_radius * np.cos(phi)
    vec.append(str(xx))
    vec.append(str(yy))
    vec.append(str(zz))
    vec = ",".join(vec)
    return vec


def sample_resurfacingevent():
    vec = []
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    xx = Venus_radius * np.sin(phi) * np.cos(theta)
    yy = Venus_radius * np.sin(phi) * np.sin(theta)
    zz = Venus_radius * np.cos(phi)
    vec.append(str(xx))
    vec.append(str(yy))
    vec.append(str(zz))
    vec = ",".join(vec)
    return vec


def select_radius():
    ##randomly selecting radius of resurfacing event from the list
    rand_idx = random.randint(0, len(radius_resurfacing_list) - 1)
    radius_resurfacing = radius_resurfacing_list[rand_idx]
    return radius_resurfacing


##for when i use function need to put [j] next to list_of
def crater_classify(
    list_of_impactlocations, list_of_craterradii, eventlocation, event_radius_selected
):
    ##finding distance along surface, radius of sphere is 1 here
    # split each index into 3 strings
    impactcoordinates = list_of_impactlocations.split(",")
    eventcoordinates = eventlocation.split(",")
    xi = float(impactcoordinates[0])
    yi = float(impactcoordinates[1])
    zi = float(impactcoordinates[2])

    xj = float(eventcoordinates[0])
    yj = float(eventcoordinates[1])
    zj = float(eventcoordinates[2])

    delta = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    if delta > (2 * Venus_radius):
        return "pristine"
    else:
        phi = math.asin((delta / 2) / Venus_radius)
        distance = 2 * phi * Venus_radius
        if distance > (2 * list_of_craterradii) + event_radius_selected:
            return "pristine"
        if distance >= (event_radius_selected - 2 * list_of_craterradii) and (
            distance <= (2 * list_of_craterradii + event_radius_selected)
        ):
            return "modified"
        else:
            if list_of_craterradii > event_radius_selected:
                return "pristine"
            else:
                return "destroyed"


##this variable is givong nthe retruned rate for impact
rate_impact = impact_rate(num_impacts, max_num_years)
print("Impact Rate " + str(rate_impact))

##this variable is giving the returned resurfacing rate
rate_resurfacing = resurfacing_rate(num_expectedresurfacing, max_num_years)
print("Resurfacing Rate " + str(rate_resurfacing))

max_num_years = 45000000

##num_impacts is set down to 0 since it was used as 1000 in rate determination
I = 0
num_impacts = 0
num_resurface = 0
list_of_impactlocations = []
list_of_eventlocations = []
list_of_craterclass = []
list_of_craterradii = []
list_of_resurfacingradii = []
coo = []
coo2 = []
coord = []
##keeping track of the number of resurfacing events and craters
while I < max_num_years:
    # this tests to see if crater occurred in one time increment
    impacts = impact(rate_impact)
    if impacts == True:
        num_impacts += 1
        impactlocation = sample_craters()
        list_of_impactlocations.append(impactlocation)
        coo.append(impactlocation.split(","))
        # when craters are emplaced, classified as pristine, and indexes of locations and class correspond to same crater
        list_of_craterclass.append("pristine")
        ##append the radius (the number) of that crater
        list_of_craterradii.append(radius_craters)
        with open("craterinfo8.txt", "a") as f:
            f.writelines(
                " In Year " + str(I) + " Impact Occurred at Location " + impactlocation
            )
            f.write("\n")
    else:
        impacts = impacts
    resurface = resurfacing(rate_resurfacing)
    if resurface == True:
        num_resurface += 1
        eventlocation = sample_resurfacingevent()
        list_of_eventlocations.append(eventlocation)
        coo2.append(eventlocation.split(","))
        # write function outside that shows which radius was selected --need return statement!
        # call the function
        event_radius_selected = select_radius()
        # append the radius of the event to list of radii
        list_of_resurfacingradii.append(event_radius_selected)
        with open("resurfacinginfo8.txt", "a") as f:
            f.writelines(
                " In Year "
                + str(I)
                + " Resurfacing Event Occurred at Location "
                + eventlocation
                + " with radius "
                + str(event_radius_selected)
            )
            f.write("\n")
        J = 0
        while J < len(list_of_impactlocations):
            classified = crater_classify(
                list_of_impactlocations[J],
                list_of_craterradii[J],
                eventlocation,
                event_radius_selected,
            )
            list_of_craterclass[J] = classified
            coord.append(classified)
            if classified != "pristine":
                if classified != "modified":
                    # modified = classified.replace('pristine', 'modified')
                    # list_of_craterclass.append(modified)
                    with open("modificationinfo8.txt", "a") as f:
                        f.writelines(
                            "Resurfacing event at "
                            + eventlocation
                            + " destroyed crater at "
                            + list_of_impactlocations[J]
                        )
                        f.write("\n")
                else:
                    # destroyed = classified.replace('pristine', 'destroyed')
                    # list_of_craterclass.append(modified)
                    with open("modificationinfo8.txt", "a") as f:
                        f.writelines(
                            "Resurfacing event at "
                            + eventlocation
                            + " modified crater at "
                            + list_of_impactlocations[J]
                        )
                        f.write("\n")
            # else:
            # print("PristineCrater")
            J += 1
    I += 1
print("Total Resurfacing Events " + str(num_resurface))
print("Total Impact Events " + str(num_impacts))
print(list_of_craterclass)
#%%


# In[ ]:

import pandas as pd

import cartopy.crs as ccrs

resultx = [ls[0] for ls in coo]
resulty = [ls[1] for ls in coo]
resultz = [ls[2] for ls in coo]


df = pd.DataFrame(np.column_stack([resultx, resulty, resultz]), columns=["x", "y", "z"])
# x=df["x"]
# y=df["y"]
# z=df["z"]

df = df.astype(float)
df["Lon"] = np.degrees(np.arctan2(df["y"], df["x"]))
df["Lat"] = np.degrees(np.arcsin(df["z"] / Venus_radius))
df["Class"] = list_of_craterclass
print(df)

print(list_of_craterclass)

prisdf = df[df["Class"] == "pristine"]
modifydf = df[df["Class"] == "modified"]
destroydf = df[df["Class"] == "destroyed"]

print(prisdf)
print(modifydf)
print(destroydf)

priscraterLon = prisdf["Lon"].to_numpy()
# print(craterLon)
priscraterLat = prisdf["Lat"].to_numpy()
# print(craterLat)
modcraterLon = modifydf["Lon"].to_numpy()
# print(craterLon)
modcraterLat = modifydf["Lat"].to_numpy()
destroycraterLon = destroydf["Lon"].to_numpy()
# print(craterLon)
destroycraterLat = destroydf["Lat"].to_numpy()

##For Resurfacing Events

resultx2 = [ls[0] for ls in coo2]
resulty2 = [ls[1] for ls in coo2]
resultz2 = [ls[2] for ls in coo2]
df2 = pd.DataFrame(
    np.column_stack([resultx2, resulty2, resultz2]), columns=["x2", "y2", "z2"]
)
x2 = df2["x2"]
y2 = df2["y2"]
z2 = df2["z2"]

df2 = df2.astype(float)


df2["Lon"] = np.degrees(np.arctan2(df2["y2"], df2["x2"]))
df2["Lat"] = np.degrees(np.arcsin(df2["z2"] / Venus_radius))
print(df2)

df3 = pd.DataFrame(np.column_stack([list_of_resurfacingradii]), columns=["r"])

eventLon = df2["Lon"].to_numpy()

eventLat = df2["Lat"].to_numpy()

eventR = df3["r"].to_numpy()

#%% Previous Mollweides
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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
    x=df2["Lon"],
    y=df2["Lat"],
    color="blue",
    s=df3["r"],
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


#%% New Mollweides from sky prokection
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML  # To include images as HTML


def plot_craters(
    craterLon,
    craterLat,
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

    x = np.remainder(craterLon + 360 - org, 360)  # shift RA values
    xx = np.remainder(eventLon + 360 - org, 360)
    ind = x > 180
    x[ind] -= 360  # scale conversion to [-180, 180]
    x = -x  # reverse the scale: East to the left
    ind = xx > 180
    xx[ind] -= 360  # scale conversion to [-180, 180]
    xx = -xx
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + org, 360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection, facecolor="LightCyan")
    ax.scatter(np.radians(x), np.radians(craterLat))
    ax.scatter(np.radians(xx), np.radians(eventLat))  # convert degrees to radians

    ax.set_xticklabels(tick_labels)  # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("Lon")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Lat")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)


plot_craters(craterLon, craterLat, eventLon, eventLat, eventR)

#%% New Mollweides Useful

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML  # To include images as HTML


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
