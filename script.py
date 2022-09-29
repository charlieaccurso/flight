import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib.pyplot as plt
import math
# import codecademylib3

print("In this project, I analyze a large set of data from an airline. This dataset was supplied my Codecademy's learning platform. Data include flight miles, number of passengers, delays, inflight amenities, prices, etc.")
print('\n')

## Read in Data
flight = pd.read_csv("flight.csv")
print(flight.head())

## Task 1
print("We first want to examine coach ticket prices. To do so, we generate a box plot.")
print('\n')

sns.boxplot(x='coach_price', data=flight)
plt.show()
plt.clf()
print("A low-priced coach ticket is about $40, whereas a high-priced one is about $600. The average coach ticket is about $380. $500 is a relatively expensive coach ticket.")
print('\n')

## Task 2
print("Next, we plot a histogram to specifically examine flights that are 8 hours long.")
print('\n')
sns.histplot(flight.coach_price[flight.hours == 8])
plt.show()
plt.clf()

print(np.mean(flight.coach_price[flight.hours == 8]))

print("For 8-hour-long flights, a high-priced ticket is about $600. A low-priced ticket is about $175. The average ticket price is $431.83. In this case, a $500 ticket seems more reasonable.")
print('\n')

## Task 3
print("To see how flight delay times are distributed, we subset the data to only include flight delays under 50 minutes. There are very few delays longer than this.")
print('\n')

sns.histplot(flight.delay[flight.delay < 50])
plt.show()
plt.clf()
print("Flight delays of 10 minutes are the most common. There is a significant right-skew of delays lasting between 20 and 40 minutes, but delays over 50 minutes are exceptionally uncommon (They are considered outliers and not plotted in the histogram). Nevertheless, a significant number of flights experience no delays.")
print('\n')

## Task 4
print("To examine the relationship between coach and first-class prices, we examine a random 5% subset of the data, due to the large size of the dataset. We use this subset of data to generate a scatterplot and a LOWESS (Locally Weighted Scatterplot Smoothing) plot.")
print('\n')

perc= 0.05
flight_sub= flight.sample(n=int(flight.shape[0]*perc))
sns.scatterplot(flight_sub.coach_price, flight_sub.firstclass_price, alpha=0.2)
plt.show()
plt.clf()

sns.lmplot(x='coach_price', y='firstclass_price', data=flight_sub, line_kws={'color':'black'}, lowess=True)
plt.show()
plt.clf()

print("Flights with higher coach prices always have higher first-class prices. The scatterplot of prices shows two distinct clusters of price correlation centered around coach prices of about $320 and coach prices of about $400. The LOWESS plot reflects these clusters as there is a distinct increase in first-class prices at around $350 coach prices.")
print('\n')

## Task 5
print("To examine the relationship between coach prices and the presence of inflight features, we generate overlapping histograms for meals, entertainment, and WiFi.")
print('\n')

# Hist 1: inflight_meal
sns.histplot(flight, x="coach_price", hue=flight.inflight_meal)
plt.show()
plt.clf()
# Hist 2: inflight_entertainment
sns.histplot(flight, x="coach_price", hue=flight.inflight_entertainment)
plt.show()
plt.clf()
# Hist 3: inflight_wifi
sns.histplot(flight, x="coach_price", hue=flight.inflight_wifi)
plt.show()
plt.clf()

print("Higher prices are more associated with the presence of inflight meals, inflight entertainment, and inflight WiFi, as would be expected. However, among the highest coach prices, more flights actually offer meals than do not. Conversely, among the lowest coach prices, more flights do not provide inflight entertainment than do.")
print('\n')

## Task 6
print('To see how the number of passengers changes in relation to the length of flights, we create a jittered scatterplot.')
print('\n')

sns.lmplot('hours', 'passengers', data=flight_sub, x_jitter=0.15, y_jitter=0.15, fit_reg=False)
plt.show()
plt.clf()

print("It is interesting to note that no matter how long or short the flight, a flight generally either has over 190 passengers, or under 180 passengers; there is a clear split between the data points that illustrates this pattern. Additionally, there are much fewer flights of six- and eight-hour lengths, compared to other time lengths.")
print('\n')

## Task 7
print("To examine the relationship between coach and first-class prices on weekends compared to weekdays, we create a scatterplot that uses color to distinguish weekend and weekday flights.")
print('\n')

sns.lmplot(x='coach_price', y='firstclass_price', hue='weekend', data=flight_sub, fit_reg=False)
plt.show()
plt.clf()

print("It is clear from the scatterplot that weekend flights cost substantially more than weekday flights. Additionally, the clouding seen in the scatterplot comparing coach prices to first-class prices can now be explained as result of whether the flight occurs on a weekday or a weekend.")
print('\n')

## Task 8
print("Finally, to determine how coach prices differ for redeyes and non-redeyes on each day of the week, we can create a set of side-by-side boxplots.")
print('\n')

days_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(x='day_of_week', y='coach_price', hue='redeye', palette='bright', data=flight, order=days_order)
plt.xticks(rotation=30)
plt.show()
plt.clf()

print("The side-by-side box plots clearly show that redeye flights are significantly cheaper than non-redeye flights on all days of the week. Nevertheless, both redeye and non-redeye flights are more expensive on the weekend than on weekdays.")


