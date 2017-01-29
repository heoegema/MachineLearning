import pandas as pd
import numpy as np
import warnings
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', color_codes=True)


warnings.filterwarnings('ignore');

#Importing data
guns = pd.read_csv('./guns.csv', index_col=0)
print(guns.shape)
print(guns.head())
guns.index.name = 'Index'

#Viewing first 5 entries of data
print(guns.head())

#Getting column names
guns.columns = map(str.capitalize, guns.columns)
print(guns.columns)

#Viewing data types for each column
print(guns.dtypes)

#Viewing how many null values exist per column
#Incompleteness/completeness....which columns are useful?
print(guns.notnull().sum())

#Same thing as before but as a percentage
print(guns.notnull().sum() * 100.0/guns.shape[0])

#Organizing our data
print(guns.sort_values(['Year', 'Month'], inplace=True))

print(guns.head(10))

print(guns.Intent.value_counts(ascending=False))

#Removing any entries that have a null entry
print(guns.Intent.value_counts(ascending=False, dropna=False, normalize=True))

#looking at distributions of age + education
cols = ['Education', 'Age']
for col in cols:
    print(col + ':')
    print(guns[col].describe())
    print('-' * 20 + '\n')
#Percntiles
percentiles = np.arange(0.1, 1.1, 0.1)
for col in cols:
    print(col + ':')
    print(guns[col][guns[col].notnull()].describe(percentiles=percentiles))
    print('-'*20+ '\n')

#1841, 10 (1841 gun deaths under 16)
print(guns[guns['Age'] < 16].shape)

#Viewing first 20 entries of children killed by guns under 16
print(guns[guns['Age']< 16].head(20))

#filling in missing education fields
print(guns[(guns['Age'] < 16) & ((guns['Education'].isnull()) | (guns['Education'] == 5.0))].head())

#
index_temp = guns[(guns['Age'] < 16) &
                  ((guns['Education'].isnull()) | (guns['Education'] == 5.0))].index
guns.loc[index_temp, 'Education'] = 1.0
print(guns[guns.Education.isnull()].shape)

#If under 5 assume less than elementary education education (0.0)

index_temp = guns[(guns['Age'] < 5)].index
guns.loc[index_temp, 'Education'] = 0.0
print(guns.Education.describe())

#Getting rid of rows with null values in education + 5.0 (unavaible)

guns.dropna(inplace=True)
guns = guns[guns.Education != 5.0]
print(guns.Education.value_counts())


#Looking at unique values in each column....

for col in guns.columns:
    if col not in ['Age', '']:
        print(col, ':', guns[col].unique())


#How many males vs females died
print(guns.Sex.value_counts())
#83883 M 14209 F

#Getting deaths for each year due to guns
print(guns.Year.value_counts(sort=False))

#Evaluating percetage change between years
n2012 = guns[2012== guns['Year']].shape[0]
print((guns.Year.value_counts(sort=False) - n2012) * 100./ n2012)

#Viewing deaths by month
print(guns.Month.value_counts(sort=True))

#Percentage change between months
nexpected_month = guns.shape[0]/12.
print((guns.Month.value_counts(sort=True) - nexpected_month) * 100./nexpected_month)

guns.sort_values(['Year', 'Month'], inplace=True)
print(guns.head())

#Creating datetime object (combining Year + Month)

#Creating a new column
guns['Date'] = pd.to_datetime((guns.Year * 10000 + guns.Month*100 +1).apply(str),format='%Y%m%d')
guns.dtypes.tail(1)

#Since we have a new column date...we don't need these anymore
del guns['Year']
del guns['Month']
print(guns.head())

month_rates = pd.DataFrame(guns.groupby('Date').size(),
columns=['Count'])
month_rates.index.to_datetime
print(month_rates.index.dtype)
print(month_rates.shape)
print(month_rates.head())

days_per_month = []
for val in month_rates.index:
    days_per_month.append(calendar.monthrange(val.year,
    val.month)[1])
month_rates['Days_per_month'] = days_per_month
print(month_rates.head())

month_rates['Average_per_day'] = month_rates['Count']*1./month_rates['Days_per_month']
print(month_rates.shape)
print(month_rates.tail())

month_rate_dict = {}
for i in range(1,13):
    bool_temp = month_rates.index.month == i
    month_average = (sum(month_rates.loc[bool_temp, 'Average_per_day']))/3.
    month_rate_dict[i] = month_average

avg_month_rate = pd.DataFrame(month_rate_dict.items(), columns=['Month', 'Value'])
avg_month_rate = pd.DataFrame.from_dict(list(month_rate_dict.items()))
avg_month_rate.columns = ['Month', 'Value']
print(avg_month_rate)

#Calculating expect values per days_per_month
nexpected_day = guns.shape[0]/(365*3 + 1.)
print(nexpected_day)
#89.5
avg_month_rate['Percent_change'] = (avg_month_rate.Value - nexpected_day) * 100./ nexpected_day
print(avg_month_rate.sort('Percent_change'))

#What percentage of cases were police officers involved in?
print(100*guns.Police.value_counts(normalize=True))

#Less than 0.1% involves the police...we can probably just delete this columns
del guns['Police']
print(guns.shape)
print(guns.head())

print(guns.Race.value_counts(sort=True, normalize=True))

#Should probably consider race percentage in the states as well

sample_guns = guns.sample(n=10000)
print(sample_guns.head())

print(sample_guns.Sex.value_counts(normalize=True))

#sorting by 'Homicide, Suicide, Accidental, Undetermined'

list_ordered = ['Homicide', 'Suicide', 'Accidental', 'Undetermined']
guns['Intent'] = guns['Intent'].astype('category')
guns.Intent.cat.set_categories(list_ordered, inplace=True)
print(guns.sort_values(['Intent']).head())

#Dropping undetermined values

guns = guns[guns.Intent !='Undetermined']
print(guns.Intent.value_counts())

#Removing the value altogether
list_ordered = list_ordered[:-1]
guns.Intent.cat.set_categories(list_ordered, inplace=True)
print(guns.Intent.value_counts())

print(guns.Race.str.len().unique())

#Getting all the races included in the data set
print(guns.Race.unique())

#Data Visualization

#Line chart
# years 2012 - 2014
fig = plt.figure()
plt.plot(month_rates.index.month[0:12], month_rates['Count'][0:12], label='2012',
        linestyle='-', linewidth=2., alpha=0.6)
plt.plot(month_rates.index.month[12:24], month_rates['Count'][12:24], label='2013',
        linestyle='-', linewidth=2., alpha=0.6, color='r')
plt.plot(month_rates.index.month[24:36], month_rates['Count'][24:36], label='2014',
        linestyle='-', linewidth=2., alpha=0.6, color='g')
plt.xlim(xmin=1, xmax=12)
plt.ylim(ymin=0, ymax=max(month_rates['Count'])+100)
plt.tick_params(axis='both', which='both',length=0)
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc='center right', frameon=False)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Gun Death\nCount', fontsize=14)
plt.title('Monthly Gun Death Count in the US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()
