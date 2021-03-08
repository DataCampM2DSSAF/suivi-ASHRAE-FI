import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from tqdm import tqdm_notebook as tqdm
import datetime
from sklearn import metrics
import gc
import os


def fill_weather_dataset(weather_df):
    
    # Add Day,Week & Month features This dataset consits of hourly weather information. 
    # So we are going to fill missing values based on below new date features.
    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    # adding day ,moth and week as features
    weather_df["day"] = weather_df["datetime"].dt.day 
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])
     
    # fill missing air temperature with mean temperature of day of the month. 
    # Each month comes in a season and temperature varies lots in a season. So filling with yearly mean value is not a good idea.
    
    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)


    #Data is missing for most of days and even many consecutive days. So, first, calculate mean cloud_coverage of day of the month 
    # then fill rest missing values with last valid observation.(fillna with the method='ffill')
    
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"]) # imputing with daily means per site id
    weather_df.update(cloud_coverage_filler,overwrite=False)

    #fillna with the method='ffill' option. 'ffill' stands for 'forward fill' and will propagate last valid observation forward
    
    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)  

    
    # Data is missing for most of days and even many consecutive days. So, first, calculate mean sea_level of day of the month 
    # then fill rest missing values with last valid observation.(fillna with the method='ffill')
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])
    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)


    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])
    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)


    return weather_df




def calculate_rh(df):
  df['relative_humidity'] = 100 * (np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp((17.625 * df['air_temperature'])/(243.04 + df['air_temperature'])))
  return df

def calculate_fl(df):
  flike_final = []
  flike = []
  # calculate Feels Like temperature
  for i in range(len(df)):
      at = df['air_temperature'][i]
      rh = df['relative_humidity'][i]
      ws = df['wind_speed'][i]
      flike.append(feels_like(Temp(at, unit = 'C'), rh, ws))
  for i in range(len(flike)):
      flike_final.append(flike[i].f)
  df['feels_like'] = flike_final
  del flike_final, flike, at, rh, ws

def get_meteorological_features(data):
  data = calculate_rh(data)
  data = calculate_fl(data)
  return data


def features_engineering(df):
    
  # Sort by timestamp
  df.sort_values("timestamp")
  df.reset_index(drop=True)
  
  # Add more features
  df["timestamp_2"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
  df["hour"] = df["timestamp_2"].dt.hour
  df["dayofweek"] = df["timestamp_2"].dt.weekday

  df['group'] = df['timestamp_2'].dt.month
  df['group'].replace((1, 2, 3, 4), 1, inplace = True)
  df['group'].replace((5, 6, 7, 8), 2, inplace = True)
  df['group'].replace((9, 10, 11, 12), 3, inplace = True)
  df = df.drop('timestamp_2', 1)
  # df["timestamp"] = pd.to_datetime(df["timestamp"], utc = True)
  return df



