# Import required library
import pandas as pd

# Import the CSV file into Python
A_data = pd.read_csv("../input/hr-ana/train.csv")
A_data = A_data.dropna()

# Directly assigning individual field columns different integer value #

# gender
A_data.gender[A_data.gender == 'm'] = 1 #male -> 1
A_data.gender[A_data.gender == 'f'] = 2 #femal -> 2
A_data["gender"] = A_data["gender"].astype('int64')

# department
A_data.department[A_data.department == 'Analytics'] = 1         #Analytics -> 1
A_data.department[A_data.department == 'Finance'] = 2           #Finance -> 2
A_data.department[A_data.department == 'HR'] = 3                #HR -> 3
A_data.department[A_data.department == 'Legal'] = 4             #Legal -> 4
A_data.department[A_data.department == 'Operations'] = 5        #Operations -> 5
A_data.department[A_data.department == 'Procurement'] = 6       #Procurement -> 6
A_data.department[A_data.department == 'R&D'] = 7               #R&D -> 7
A_data.department[A_data.department == 'Sales & Marketing'] = 8 #Sales & Marketing -> 8
A_data.department[A_data.department == 'Technology'] = 9        #Technology -> 9
A_data["department"] = A_data["department"].astype('int64')

# region
A_data.region[A_data.region == 'region_1'] = 1   #region_1 -> 1
A_data.region[A_data.region == 'region_2'] = 2   #region_2 -> 2
A_data.region[A_data.region == 'region_3'] = 3   #region_3 -> 3
A_data.region[A_data.region == 'region_4'] = 4   #region_4 -> 4
A_data.region[A_data.region == 'region_5'] = 5   #region_5 -> 5
A_data.region[A_data.region == 'region_6'] = 6   #region_6 -> 6
A_data.region[A_data.region == 'region_7'] = 7   #region_7 -> 7
A_data.region[A_data.region == 'region_8'] = 8   #region_8 -> 8
A_data.region[A_data.region == 'region_9'] = 9   #region_9 -> 9
A_data.region[A_data.region == 'region_10'] = 10 #region_10 -> 10
A_data.region[A_data.region == 'region_11'] = 11 #region_11 -> 11
A_data.region[A_data.region == 'region_12'] = 12 #region_12 -> 12
A_data.region[A_data.region == 'region_13'] = 13 #region_13 -> 13
A_data.region[A_data.region == 'region_14'] = 14 #region_14 -> 14
A_data.region[A_data.region == 'region_15'] = 15 #region_15 -> 15
A_data.region[A_data.region == 'region_16'] = 16 #region_16 -> 16
A_data.region[A_data.region == 'region_17'] = 17 #region_17 -> 17
A_data.region[A_data.region == 'region_18'] = 18 #region_18 -> 18
A_data.region[A_data.region == 'region_19'] = 19 #region_19 -> 19
A_data.region[A_data.region == 'region_20'] = 20 #region_20 -> 20
A_data.region[A_data.region == 'region_21'] = 21 #region_21 -> 21
A_data.region[A_data.region == 'region_22'] = 22 #region_22 -> 22
A_data.region[A_data.region == 'region_23'] = 23 #region_23 -> 23
A_data.region[A_data.region == 'region_24'] = 24 #region_24 -> 24
A_data.region[A_data.region == 'region_25'] = 25 #region_25 -> 25
A_data.region[A_data.region == 'region_26'] = 26 #region_26 -> 26
A_data.region[A_data.region == 'region_27'] = 27 #region_27 -> 27
A_data.region[A_data.region == 'region_28'] = 28 #region_28 -> 28
A_data.region[A_data.region == 'region_29'] = 29 #region_29 -> 29
A_data.region[A_data.region == 'region_30'] = 30 #region_30 -> 30
A_data.region[A_data.region == 'region_31'] = 31 #region_31 -> 31
A_data.region[A_data.region == 'region_32'] = 32 #region_32 -> 32
A_data.region[A_data.region == 'region_33'] = 33 #region_33 -> 33
A_data.region[A_data.region == 'region_34'] = 34 #region_34 -> 34
A_data["region"] = A_data["region"].astype('int64')

# education
A_data.education[A_data.education == "Bachelor's"] = 1       #Bachelor's -> 1
A_data.education[A_data.education == "Below Secondary"] = 2  #Below Secondary -> 2
A_data.education[A_data.education == "Master's & above"] = 3 #Master's & above -> 3
A_data["education"] = A_data["education"].astype('int64')

# recruitment_channel
A_data.recruitment_channel[A_data.recruitment_channel == 'other'] = 1    #other -> 1
A_data.recruitment_channel[A_data.recruitment_channel == 'sourcing'] = 2 #sourcing -> 2
A_data.recruitment_channel[A_data.recruitment_channel == 'referred'] = 3 #referred -> 3
A_data["recruitment_channel"] = A_data["recruitment_channel"].astype('int64')

# drop unecessary employee ID for analysis
A_data.drop(['employee_id'], axis = 1, inplace = True)

#set cleaned data to new csv file
A_data.to_csv('c_train.csv',index=False)
