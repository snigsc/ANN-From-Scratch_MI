#TEAM: 0045_0190_1143
# preprocessing.py creates a clean dataset named 'Clean_LBW_Dataset.csv'

import pandas as pd
import numpy as np

# Outlier detection: generating z-score
def z_score(df):
    df.columns = [x + "_zscore" for x in df.columns.tolist()]
    return ((df - df.mean())/df.std(ddof=0))


# Cleaning the dataset
def pre_processing(d):
    # Missing Education values replaced by 5.0 
    # Reason: All non-null values of Education are 5.0
    d['Education'] = d['Education'].fillna(5.0)
    
    # Outlier detection: printing rows where zscore of any attribute > 3
    dz = d.copy()
    dz = z_score(dz)

    #print('Weight outliers: ',len(dz.loc[abs(dz['Weight_zscore']) > 3]))
    #print('HB outliers: ',len(dz.loc[abs(dz['HB_zscore']) > 3]))
    #print('IFA outliers: ',len(dz.loc[abs(dz['IFA_zscore']) > 3]))
    #print('BP outliers: ',len(dz.loc[abs(dz['BP_zscore']) > 3]))
    # Using the above print statements, we evaluated the number of outliers for each of the specified attributes. 

    # There was 1 outlier in BP which we rectified as follows.
    # Replacing BP with the median of the column (calculated excluding the value whose Z score is greater than 3)
    bp_median = d['BP'][abs(dz['BP_zscore'])<3].median()
    d.loc[dz['BP_zscore']>3,'BP'] = np.nan
    d['BP'].fillna(bp_median,inplace=True)

    # Filling missing 'Age' values with the mean of the column (excluding missing cells)
    age_mean = d['Age'].mean(skipna=True)
    d['Age'].fillna(age_mean,inplace=True)

    # Filling missing 'Weight' values with the median of the column (excluding missing cells)
    weight_median = d['Weight'].median(skipna=True)
    d['Weight'].fillna(weight_median,inplace=True)

    # Filling missing 'Delivery phase' values with the mode of the column
    # Reason: Delivery phase is a binary attribute (1 or 2)
    d['Delivery phase'] = d['Delivery phase'].fillna(d['Delivery phase'].mode()[0])

    # Filling missing 'Residence' values with the mode of the column
    # Reason: Residence is a binary attribute (1 or 2)
    d['Residence'] = d['Residence'].fillna(d['Residence'].mode()[0])

    # Filling missing 'HB' values with mean of 'HB' values for the same 'Delivery phase'
    # Reason: Pregnancy has various effects on hematologic parameters. Hemoglobin (Hb) levels decrease during the first trimester, reaching minimum values in the late second trimester and tend to increase during the third trimester of pregnancy. Hence, BP is highly dependent on delivery phase.
    # Average HB for Delivery phase=1 is 9.086667, Delivery phase=2 is 8.400000
    d['HB'] = d.groupby('Delivery phase')['HB'].apply(lambda x: x.fillna(x.mean()))
    
    return d

df = pd.read_csv('LBW_Dataset.csv')
df = pre_processing(df)
df.to_csv(r'Clean_LBW_Dataset.csv',index=False)