import pandas as pd

# df = pd.read_csv('data/URL Classification.csv', header=None, usecols=[1, 2])
# df.drop_duplicates(inplace=True)
# df.dropna(inplace=True)
# df.to_csv('URL Classification_cleaned.csv', header=False, index=False)

df = pd.read_csv('URL Classification_cleaned.csv', header=None)
# TARGETED AT KIDS: Kids  
# 2
targeted_at_kids = df[df[1] == 'Kids']
# SAFE FOR KIDS: Business, Computers, Health, Home, Reference, Science, Sports
# 1
safe_for_kids = df[(df[1] == 'Business') | (df[1] == 'Computers') | (df[1] == 'Health') | (df[1] == 'Home') 
            | (df[1] == 'Reference') | (df[1] == 'Science') | (df[1] == 'Sports')]
# POTENTIALLY UNSAFE FOR KIDS: Arts, Shopping, Society, Recreation, News, Games 
# -1
potentially_unsafe_for_kids = df[(df[1] == 'Arts') | (df[1] == 'Shopping') | (df[1] == 'Society') 
            | (df[1] == 'Recreation') | (df[1] == 'News') | (df[1] == 'Games')]
# UNSAFE FOR KIDS: Adult 
# -2
unsafe_for_kids = df[df[1] == 'Adult']

# 5000-25000-25000-5000 (1:5:5:1)
targeted_at_kids_downsampled = targeted_at_kids.sample(n=5000, random_state=123)
targeted_at_kids_downsampled.loc[:, 1] = 2
safe_for_kids_downsampled = safe_for_kids.sample(n=25000, random_state=123)
safe_for_kids_downsampled.loc[:, 1] = 1
potentially_unsafe_for_kids_downsampled = potentially_unsafe_for_kids.sample(n=25000, random_state=123)
potentially_unsafe_for_kids_downsampled.loc[:, 1] = -1
unsafe_for_kids_downsampled = unsafe_for_kids.sample(n=5000, random_state=123)
unsafe_for_kids_downsampled.loc[:, 1] = -2

total = pd.concat([targeted_at_kids_downsampled, safe_for_kids_downsampled, 
    potentially_unsafe_for_kids_downsampled, unsafe_for_kids_downsampled], ignore_index=True)
total_shuffled = total.sample(frac=1)
total_shuffled.to_csv('1_5_5_1_60000.csv', header=False, index=False)

# 5000-75000-75000-5000 (1:15:15:1)
targeted_at_kids_downsampled = targeted_at_kids.sample(n=5000, random_state=123)
targeted_at_kids_downsampled.loc[:, 1] = 2
safe_for_kids_downsampled = safe_for_kids.sample(n=75000, random_state=123)
safe_for_kids_downsampled.loc[:, 1] = 1
potentially_unsafe_for_kids_downsampled = potentially_unsafe_for_kids.sample(n=75000, random_state=123)
potentially_unsafe_for_kids_downsampled.loc[:, 1] = -1
unsafe_for_kids_downsampled = unsafe_for_kids.sample(n=5000, random_state=123)
unsafe_for_kids_downsampled.loc[:, 1] = -2

total = pd.concat([targeted_at_kids_downsampled, safe_for_kids_downsampled, 
    potentially_unsafe_for_kids_downsampled, unsafe_for_kids_downsampled], ignore_index=True)
total_shuffled = total.sample(frac=1)
total_shuffled.to_csv('1_15_15_1_160000.csv', header=False, index=False)

# 20000-20000-20000-20000 (1:1:1:1)
targeted_at_kids_downsampled = targeted_at_kids.sample(n=20000, random_state=123)
targeted_at_kids_downsampled.loc[:, 1] = 2
safe_for_kids_downsampled = safe_for_kids.sample(n=20000, random_state=123)
safe_for_kids_downsampled.loc[:, 1] = 1
potentially_unsafe_for_kids_downsampled = potentially_unsafe_for_kids.sample(n=20000, random_state=123)
potentially_unsafe_for_kids_downsampled.loc[:, 1] = -1
unsafe_for_kids_downsampled = unsafe_for_kids.sample(n=20000, random_state=123)
unsafe_for_kids_downsampled.loc[:, 1] = -2

total = pd.concat([targeted_at_kids_downsampled, safe_for_kids_downsampled, 
    potentially_unsafe_for_kids_downsampled, unsafe_for_kids_downsampled], ignore_index=True)
total_shuffled = total.sample(frac=1)
total_shuffled.to_csv('1_1_1_1_80000.csv', header=False, index=False)


