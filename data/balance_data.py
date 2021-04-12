import pandas as pd

df = pd.read_csv('URL Classification_cleaned.csv', header=None)
# TARGETED AT KIDS: Kids  
# 3
targeted_at_kids = df[df[1] == 'Kids']
# print(len(targeted_at_kids))

# SAFE FOR KIDS: Business, Computers, Health, Home, Reference, Science, Sports
# 2
safe_for_kids = df[(df[1] == 'Business') | (df[1] == 'Computers') | (df[1] == 'Health') | (df[1] == 'Home') 
            | (df[1] == 'Reference') | (df[1] == 'Science') | (df[1] == 'Sports')]
# print(len(safe_for_kids))

# POTENTIALLY UNSAFE FOR KIDS: Arts, Shopping, Society, Recreation, News, Games 
# 1
potentially_unsafe_for_kids = df[(df[1] == 'Arts') | (df[1] == 'Shopping') | (df[1] == 'Society') 
            | (df[1] == 'Recreation') | (df[1] == 'News') | (df[1] == 'Games')]
# print(len(potentially_unsafe_for_kids))

# UNSAFE FOR KIDS: Adult 
# 0
unsafe_for_kids = df[df[1] == 'Adult']
# print(len(unsafe_for_kids))


# 35123-35123-35123-35123 (1:1:1:1)
targeted_at_kids_downsampled = targeted_at_kids.sample(n=35123, random_state=123)
targeted_at_kids_downsampled.loc[:, 1] = 3
safe_for_kids_downsampled = safe_for_kids.sample(n=35123, random_state=123)
safe_for_kids_downsampled.loc[:, 1] = 2
potentially_unsafe_for_kids_downsampled = potentially_unsafe_for_kids.sample(n=35123, random_state=123)
potentially_unsafe_for_kids_downsampled.loc[:, 1] = 1
unsafe_for_kids_downsampled = unsafe_for_kids.sample(frac=1)
unsafe_for_kids_downsampled.loc[:, 1] = 0

# print(len(targeted_at_kids_downsampled))
# print(len(safe_for_kids_downsampled))
# print(len(potentially_unsafe_for_kids_downsampled))
# print(len(unsafe_for_kids_downsampled))

total = pd.concat([targeted_at_kids_downsampled, safe_for_kids_downsampled, 
    potentially_unsafe_for_kids_downsampled, unsafe_for_kids_downsampled], ignore_index=True)
total_shuffled = total.sample(frac=1)
total_shuffled.to_csv('balanced_data_3210.csv', header=False, index=False)