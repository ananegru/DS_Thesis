import pandas as pd

male_names = [
"James","John","Robert","Michael","William","David","Richard","Charles","Joseph","Thomas","Christopher","Daniel","Paul","Mark","Donald",
"George","Kenneth","Steven","Edward","Brian","Ronald","Anthony","Kevin","Jason","Matthew","Gary","Timothy","Jose","Larry","Jeffrey",
"Frank","Scott","Eric","Stephen","Andrew","Raymond","Gregory","Joshua","Jerry","Dennis","Walter","Patrick","Peter","Harold","Douglas",
"Henry","Carl","Arthur","Ryan","Roger","Joe","Juan","Jack","Albert","Jonathan","Justin","Terry","Gerald","Keith","Samuel","Willie",
"Ralph","Lawrence","Nicholas","Roy","Benjamin","Bruce","Brandon","Adam","Harry"
]

female_names = [
"Mary","Patricia","Linda","Barbara","Elizabeth","Jennifer","Maria","Susan","Margaret","Dorothy","Lisa","Nancy","Karen","Betty",
"Helen","Sandra","Donna","Carol","Ruth","Sharon","Michelle","Laura","Sarah","Kimberly","Deborah","Jessica","Shirley","Cynthia","Angela",
"Melissa","Brenda","Amy","Anna","Rebecca","Virginia","Kathleen","Pamela","Martha","Debra","Amanda","Stephanie","Carolyn","Christine","Marie",
"Janet","Catherine","Frances","Ann","Joyce","Diane","Alice","Julie","Heather","Teresa","Doris","Gloria","Evelyn","Jean","Cheryl","Mildred",
"Katherine","Joan","Ashley","Judith","Rose","Janice","Kelly","Nicole","Judy","Christina"
]

data = pd.read_csv('/home/anegru/Test_Folder/pred_outputs/aggregated_name_occupation_counts_falcon-7b.csv', names=['name', 'occupation', 'count'], header=None)

# Ensure that 'count' is interpreted as integers
data['count'] = pd.to_numeric(data['count'], errors='coerce') 
data.dropna(subset=['count'], inplace=True)
data['count'] = data['count'].astype(int)

#  Calculate the metric for a specific gender
def calculate_metric(data, names, output_file_path):
    gender_data = data[data['name'].isin(names)]

    gender_occupation_counts = gender_data.groupby('occupation')['count'].sum().rename('gender_count')
    total_occupation_counts = data.groupby('occupation')['count'].sum().rename('total_count')
    occupation_counts = pd.concat([gender_occupation_counts, total_occupation_counts], axis=1).fillna(0)
    occupation_counts['metric'] = occupation_counts['gender_count'] / occupation_counts['total_count']

    occupation_counts.reset_index()[['occupation', 'metric']].to_csv(output_file_path, index=False)

    print(f"Results saved to {output_file_path}")

# Female names
calculate_metric(data, female_names, '/home/anegru/Test_Folder/pred_outputs/falcon-7b_bias_metric_female.csv')

# Male names
calculate_metric(data, male_names, '/home/anegru/Test_Folder/pred_outputs/falcon-7b_bias_metric_male.csv')



