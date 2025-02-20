#Imports
from Helpers.Mantels_Helpers import extract_all_files, mantel_test, create_tasks_for_parrelization
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import random

print("Imports complete.")

#Get all files
df = extract_all_files()

#Create the tasks
tasks = create_tasks_for_parrelization(df)

random.shuffle(tasks)

print("Tasks created.")

with tqdm_joblib(tqdm(total=len(tasks))): #This includes a progress bar :)
    with Parallel(n_jobs=-5) as parallel:
        parallel(
            delayed(mantel_test)(*task)
            for task in tasks
        )

#NOTE: You can command click the stub to access where its defined. 

print("All tasks complete.")

