#Imports
from Helpers.grae_pipeline_helpers import extract_all_files, GRAE_tests, create_tasks_for_parrelization
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
    with Parallel(n_jobs=-10) as parallel:
        parallel(
            delayed(GRAE_tests)(*task)
            for task in tasks
        )

#NOTE: You can command click the stub to access where its defined. 

print("All tasks complete.")

