#Imports
from Helpers.Mantels_Helpers import extract_all_files, create_tasks_for_parrelization, get_embeddings, stub_function_for_MARSHALL
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm



print("Imports complete.")

#Get all files
df = extract_all_files()

#Create the tasks
tasks = create_tasks_for_parrelization(df)

print("Tasks created.")

with tqdm_joblib(tqdm(total=len(tasks))): #This includes a progress bar :)
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(stub_function_for_MARSHALL)(*task, return_labels=True)
            for task in tasks
        )


#NOTE: You can command click the stub to access where its defined. 

print("All tasks complete.")