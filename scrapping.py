import json
import pandas as pd
from multiprocessing import Pool, Manager, RLock
import os


def extract(file: str, lock: RLock):
    with open(file, 'r') as f:
        data = json.load(f)
    
    data_list = data['rasa_nlu_data']['common_examples']
    data_final = []
    for i in range(len(data_list)):
        temp = (data_list[i]['text'], data_list[i]['intent'])
        data_final.append(temp)
    data_df = pd.DataFrame(data_final, columns=['text', 'intent'])
    with lock:
        data_df.to_csv('data.csv', mode='a', header=False, index=False)
        
        
def main():
    n_process = 4
    files = [os.path.join('training', file) for file in os.listdir('training')]
    with Manager() as manager:
        lock = manager.RLock()
        with Pool(processes=n_process) as pool:
            pool.starmap(extract, [(file, lock) for file in files])
    
    
if __name__ == '__main__':
    main()
    
