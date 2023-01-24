from money_model import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

params = {"width": 10, "height": 10, "N": range(10, 500, 10)}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
            MoneyModel,
            parameters=params,
            iterations=5,
            max_steps=100,
            number_processes=None,
            data_collection_period=1,
            display_progress=True,
    )
    results_df = pd.DataFrame(results)
    print(results_df.keys())
    

