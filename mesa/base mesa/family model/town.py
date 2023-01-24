import mesa
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

class Village(mesa.Agent):
    def __init__(self, unique_id, model, population = 100):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.population = population
    
    
class Town(mesa.Model):
    def __init__(self, width, height, N):
        self.width = width
        self.height = height
        self.N = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        # Create agents
        for i in range(self.N):
            a = Village(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        self.datacollector = mesa.DataCollector(
            model_reporters={},
            agent_reporters={"Population": "population"}
        )
    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def get_agent_count(self):
        return len(self.schedule.agents)
    
    
params = {"width": 10, "height": 10, "N": range(10, 500, 10)}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
                Town,
                parameters=params,
                iterations=5,
                max_steps=100,
                number_processes=None,
                data_collection_period=1,
                display_progress=True,
        )
    results_df = pd.DataFrame(results)
    print(results_df.keys())