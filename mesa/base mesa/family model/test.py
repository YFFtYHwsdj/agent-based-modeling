import mesa 

# Test(unique_id, model)
# 
# Parameters
# ----------
# unique_id : int
#     A unique identifier for the agent.
# model : Mesa Model
#     The model the agent is part of.
# 
# Attributes
# ----------
# unique_id : int
#     A unique identifier for the agent.
# model : Mesa Model
#     The model the agent is part of.
class Test(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
    def stage1(self):
        self.model.total += 1
        print("agent"+str(self.unique_id)+': '+str(self.model.schedule.time))
    
    def stage2(self):
        print("agent"+str(self.unique_id)+': '+str(self.model.schedule.time))
    
        
# TestModel(N)
#     Create a new TestModel instance with N agents.
# 
#     Args:
#         N (int): Number of agents
# 
#     Attributes:
#         num_agents (int): Number of agents
#         running (bool): True if the model is running, False if not
#         total (int): Total number of agents
#         schedule (mesa.time.BaseScheduler): Scheduler for the model
#         datacollector (mesa.datacollector.DataCollector): Data collector for the model
# 
#     Methods:
#         step(): Advance the model by one step
# 
#     Examples:
#         >>> model = TestModel(10)
#         >>> model.step()
#         >>> model.running
#         False
#         >>> model.total
#         10
#         >>> model.schedule.steps
#         1
#         >>> model.schedule.stage
#         'stage1'
class TestModel(mesa.Model):
    def __init__(self, N):
        self.num_agents = N
        model_stages = ['stage1', 'stage2']
        self.running = True
        self.total = 0
        self.schedule = mesa.time.StagedActivation(
            model=self,
            stage_list=model_stages,
            shuffle=False
        )
        #self.schedule = mesa.time.RandomActivation(self)
        self.datacollector = mesa.DataCollector(
            model_reporters={},
            agent_reporters = {},
        )
        for i in range(self.num_agents):
            a = Test(i, self)
            self.schedule.add(a)
            
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
params = {"N": 2}
mesa.batch_run(
            TestModel,
            parameters=params,
            iterations=1,
            max_steps=3,
            number_processes=1,
            data_collection_period=10,
            display_progress=True,
    )