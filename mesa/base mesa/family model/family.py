import mesa
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
import math


class rentingMarket(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # variables that are added to avoid errors
        self.stock = -1
        self.adults = -1
        self.children = -1
        self.lands_owned = -1
        self.productivity_land_real = -1
        self.lease_lands = -1
        self.rented_lands = -1
        
    def stage1_lease_lands(self):
        pass
    
    def stage2_calculate_rents(self):
        if self.model.total_lease_lands > 0 and self.model.total_lands_requested > 0:
            lands_rented_portion = 0
            lands_leased_portion = 0
            x = self.model.total_lease_lands/self.model.total_lands_requested
            self.model.rents_part = 0.0181*x**3-0.1691*x**2+0.5874*x-0.0563
            # limit rents_part in 0.05-0.9
            if self.model.rents_part > 0.9:
                self.model.rents_part = 0.9
            elif self.model.rents_part < 0.05:
                self.model.rents_part = 0.05
            # calculate how much lands are really rented or leased
            if self.model.total_lease_lands > self.model.total_lands_requested:
                lands_rented_portion = 1
                lands_leased_portion = self.model.total_lands_requested/self.model.total_lease_lands
            else:
                lands_rented_portion = self.model.total_lease_lands/self.model.total_lands_requested
                lands_leased_portion = 1
            # transit values to model
            self.model.lands_rented_portion = lands_rented_portion
            self.model.lands_leased_portion = lands_leased_portion

    def stage3_rent_lands(self):
        pass

    def stage4_agriculture(self):
        pass

    def stage5_household_life(self):
        pass


class Family(mesa.Agent):
    def __init__(self, unique_id, model, stock=0, adults=1, children=0, lands_owned=1):
        super().__init__(unique_id, model)
        self.stock = stock
        self.adults = adults
        self.children = children
        self.lands_owned = lands_owned
        self.productivity_land_real = self.model.productivity
        self.output = 0
        self.grown_this_step = 0
        self.lease_lands = 0
        self.rented_lands = 0
        self.landsToRent = 0
        self.landsToLease = 0
        # create numpy array to store the ages of adults
        self.arrayAdults = np.array([1], dtype=int)
        # bool to check if the family need to rent or lease, default is False
        self.boolLease = False
        self.boolRent = False

    def stage1_lease_lands(self):
        #print(self.model.schedule.time)
        # renew the values
        self.lease_lands = 0
        self.rented_lands = 0
        self.landsToRent = 0
        self.landsToLease = 0
        self.arrayStatistics()
        if self.lands_owned > self.adults:
            self.boolLease = True
            self.boolRent = False
        elif self.lands_owned < self.adults:
            self.boolRent = True
            self.boolLease = False
        else:
            self.boolLease = False
            self.boolRent = False

        if self.boolLease:
            self.landsToLease = self.lands_owned - self.adults
            self.model.total_lease_lands += self.landsToLease
        elif self.boolRent:
            self.landsToRent = self.adults - self.lands_owned
            self.model.total_lands_requested += self.landsToRent

    def stage2_calculate_rents(self):
        pass

    def stage3_rent_lands(self):
        if self.boolLease:
            self.lease_lands = self.landsToLease*self.model.lands_leased_portion
        elif self.boolRent:
            self.rented_lands = self.landsToRent*self.model.lands_rented_portion

    def stage4_agriculture(self):
        # calculate real lands to cultivate
        self.lands = self.lands_owned - self.lease_lands + self.rented_lands
        self.cultivate()
        #pay rents to rentingMarket
        if self.boolRent:
            rents = self.output*(self.rented_lands/self.lands)*self.model.rents_part
            self.model.rentingOutputs += rents
            self.stock -= rents
        

    def stage5_household_life(self):
        # collect rents from rentingMarket
        if self.boolLease:
            rents = self.model.rentingOutputs * (self.lease_lands/self.model.total_lease_lands)
            self.stock += rents
        # consume resources
        self.eat()
        self.growUp()
        self.reproduce()
        self.split()
        self.arrayStatistics()
        
    def cultivate(self):
        if self.lands >= self.adults:
            self.output = self.adults * self.model.productivity
            self.stock += self.output
        elif self.lands > 0:
            self.productivity_land_real = (
                self.model.productivity+2*math.log(self.adults/self.lands))
            self.output = self.productivity_land_real*self.lands
            self.stock += self.output

    
    def arrayStatistics(self):
        self.adults = self.arrayAdults.size

    def eat(self):
        # adults
        if self.stock >= self.adults:
            self.stock -= self.adults
        else:
            die = self.adults - round(self.stock)
            self.model.starve_adults += die
            self.model.starve_adults_step += die
            self.arrayAdults = self.arrayAdults[die:]
            self.stock = 0
        # children
        if self.stock >= self.children:
            self.stock -= self.children
        else:
            die = self.children - round(self.stock)
            self.model.starve_children += die
            self.model.starve_children_step += die
            self.children = round(self.stock)
            self.stock = 0

    def growUp(self):
        # adults gain ages
        self.arrayAdults += 1
        # children become adults
        self.adults += self.children
        arrToAdd = np.ones(self.children, dtype=int)
        self.arrayAdults = np.concatenate(
            (self.arrayAdults, arrToAdd), dtype=int)
        self.grown_this_step = self.children
        self.children = 0
        # adults die because of old age
        original_size = self.arrayAdults.size
        self.arrayAdults = self.arrayAdults[self.arrayAdults < 7]
        self.model.natural_deaths_step += original_size - self.arrayAdults.size

    def reproduce(self):
        if self.stock >= 1:
            # calculate number of adults whose ages are in [1, 3]
            adults_in_range = self.arrayAdults[(self.arrayAdults >= 1) & (
                self.arrayAdults <= 3)].size
            self.children += max(1, round(adults_in_range/2))

    # split the family into two when the number of adults is too large
    def split(self):
        if self.adults > 6:
            self.model.num_agents += 1
            self.stock = self.stock/2
            self.children = round(self.children/2)
            self.lands_owned = self.lands_owned/2
            np.random.shuffle(self.arrayAdults)
            new_family_arrayAdults = self.arrayAdults[0:round(self.adults/2)]
            self.arrayAdults = self.arrayAdults[round(self.adults/2)+1:]
            new_family = Family(self.model.num_agents, self.model, stock=self.stock,
                                adults=self.adults, children=self.children, lands_owned=self.lands_owned)
            new_family.arrayAdults = new_family_arrayAdults
            self.model.schedule.add(new_family)


def productivityUpdate(model, improvement):
    model.productivity = model.productivity * (1+improvement)
    return model.productivity


def averageLaborProductivity(model):
    num_adults = 0
    num_output = 0
    for agents in model.schedule.agents:
        if isinstance(agents, Family):
            num_adults += agents.adults - agents.grown_this_step
            num_output += agents.output
    return num_output/num_adults

# make step data 0 again


def re0(model):
    model.starve_adults_step = 0
    model.starve_children_step = 0
    model.natural_deaths_step = 0
    model.total_lease_lands = 0
    model.total_lands_requested = 0
    model.total_lands_rented = 0
    model.rentingOutput = 0

# calculate popualtion of each step


def num_adults(model):
    num_adults = 0
    for agents in model.schedule.agents:
        if isinstance(agents, Family):
            num_adults += agents.adults
    return num_adults


def num_children(model):
    num_children = 0
    for agents in model.schedule.agents:
        if isinstance(agents, Family):
            num_children += agents.children
    return num_children


class FamilyModel(mesa.Model):
    def __init__(self, productivity_start, num_agents_satrt, improvement):
        self.num_agents = num_agents_satrt
        self.productivity = productivity_start
        model_stage = ["stage1_lease_lands", "stage2_calculate_rents",
                       "stage3_rent_lands", "stage4_agriculture", "stage5_household_life"]
        self.schedule = mesa.time.StagedActivation(
            self, stage_list=model_stage, shuffle=True)
        self.running = True
        self.improvement = improvement
        self.starve_adults = 0
        self.starve_adults_step = 0
        self.starve_children = 0
        self.starve_children_step = 0
        self.natural_deaths_step = 0
        # renting market
        self.total_lease_lands = 0
        self.total_lands_requested = 0
        self.total_lands_rented = 0
        self.rents_part = 0
        self.lands_rented_portion = 0
        self.lands_leased_portion = 0
        self.rentingOutputs = 0
        # Create families with random starting lands
        for i in range(self.num_agents):
            a = Family(i, self, lands_owned=self.random.randint(1, 10))
            self.schedule.add(a)
        # create rentingMarket agents
        self.num_agents += 1
        self.schedule.add(rentingMarket(self.num_agents,self))
        # data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={"agent_count": lambda m: m.schedule.get_agent_count(), 'productivity': 'productivity', 'average_productivity': averageLaborProductivity, 'starve_adults': 'starve_adults', 'starve_children': 'starve_children',
                             'starve_adults_step': 'starve_adults_step', 'starve_children_step': 'starve_children_step', 'num_adults': num_adults, 'num_children': num_children, 'natural_deaths_step': 'natural_deaths_step'},
            agent_reporters={"stock": "stock",
                             "adults": "adults",
                             "children": "children",
                             "lands_owned": "lands_owned",
                             'productivity_land_real': 'productivity_land_real',
                             'rented_lands': 'rented_lands',
                             'lease_lands': 'lease_lands'}
        )

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        productivityUpdate(self, self.improvement)
        # collect data
        self.datacollector.collect(self)
        # make step data 0 again
        re0(model=self)


params = {"productivity_start": 3, "num_agents_satrt": 10, "improvement": 0.1}

if __name__ == '__main__':
    freeze_support()
    results = mesa.batch_run(
        FamilyModel,
        parameters=params,
        number_processes=1,
        iterations=1,
        data_collection_period=1,
        max_steps=50,
        display_progress=True,
    )
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df.adults != -1]
    print(results_df.keys())
    results_df.to_csv('results.csv')
