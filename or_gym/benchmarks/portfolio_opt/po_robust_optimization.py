import numpy as np 
from pyomo.environ import *
from pyomo.opt import SolverFactory 

## Get Robust Solution 

ro = ConcreteModel()

investment_horizon = 10

assets = [a+1 for a in range(3)]
periods = [p for p in range(investment_horizon+1)]

#Parameters 
initial_cash = 150
initial_assets = [0, 0, 0]
buy_cost = [0.045, 0.025, 0.035]
sell_cost = [0.040, 0.020, 0.030]
asset1mean = np.array([1.25, 1.25, 2, 4, 5, 3, 2, 3, 6, 9, 7]).reshape(1, -1) #up and down all the way 
asset2mean = np.array([5, 5, 3, 2, 2, 1.25, 4, 5, 6, 7, 8]).reshape(1, -1) #down intially then up 
asset3mean = np.array([3, 3, 5, 6, 9, 10, 8, 4, 2, 1.25, 4]).reshape(1, -1) #up initially then down 
asset_prices_means = np.vstack([asset1mean, asset2mean, asset3mean])
asset_prices_variance = np.ones(3) * 0.45 #could use covariance matrix here instead 

#Variables 
ro.Cash = Var(periods, domain=NonNegativeReals)
ro.Cash[0].fix(initial_cash)
ro.Holdings = Var(assets, periods, domain=NonNegativeReals)
for a in assets: 
	ro.Holdings[a,0].fix(initial_assets[0])
ro.Asset_Sell = Var(assets, periods[1:], domain=NonNegativeReals)
ro.Asset_Buy = Var(assets, periods[1:], domain=NonNegativeReals)
ro.T = Var()

#Constraints 
def PortfolioValueConstraint(ro): 
	return ro.T <= ro.Cash[investment_horizon] + \
	sum(asset_prices_means[a-1][-1]*ro.Holdings[a,investment_horizon] for a in assets) \
	- 3*(sum(ro.Holdings[a,investment_horizon]*asset_prices_variance[a-1]*\
	ro.Holdings[a,investment_horizon] for a in assets))**0.5

def CashAccounting(ro, p): 
	return ro.Cash[p] - ro.Cash[p-1] <= \
	sum(asset_prices_means[a-1][p]*((1-sell_cost[a-1])*ro.Asset_Sell[a,p] - (1+buy_cost[a-1])*ro.Asset_Buy[a,p]) \
		for a in assets) \
	- 3*(sum((1-sell_cost[a-1])*ro.Asset_Sell[a,p] - (1+buy_cost[a-1])*ro.Asset_Buy[a,p] for a in assets)**2 \
		)**0.5

def AssetBalance(ro, a, p): 
	return ro.Holdings[a,p] == ro.Holdings[a,p-1] \
	- ro.Asset_Sell[a,p] + ro.Asset_Buy[a,p]

ro.PortfolioValueConstraint = Constraint(rule=PortfolioValueConstraint)
ro.CashAccounting = Constraint(periods[1:], rule=CashAccounting)
ro.AssetBalance = Constraint(assets, periods[1:], rule=AssetBalance)

#Objective
def PortfolioValue(ro): 
	return ro.T
	
ro.PortfolioValue = Objective(rule=PortfolioValue, sense=maximize)

opt = SolverFactory('baron')
results = opt.solve(ro, tee=False, keepfiles=False)

# ro.Holdings.pprint()
# ro.Cash.pprint()
ro.Asset_Buy.pprint()
ro.Asset_Sell.pprint()
print(results)

## Use Robust Solution to solve several instances of Deterministic PO
## for various parameters and determine average reward 

asset1mean = np.array([1.25, 2, 4, 5, 3, 2, 3, 6, 9, 7]).reshape(1, -1) #up and down all the way 
asset2mean = np.array([5, 3, 2, 2, 1.25, 4, 5, 6, 7, 8]).reshape(1, -1) #down intially then up 
asset3mean = np.array([3, 5, 6, 9, 10, 8, 4, 2, 1.25, 4]).reshape(1, -1) #up initially then down 
asset_prices_means = np.vstack([asset1mean, asset2mean, asset3mean])
asset_prices_variance = np.ones(asset_prices_means.shape) * 0.45 #could use covariance matrix here instead 

Portfolio_value_list = []
opt = SolverFactory('gurobi')

for instance in range(1000): 

	asset_prices = np.array([np.random.normal(mu, sig) for mu, sig in 
            zip(asset_prices_means.flatten(), asset_prices_variance.flatten())]
            ).reshape(asset_prices_means.shape)

	do = ConcreteModel()

	#Variables 
	do.Cash = Var(periods, domain=NonNegativeReals)
	do.Cash[0].fix(initial_cash)
	do.Holdings = Var(assets, periods, domain=NonNegativeReals)
	for a in assets: 
		do.Holdings[a,0].fix(initial_assets[0])

	do.Asset_Sell = Var(assets, periods[1:], domain=NonNegativeReals)
	do.Asset_Buy = Var(assets, periods[1:], domain=NonNegativeReals)

	for a in assets: 
		for p in periods[1:]: 
			do.Asset_Sell[a, p].fix(value(ro.Asset_Sell[a, p]))
			do.Asset_Buy[a, p].fix(value(ro.Asset_Buy[a, p]))
			# do.Holdings[a, p].fix(value(ro.Holdings[a, p]))
			# if a == 1: 
			# 	do.Cash[p].fix(value(ro.Cash[p]))

	#Constraints 
	def CashAccounting(do, p): 
		return do.Cash[p] == do.Cash[p-1] \
		+ sum((1-sell_cost[a-1]) * asset_prices[a-1][p-1] * do.Asset_Sell[a,p] - \
			(1+buy_cost[a-1]) * asset_prices[a-1][p-1]* do.Asset_Buy[a,p] for a in assets)

	def AssetBalance(do, a, p): 
		return do.Holdings[a,p] == do.Holdings[a,p-1] \
		- do.Asset_Sell[a,p] + do.Asset_Buy[a,p]

	do.CashAccounting = Constraint(periods[1:], rule=CashAccounting)
	do.AssetBalance = Constraint(assets, periods[1:], rule=AssetBalance)

	#Objective
	def PortfolioValue(do): 
		return do.Cash[investment_horizon] + \
		sum(asset_prices[a-1][-1] * do.Holdings[a, investment_horizon] for a in assets)

	do.PortfolioValue = Objective(rule=PortfolioValue, sense=maximize)

	results = opt.solve(do, tee=False, keepfiles=False)
	Portfolio_value_list.append(value(do.PortfolioValue))
	print(value(do.PortfolioValue))

print("Mean Portfolio Value: ", np.mean(Portfolio_value_list))
print("StDev: ", np.std(Portfolio_value_list))
print("Minimum Portfolio Value:", min(Portfolio_value_list))