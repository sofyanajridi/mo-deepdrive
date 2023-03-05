1. Multi-Agent Single Objective
	We only have 1 multi agent environment: Deepdrive intersection
	- IPPO (Independant PPO)
		Two independant  PPO learners
		
		Following paper: https://arxiv.org/pdf/2103.01955.pdf looks at the performance of decentralized IPPO in a MA setting compared to PPO with centralized value function inputs (MAPPO)
		
		They focus on environments with a cooperative setting (same reward function) with a small amount of agents.
		
		Their paper concludes that IPPO with some consideration about some hyperparameters offers very similar performance as MAPPO in the disccused setting.
		
		-> This is exactly what we have with the deepdrive intersection environment (2 agents, cooperative setting)
		-> Maybe nice to see how they would compare here? IPPO vs MAPPO
		
	- MAPPO
		- Centralized value function inputs
		- Centralized training and decentralized execution
		  
	- MADDPG
	  Paper: https://arxiv.org/pdf/1706.02275.pdf
	  with code implementation: https://github.com/openai/maddpg
	  
	  - Centralized training and decentralized execution
	  - a simple extension of actor-critic policy gradient methods where the critic is augmented with extra information about the policies of other agents vs actor only has access to local information
	  - centralized critic for each agent compared to other solutions that most of the time use centralized critic for all agents (above MAPPO example). (we are in a cooperative setting with same reward function, so this should not gives us much advantage)
	    
	-> I propose IPPO, MAPPO and MADDPG as such we have comparison between decentralized and centralized and between on-policy (PPO) and off-policy (DDPG)
	  
2. Single-Agent Multi Objective
   Here we can use deepdrive and also powergym
   Deepdrive:
	   - Waypoint environment
	   - Static obstactle environment
	Powergym:
	- 13bus
	- ...
	
	Following the utility based approach:
	I think because of the scope of this thesis ,we made the assumption that we are in a known utility function scenario and under SER
	We assume that the utility function is completly known -> Single policy algorithm

	Different types of utility functions we can test on:
	1. Linear
	2. Exponentional
	3. Concave
	
	Algorithms:
	- Linear utility
	  Any single objective RL algorithm
	- Non-linear
	  
	  Q-actor critic or even Advantage Actor critic using accrued rewards and state conditioning
	  http://roijers.info/pub/reymond2021modem.pdf
		http://proceedings.mlr.press/v119/siddique20a/siddique20a.pdf 
		
		
		
3. Multi-Agent Multi Objective
   Again here we only have 1 environment:
   Deepdrive Intersection evironment
   
   Here we can maybe have two different settings:
   - Team reward team utility
     Both vehicles have the same utility function
   - Team reward individual utility
	e.g: one vehicle uses a utility function that focuses on more aggressive driving behaviour vs other vehicle with a more relaxed driving behaviour
	
	Algorithms:
	- MO-Wolf-PHC
	
4. Alternative
   (If their is some time left) We can also use the following environment/benchmark proposed by Willem:
   https://github.com/LucasAlegre/sumo-rl
   Multi agent environment that by looking at the code can be easily extendend to support multi objective. In that way we have an extra enviroment to test on for category 1 and 3.
   






