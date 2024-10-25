##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### @Altered by João Lucas Marques Camilo
### Demo of use of VictimSim



import os
import random
import numpy as np
import heapq
import math
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
##Classe no para criação de A*
class No:
    def __init__(self):        
        self.parent_x = 0       #coordenada pai x
        self.parent_y = 0
        self.fn = float('inf')  #custo total g+h
        self.gn = float('inf')  #Custo apartir do nó inicial
        self.hn = 0              #Custo do nó até a vitima
        
## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        

    # Check if a no is valid (within the grid)

    # Check if a no is unblocked
    def is_unblocked(grid, row, col):
        return grid[row][col] == 1
    
    # Check if a no is the destination
    def is_destination(self,row, col, dest):
        return row == dest[0] and col == dest[1]
    
    #Calcula o valor de h
    def calculate_h_value(self,row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

    def track(self,no_details, dest):
        print("The Path is ")
        row = dest[0]
        col = dest[1]
        self.plan_x = row
        self.plan_y = col
        
        dx = 0
        dy = 0

        
        plano = []
        plano.append((dx, dy, True))
        # Trace the path from destination to source using parent cells
        while not (no_details[row][col].parent_i == row and no_details[row][col].parent_j == col):
            #self.plan.append((row, col,False))
            temp_row = no_details[row][col].parent_i
            temp_col = no_details[row][col].parent_j

            dx = row - temp_row
            dy = col - temp_col

            row = temp_row
            col = temp_col
            # Add the source cell to the path
            plano.append((dx, dy,False))
        plano.reverse()
        self.plan.extend(plano)
        

        
        
        
    
    # Implement the A* search algorithm
    #TODO: Colocar tempo restante do resgate
    #TODO: Ir até vitima e voltar para a base?
    def a_star_search(self, src, dest):
        # Check if the source and destination are valid

        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return

        #TODO: Não tenho tamannho do mapa para inicializar as listas
        min_x,max_x,min_y,max_y = self.map.get_min_max_map()
        max_y = max_y - min_y + 1
        max_x = max_x - min_x + 1
        
        # Inicializa lista fechada de nós
        closed_list = [[False for _ in range(max_y)] for _ in range(max_x)]
        #Inicializa os nós do mapa inteiro
        no_details = [[No() for _ in range(max_y)] for _ in range(max_x)]

        # Initialize the start no details
        i = src[0]
        j = src[1]
        no_details[i][j].fn = 0
        no_details[i][j].gn = 0
        no_details[i][j].hn = 0
        no_details[i][j].parent_i = i
        no_details[i][j].parent_j = j

        #Inicializa lista aberta (Nos para ser visitado) com o começo em src
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Remove o elemento com o menor f
            p = heapq.heappop(open_list)

            # Marca na lista fechada que a coordenada (i,j) foi visitada
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            
            # Pega direção disponivel
            actions_res = self.map.get((self.plan_x, self.plan_y))[2]

            for k, ar in enumerate(actions_res):
                #Se caminho não for livre, apenas pulamos. 
                #No mapa não foi implementado VS.UNKS
                if ar != VS.CLEAR:
                    print(f"{self.NAME} {k} not clear")
                    continue
                
                dir = Rescuer.AC_INCR[k]
               
               
                new_i = i + dir[0]
                new_j = j + dir[1]
                

                #Verifica se o sucessor é valido. 
                difficulty = self.map.get((new_i,new_j))
                if difficulty == None:
                    #print("posição mapa NUla")
                    continue

        
                if self.is_destination(new_i, new_j, dest):
                    # Set the parent of the destination no
                    no_details[new_i][new_j].parent_i = i
                    no_details[new_i][new_j].parent_j = j
                    print("The destination no is found")
                    self.track(no_details,(new_i,new_j))
                    found_dest = True
                    return
                else:
                 
                    
                    # Calculate the new f, g, and h values
                    if dir[0] == 0 or dir[1] == 0:
                        g_new = no_details[i][j].gn + self.COST_LINE * difficulty[0]
                    else:
                        g_new = no_details[i][j].gn +self.COST_DIAG * difficulty[0]
                   
                    h_new = self.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the no is not in the open list or the new f value is smaller
                    if no_details[new_i][new_j].fn == float('inf') or no_details[new_i][new_j].fn > f_new:
                        # Add the no to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the no details
                        no_details[new_i][new_j].fn = f_new
                        no_details[new_i][new_j].gn = g_new
                        no_details[new_i][new_j].hn = h_new
                        no_details[new_i][new_j].parent_i = i
                        no_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all nos
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    

    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print(f"\n\n*** R E S C U E R ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        self.map.draw()

        print()
        #print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims

        # print the found victims - you may comment out
        #for seq, data in self.victims.items():
        #    coord, vital_signals = data
        #    x, y = coord
        #    print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()
        print(f"{self.NAME} PLAN")
        i = 1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
                  
    
        self.set_state(VS.ACTIVE)

            
   
                       

    def __depth_search(self, actions_res):
        enough_time = True
        ##print(f"\n{self.NAME} actions results: {actions_res}")
        for i, ar in enumerate(actions_res):

            if ar != VS.CLEAR:
                ##print(f"{self.NAME} {i} not clear")
                continue
            
            # planning the walk
            dx, dy = Rescuer.AC_INCR[i]  # get the increments for the possible action
            target_xy = (self.plan_x + dx, self.plan_y + dy)

            # checks if the explorer has not visited the target position
            if not self.map.in_map(target_xy):
                ##print(f"{self.NAME} target position not explored: {target_xy}")
                continue

            # checks if the target position is already planned to be visited 
            if (target_xy in self.plan_visited):
                ##print(f"{self.NAME} target position already visited: {target_xy}")
                continue

            # Now, the rescuer can plan to walk to the target position
            self.plan_x += dx
            self.plan_y += dy
            difficulty, vic_seq, next_actions_res = self.map.get((self.plan_x, self.plan_y))
            #print(f"{self.NAME}: planning to go to ({self.plan_x}, {self.plan_y})")

            if dx == 0 or dy == 0:
                step_cost = self.COST_LINE * difficulty
            else:
                step_cost = self.COST_DIAG * difficulty

            #print(f"{self.NAME}: difficulty {difficulty}, step cost {step_cost}")
            #print(f"{self.NAME}: accumulated walk time {self.plan_walk_time}, rtime {self.plan_rtime}")

            # check if there is enough remaining time to walk back to the base
            if self.plan_walk_time + step_cost > self.plan_rtime:
                enough_time = False
                #print(f"{self.NAME}: no enough time to go to ({self.plan_x}, {self.plan_y})")
            
            if enough_time:
                # the rescuer has time to go to the next position: update walk time and remaining time
                self.plan_walk_time += step_cost
                self.plan_rtime -= step_cost
                self.plan_visited.add((self.plan_x, self.plan_y))

                if vic_seq == VS.NO_VICTIM:
                    self.plan.append((dx, dy, False)) # walk only
                    #print(f"{self.NAME}: added to the plan, walk to ({self.plan_x}, {self.plan_y}, False)")

                if vic_seq != VS.NO_VICTIM:
                    # checks if there is enough remaining time to rescue the victim and come back to the base
                    if self.plan_rtime - self.COST_FIRST_AID < self.plan_walk_time:
                        print(f"{self.NAME}: no enough time to rescue the victim")
                        enough_time = False
                    else:
                        self.plan.append((dx, dy, True))
                        #print(f"{self.NAME}:added to the plan, walk to and rescue victim({self.plan_x}, {self.plan_y}, True)")
                        self.plan_rtime -= self.COST_FIRST_AID

            # let's see what the agent can do in the next position
            if enough_time:
                self.__depth_search(self.map.get((self.plan_x, self.plan_y))[2]) # actions results
            else:
                return

        return
    
    def __planner(self):
        """ A private method that calculates the walk actions in a OFF-LINE MANNER to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        """ This plan starts at origin (0,0) and chooses the first of the possible actions in a clockwise manner starting at 12h.
        Then, if the next position was visited by the explorer, the rescuer goes to there. Otherwise, it picks the following possible action.
        For each planned action, the agent calculates the time will be consumed. When time to come back to the base arrives,
        it reverses the plan."""

        # This is a off-line trajectory plan, each element of the list is a pair dx, dy that do the agent walk in the x-axis and/or y-axis.
        # Besides, it has a flag indicating that a first-aid kit must be delivered when the move is completed.
        # For instance (0,1,True) means the agent walk to (x+0,y+1) and after walking, it leaves the kit.

        self.plan_visited.add((0,0)) # always start from the base, so it is already visited
        difficulty, vic_seq, actions_res = self.map.get((0,0))
        #self.__depth_search(actions_res)
        vic = []
        for i in self.victims:
            vic.append(self.victims[i])

        while (vic):
            coord,vs=vic.pop()
            self.a_star_search((self.plan_x,self.plan_y),coord)

        # Reverse the path to get the path from source to destination
        #self.plan.reverse()
        # push actions into the plan to come back to the base
        if self.plan == []:
            return

        come_back_plan = []

        for a in reversed(self.plan):
            # triple: dx, dy, no victim - when coming back do not rescue any victim
            come_back_plan.append((a[0]*-1, a[1]*-1, False))

        self.plan = self.plan + come_back_plan
        
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           #input(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

