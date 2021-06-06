#!/usr/bin/env python
# encoding: utf8
import rospy
import turtlesim
from turtlesim.msg import Pose
from turtlesim.srv import SetPenRequest
from TurtlesimSIU import TurtlesimSIU
from geometry_msgs.msg import Twist
from collections import defaultdict
import signal
import sys
import numpy as np
import csv
import random
import sys

GRID_RES = 5  # liczba komórek siatki
CAM_RES = 6  # dł. boku siatki [px] ?????
SEC_PER_STEP = 1.0  # okres dyskretyzacji sterowania - nie mniej niż 1 [s]
WAIT_AFTER_MOVE = 0.01  # oczekiwanie po setPose() [s]

class TurtleAgent:
    def __init__(self, route, sec_id, seq, tname):
        self.route = route
        self.sec_id=sec_id
        self.seq=seq
        self.pose=Pose()
        self.fd=0
        self.goal_loc=Pose()
        self.color_api = TurtlesimSIU.ColorSensor(tname)

class TurtlesimEnv:
    def __init__(self, cam_res, grid_res, sec_per_step):
        self.turtle_api = TurtlesimSIU.TurtlesimSIU()
        self.px_meter_ratio = self.turtle_api.pixelsToScale()  # skala w pikselach na metr
        self.cam_res = cam_res  # długość boku kadru kamery w px
        self.grid_res = grid_res  # liczba komórek siatki
        self.sec_per_step = sec_per_step  # okres dyskretyzacji sterowania
        # interakcja z symulatorem
        signal.signal(signal.SIGINT, self.signal_handler)
        rospy.init_node('siu_example', anonymous=False)

    def load_routes(self, fname):  # z pliku do słownika tras agentów {agent:[(cnt,xmin,xmax,ymin,ymax,xgoal,ygoal)]}
        self.routes = defaultdict(list)
        with open(fname) as f:
            csv_reader = csv.reader(f, delimiter=';')
            for line in csv_reader:
                self.routes[line[0]].append(line[1:])
        # print(self.routes)
        self.agents={}
        #tname = list(self.routes.keys())[0]  # utworzenie żółwia dla pierwszej trasy
        # print(tname)
        #self.tname = tname
        for route, sections in self.routes.items():
            for sec_id, sec in enumerate(sections):
                for seq in range(int(sec[0])):
                    tname = route+'_'+str(sec_id)+'_'+str(seq)
                    if self.turtle_api.hasTurtle(tname):
                        self.turtle_api.killTurtle(tname)
                    self.turtle_api.spawnTurtle(tname, Pose())
                    self.turtle_api.setPen(tname, turtlesim.srv.SetPenRequest(r=255, g=255, b=255, width=5, off=1))
                    ta=TurtleAgent(route, sec_id, seq, tname)
                    self.agents[tname]=ta
	#for tname in self.agents:
	    #print(tname)

    def reset_turtle(self, tname):
        self.agents[tname].step_cnt = 0
        route_id=self.agents[tname].sec_id
        route =self.routes[self.agents[tname].route][route_id]
        self.agents[tname].goal_loc = Pose(x=float(route[5]), y=float(route[6]))
        # próba ulokowania startu we wskazanym obszarze i jednocześnie na drodze (niezerowy wektor zalecanej prędkości)
        while True:
            route_id=self.agents[tname].sec_id
            route =self.routes[self.agents[tname].route][route_id]
            x = np.random.uniform(float(route[1]), float(route[2]))
            y = np.random.uniform(float(route[3]), float(route[4]))
            # azymut początkowy w kierunku celu
            theta = np.arctan2(self.agents[tname].goal_loc.y - y, self.agents[tname].goal_loc.x - x)
            # przestawienie żółwia w losowe miejsce obszaru narodzin
            self.turtle_api.setPose(tname, Pose(x=x, y=y, theta=theta), mode='absolute')
            rospy.sleep(WAIT_AFTER_MOVE)  # odczekać UWAGA inaczej symulator nie zdąży przestawić żółwia
            # ????                            # przestawiać do skutku, aż znajdzie się na drodze
            color = self.agents[tname].color_api.check()
            if color.g != 255 or color.r != 201 or color.b != 199:
                return self.get_map(tname)
        return self.get_map(tname)

    def reset(self, max_steps,  section=None ):  # przygotowanie do nowego przejazdu, zwraca sytuację początkową
        self.max_steps = max_steps
        for tname in self.agents:
            self.agents[tname].step_cnt = 0
        # uczymy na trasie aktualnego agenta
        ret={}
        for tname in self.agents:
            route_id=self.agents[tname].sec_id
            route =self.routes[self.agents[tname].route][route_id]
            #if section is None:
                #section = random.choice(self.routes[tname])  # wybór losowego segmentu trasy
            self.agents[tname].goal_loc = Pose(x=float(route[5]), y=float(route[6]))
            # próba ulokowania startu we wskazanym obszarze i jednocześnie na drodze (niezerowy wektor zalecanej prędkości)
            while True:
                route_id=self.agents[tname].sec_id
                route =self.routes[self.agents[tname].route][route_id]
                x = np.random.uniform(float(route[1]), float(route[2]))
                y = np.random.uniform(float(route[3]), float(route[4]))
                # azymut początkowy w kierunku celu
                theta = np.arctan2(self.agents[tname].goal_loc.y - y, self.agents[tname].goal_loc.x - x)
                # przestawienie żółwia w losowe miejsce obszaru narodzin
                self.turtle_api.setPose(tname, Pose(x=x, y=y, theta=theta), mode='absolute')
                rospy.sleep(WAIT_AFTER_MOVE)  # odczekać UWAGA inaczej symulator nie zdąży przestawić żółwia
                # ????                            # przestawiać do skutku, aż znajdzie się na drodze
                color = self.agents[tname].color_api.check()
                if color.g != 255 or color.r != 201 or color.b != 199:
                    ret[tname] = self.get_map(tname)
                    break
	#print(ret)
        return ret

    def get_road(self, name):
        color = self.agents[name].color_api.check()
        fx = .02 * (color.r - 200)  # składowa x zalecanej prędkości <-1;1>
        fy = .02 * (color.b - 200)  # składowa y zalecanej prędkości <-1;1>
        if color.r == 201 and color.b == 199:  # poprawka na wypadniecie z trasy (trawa)
            fx, fy = 0.0, 0.0
        fa = color.g / 255.0  # mnożnik kary za naruszenie ograniczeń prędkości
        pose = self.turtle_api.getPose(name)  # aktualna pozycja żółwia
        fd = np.sqrt((self.agents[name].goal_loc.x - pose.x) ** 2 + (self.agents[name].goal_loc.y - pose.y) ** 2)  # odl. do celu
        fc = fx * np.cos(pose.theta) + fy * np.sin(pose.theta)  # rzut zalecanej prędkości na azymut
        fp = fy * np.cos(pose.theta) - fx * np.sin(pose.theta)  # rzut zalecanej prędkości na _|_ azymut
        return (fx, fy, fa, fd, fc + 1, fp + 1)

    def get_map(self, name):
        pose = self.turtle_api.getPose(name)
        img = self.turtle_api.readCamera(name,
                                         frame_pixel_size=self.cam_res,
					 #cell_count=16,
                                         cell_count=self.grid_res ** 2,
                                         x_offset=0,
                                         goal=self.agents[name].goal_loc,
                                         show_matrix_cells_and_goal=False)
        fx = np.eye(self.grid_res)
        fy = fx.copy()
        fa = fx.copy()
        fd = fx.copy()
	collision = False
        for i, row in enumerate(img.m_rows):
            for j, cell in enumerate(row.cells):
                fx[i, j] = cell.red
                fy[i, j] = cell.blue
                fa[i, j] = cell.green
                fd[i, j] = cell.distance
		#sys.stdout.write(str(cell.occupy)+" ")
		if cell.occupy ==0:
		    collision=True
		    
	    #print("\n")
	#print("\n")
        fc = fx * np.cos(pose.theta) + fy * np.sin(pose.theta)  # rzut zalecanej prędkości na azymut
        fp = fy * np.cos(pose.theta) - fx * np.sin(pose.theta)  # rzut zalecanej prędkości na _|_ azymut
        return (fx, fy, fa, fd, fc + 1, fp + 1, collision)

    # wykonuje zlecone działanie, zwraca sytuację, nagrodę, flagę końca przejazdu

    def step(self, actions, realtime=False):
        # pozycja PRZED krokiem sterowani
        for tname in actions:
            self.agents[tname].pose= self.turtle_api.getPose(tname)
            _, _, _,self.agents[tname].fd, _, _ = self.get_road(tname)  # odl. do celu
            self.agents[tname].step_cnt += 1

        #pose = self.turtle_api.getPose(self.tname)
        #_, _, _, fd, _, _ = self.get_road(self.tname)  # odl. do celu
        # action: [prędkość,skręt]
        if realtime:
            # ????                                         # symulacja płynna, nie skokowa
            # obliczenie i wykonanie przesunięcia
            for tname, action in actions.items():
                pose = self.agents[tname].pose
                vx = np.cos(pose.theta + action[1]) * action[0] * self.sec_per_step
                vy = np.sin(pose.theta + action[1]) * action[0] * self.sec_per_step
                x_fin = pose.x + vx
                y_fin = pose.y + vy
                theta_fin = pose.theta + action[1]
                smallest_possible_move = 0.05
                sign = lambda x: x and (1, -1)[x < 0]
                a = (pose.y - y_fin) / (pose.x - x_fin)
                b = pose.y - (a * pose.x)

                number_of_steps_x = abs(int(vx / smallest_possible_move))
                number_of_steps_theta = abs(int(action[1] / smallest_possible_move))

                for theta_step in range(number_of_steps_theta):
                    p = Pose(x=pose.x, y=pose.y, theta=pose.theta + smallest_possible_move * sign(action[1]))
                    self.turtle_api.setPose(tname, p, mode='absolute')
                    pose = p
                    rospy.sleep(WAIT_AFTER_MOVE)

                for x_step in range(number_of_steps_x):
                    curr_x = pose.x + smallest_possible_move * sign(vx)
                    p = Pose(x=curr_x, y=(a * curr_x + b), theta=theta_fin)
                    self.turtle_api.setPose(tname, p, mode='absolute')
                    pose = p
                    rospy.sleep(WAIT_AFTER_MOVE)

                p = Pose(x=x_fin, y=y_fin, theta=theta_fin)
                self.turtle_api.setPose(tname, p, mode='absolute')
                self.agents[tname].pose= p
                rospy.sleep(WAIT_AFTER_MOVE)

        else:
            for tname, action in actions.items():
                #pose = self.turtle_api.getPose(self.tname)
                pose = self.agents[tname].pose
                # obliczenie i wykonanie przesunięcia
                vx = np.cos(pose.theta + action[1]) * action[0] * self.sec_per_step
                vy = np.sin(pose.theta + action[1]) * action[0] * self.sec_per_step
                p = Pose(x=pose.x + vx, y=pose.y + vy, theta=pose.theta + action[1])
                self.turtle_api.setPose(tname, p, mode='absolute')
                self.agents[tname].pose= p
                rospy.sleep(WAIT_AFTER_MOVE)
        #print("Agent.keys: ", self.agents.keys())

        # pozycja PO kroku sterowania
        ret={}
        for tname in actions:
            ret[tname] = []
            pose1 = self.turtle_api.getPose(tname)
            fx1, fy1, fa1, fd1, _, _ = self.get_road(tname)  # warunki drogowe po przemieszczeniu
            vx1 = (pose1.x - self.agents[tname].pose.x) / self.sec_per_step  # prędkość w aktualnym kierunku
            vy1 = (pose1.y - self.agents[tname].pose.y) / self.sec_per_step
            v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
            fv1 = np.sqrt(fx1 ** 2 + fy1 ** 2)  # zalecany moduł prędkości
            SPEED_RWRD_RATE = 2.0  # wzmocnienie nagrody za jazdę w kierunku
            SPEED_RVRS_RATE = -10.0  # wzmocnienie kary za jazdę pod prąd
            SPEED_FINE_RATE = -10.0  # wzmocnienie kary za przekroczenie prędkości
            DIST_RWRD_RATE = 2.0  # wzmocnienie nagrody za zbliżanie się do celu
            OUT_OF_TRACK_FINE = -10  # ryczałtowa kara za wypadnięcie z trasy
            COLLISION_FINE = -10  # ryczałtowa kara za wypadnięcie z trasy
            reward = min(0, SPEED_FINE_RATE * (v1 - fv1))  # kara za przekroczenie prędkości
            
            if fv1 > .001:
                vf1 = (vx1 * fx1 + vy1 * fy1) / fv1  # rzut prędkości faktycznej na zalecaną
                if vf1 > 0:
                    reward += SPEED_RWRD_RATE * vf1  # nagroda za jazdę z prądem
                else:
                    reward -= SPEED_RVRS_RATE * vf1  # kara za jazdę pod prąd
            reward *= fa1  # relaksacja kar
            reward += DIST_RWRD_RATE * (self.agents[tname].fd - fd1)  # nagroda za zbliżenie się do celu
            done = False  # flaga zakończenia sesji
            collision = False
            ##KARA ZA KOLIZJE
	    map=self.get_map(tname)
	    ret[tname].append(map)
	    if map[6] == True:
		reward +=COLLISION_FINE	
                collision=True
            #for another_tname in actions:
             #   if another_tname == tname:
               #     continue;
               # else:
                #    if abs(self.agents[tname].pose.x - self.agents[another_tname].pose.x)<1 and abs(self.agents[tname].pose.y - self.agents[another_tname].pose.y)<1:
                 #       reward +=COLLISION_FINE	
                 #       collision=True
                 #       print("Turtle {} collided with {}".format(tname, another_tname)) 

            if abs(fx1) + abs(fy1) < .01 and fa1 == 1:  # wylądowaliśmy w rowie
                print("Turtle {} is in a ditch".format(tname))
                reward += OUT_OF_TRACK_FINE
                done = True
            if self.agents[tname].step_cnt > self.max_steps:
                print("Turtle {} executed max steps".format(tname))
                done = True
            ret[tname].append(reward)
            ret[tname].append(done)
            ret[tname].append(collision)
        return ret

    def signal_handler(self, sig, frame):
        print("Terminating")
        sys.exit(0)

    def is_near_goal(self, name):
        pose = self.turtle_api.getPose(name)
        # print(self.goal_loc.y, " ", self.goal_loc.x)
        # print( pose)
        if (abs(pose.x - self.agents[name].goal_loc.x) < 4 and abs(pose.y - self.agents[name].goal_loc.y) < 4):
            return True
        else:
            return False

    def set_goal(self, x, y, goal_x, goal_y, new_location, name):
        self.max_steps = 40
        self.agents[name].step_cnt = 0

        if not new_location:
            pose = self.turtle_api.getPose(name)
            x = pose.x
            y = pose.y
        self.agents[name].goal_loc = Pose(x=float(goal_x), y=float(goal_y))
        theta = np.arctan2(self.agents[name].goal_loc.y - y, self.agents[name].goal_loc.x - x)
        self.turtle_api.setPose(name, Pose(x=x, y=y, theta=theta), mode='absolute')
        return self.get_map(name)

    def is_outside_path(self, name):
        if self.agents[name].color_api.check().g != 255 or self.agents[name].color_api.check().r != 201 or self.agents[name].color_api.check().b != 199:
            return False
        else:
            return True


if __name__ == "__main__":
    env = TurtlesimEnv(CAM_RES, GRID_RES, SEC_PER_STEP)
    env.load_routes('roads.csv')
    env.reset(10)
    #rospy.sleep(1)
    #env.step((.5, -.2), False)
    #rospy.sleep(1)
