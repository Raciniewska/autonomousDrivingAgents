#!/usr/bin/env python
# encoding: utf8
import rospy
import turtlesim
from turtlesim.msg import Pose
from turtlesim.srv  import SetPenRequest
from TurtlesimSIU import TurtlesimSIU
from geometry_msgs.msg import Twist
from collections import defaultdict
import signal
import sys
import numpy as np
import csv
import random

GRID_RES = 5            # liczba komórek siatki
CAM_RES = 6         # dł. boku siatki [px] ?????
SEC_PER_STEP = 1.0      # okres dyskretyzacji sterowania - nie mniej niż 1 [s]
WAIT_AFTER_MOVE = .10   # oczekiwanie po setPose() [s]

class TurtlesimEnv:
    def __init__(self, cam_res, grid_res, sec_per_step):
        self.turtle_api = TurtlesimSIU.TurtlesimSIU()
        self.px_meter_ratio = self.turtle_api.pixelsToScale()   # skala w pikselach na metr
        self.cam_res = cam_res              # długość boku kadru kamery w px
        self.grid_res = grid_res            # liczba komórek siatki
        self.sec_per_step = sec_per_step    # okres dyskretyzacji sterowania
        # interakcja z symulatorem
        signal.signal(signal.SIGINT, self.signal_handler)
        rospy.init_node('siu_example', anonymous=False)
    def load_routes(self,fname):            # z pliku do słownika tras agentów {agent:[(cnt,xmin,xmax,ymin,ymax,xgoal,ygoal)]}
        self.routes=defaultdict(list)
	with open(fname) as f:
		csv_reader=csv.reader(f, delimiter=';')
		for line in csv_reader:
			self.routes[line[0]].append(line[1:])
	print(self.routes)
        tname=list(self.routes.keys())[0]   # utworzenie żółwia dla pierwszej trasy
	print(tname)
        self.tname=tname
        if self.turtle_api.hasTurtle(tname):
            self.turtle_api.killTurtle(tname)
        self.turtle_api.spawnTurtle(tname,Pose())
        self.turtle_api.setPen(tname,turtlesim.srv.SetPenRequest(r=255,g=255,b=255,width=5,off=1))
        self.color_api = TurtlesimSIU.ColorSensor(tname)

    def reset(self,max_steps,section=None): # przygotowanie do nowego przejazdu, zwraca sytuację początkową
        self.max_steps = max_steps
        self.step_cnt = 0
        # uczymy na trasie aktualnego agenta
        if section is None:
                section=random.choice(self.routes[self.tname])  # wybór losowego segmentu trasy
        self.goal_loc = Pose(x=float(section[5]), y=float(section[6]))
        # próba ulokowania startu we wskazanym obszarze i jednocześnie na drodze (niezerowy wektor zalecanej prędkości)
        while True:
            x=np.random.uniform(float(section[1]),float(section[2]))
            y=np.random.uniform(float(section[3]),float(section[4]))
            # azymut początkowy w kierunku celu
            theta=np.arctan2(self.goal_loc.y-y,self.goal_loc.x-x)
            # przestawienie żółwia w losowe miejsce obszaru narodzin
            self.turtle_api.setPose(self.tname,Pose(x=x,y=y,theta=theta),mode='absolute')
            rospy.sleep(WAIT_AFTER_MOVE)    # odczekać UWAGA inaczej symulator nie zdąży przestawić żółwia
            #????                            # przestawiać do skutku, aż znajdzie się na drodze
	    if self.color_api.check().g !=255 or self.color_api.check().r!=201 or self.color_api.check().b!=199:
	    	return self.get_map(self.tname)
	    
        return self.get_map(self.tname)

    def get_road(self,name):
        color = self.color_api.check()
        fx=.02*(color.r-200)                # składowa x zalecanej prędkości <-1;1>
        fy=.02*(color.b-200)                # składowa y zalecanej prędkości <-1;1>
        fa=color.g/255.0                    # mnożnik kary za naruszenie ograniczeń prędkości
        pose=self.turtle_api.getPose(name)  # aktualna pozycja żółwia
        fd=np.sqrt((self.goal_loc.x-pose.x)**2+(self.goal_loc.y-pose.y)**2)     # odl. do celu
        fc=fx*np.cos(pose.theta)+fy*np.sin(pose.theta)  # rzut zalecanej prędkości na azymut
        fp=fy*np.cos(pose.theta)-fx*np.sin(pose.theta)  # rzut zalecanej prędkości na _|_ azymut
        return (fx,fy,fa,fd,fc+1,fp+1)

    def get_map(self,name):
        pose = self.turtle_api.getPose(name)
        img = self.turtle_api.readCamera(name,
                                         frame_pixel_size=self.cam_res,
                                         cell_count=self.grid_res**2,
                                         x_offset=0,
                                         goal=self.goal_loc,
                                         show_matrix_cells_and_goal=False)
        fx=np.eye(self.grid_res)
        fy=fx.copy()
        fa=fx.copy()
        fd=fx.copy()
        for i,row in enumerate(img.m_rows):
            for j,cell in enumerate(row.cells):
                fx[i,j] = cell.red
                fy[i,j] = cell.blue
                fa[i,j] = cell.green
                fd[i,j] = cell.distance
        fc=fx*np.cos(pose.theta)+fy*np.sin(pose.theta)  # rzut zalecanej prędkości na azymut
        fp=fy*np.cos(pose.theta)-fx*np.sin(pose.theta)  # rzut zalecanej prędkości na _|_ azymut
        return (fx,fy,fa,fd,fc+1,fp+1)
    # wykonuje zlecone działanie, zwraca sytuację, nagrodę, flagę końca przejazdu

    def step(self,action,realtime=False):
        self.step_cnt += 1
        # pozycja PRZED krokiem sterowania
        pose = self.turtle_api.getPose(self.tname)
        _,_,_,fd,_,_ = self.get_road(self.tname)         # odl. do celu
        # action: [prędkość,skręt]
        if realtime:
            #????                                         # symulacja płynna, nie skokowa
	    # obliczenie i wykonanie przesunięcia
            vx = np.cos(pose.theta+action[1])*action[0]*self.sec_per_step
            vy = np.sin(pose.theta+action[1])*action[0]*self.sec_per_step
	    x_fin=pose.x+vx
	    y_fin=pose.y+vy
	    theta_fin=pose.theta+action[1]
	    smallest_possible_move=0.05
	    sign=lambda x: x and (1,-1)[x<0]
	    a= (pose.y-y_fin)/(pose.x-x_fin)
	    b=pose.y-(a*pose.x)

	    number_of_steps_x=abs(int(vx/smallest_possible_move))
	    number_of_steps_theta=abs(int(action[1]/smallest_possible_move))

	    for theta_step in range(number_of_steps_theta):
		p=Pose(x=self.pose.x,y=self.pose.y,theta=self.pose.theta+smallest_possible_move*sign(action[1]))
		self.turtle_api.setPose(self.tname,p,mode='absolute')
            	self.pose = p
            	rospy.sleep(WAIT_AFTER_MOVE)

	    for x_step in range(number_of_steps_x):
		curr_x=self.pose.x+smallest_possible_move*sign(vx)
		p=Pose(x=curr_x,y=(a*curr_x+b),theta=theta_fin)
		self.turtle_api.setPose(self.tname,p,mode='absolute')
            	self.pose = p
            	rospy.sleep(WAIT_AFTER_MOVE)

	    p=Pose(x=x_fin,y=y_fin,theta=theta_fin)
	    self.turtle_api.setPose(self.tname,p,mode='absolute')
            self.pose = p
            rospy.sleep(WAIT_AFTER_MOVE)

        else:
            # obliczenie i wykonanie przesunięcia
            vx = np.cos(pose.theta+action[1])*action[0]*self.sec_per_step
            vy = np.sin(pose.theta+action[1])*action[0]*self.sec_per_step
            p=Pose(x=pose.x+vx,y=pose.y+vy,theta=pose.theta+action[1])
            self.turtle_api.setPose(self.tname,p,mode='absolute')
            self.pose = p
            rospy.sleep(WAIT_AFTER_MOVE)
        # pozycja PO kroku sterowania
        pose1 = self.turtle_api.getPose(self.tname)
        fx1,fy1,fa1,fd1,_,_ = self.get_road(self.tname) # warunki drogowe po przemieszczeniu
        vx1 = (pose1.x-pose.x)/self.sec_per_step        # prędkość w aktualnym kierunku
        vy1 = (pose1.y-pose.y)/self.sec_per_step
        v1  = np.sqrt(vx1**2+vy1**2)
        fv1 = np.sqrt(fx1**2+fy1**2)                    # zalecany moduł prędkości
        SPEED_RWRD_RATE = 2.0                           # wzmocnienie nagrody za jazdę w kierunku
        SPEED_RVRS_RATE = -10.0                         # wzmocnienie kary za jazdę pod prąd
        SPEED_FINE_RATE = -10.0                         # wzmocnienie kary za przekroczenie prędkości
        DIST_RWRD_RATE  = 2.0                           # wzmocnienie nagrody za zbliżanie się do celu
        OUT_OF_TRACK_FINE = -10                         # ryczałtowa kara za wypadnięcie z trasy
        reward = min(0,SPEED_FINE_RATE*(v1-fv1))        # kara za przekroczenie prędkości
        if fv1>.001:
            vf1 = (vx1*fx1+vy1*fy1)/fv1                 # rzut prędkości faktycznej na zalecaną
            if vf1>0:
                reward += SPEED_RWRD_RATE*vf1           # nagroda za jazdę z prądem
            else:
                reward -= SPEED_RVRS_RATE*vf1           # kara za jazdę pod prąd
        reward *= fa1                                   # relaksacja kar
        reward += DIST_RWRD_RATE*(fd-fd1)               # nagroda za zbliżenie się do celu
        done = False                                    # flaga zakończenia sesji
        if abs(fx1)+abs(fy1)<.01 and fa1==1:            # wylądowaliśmy w rowie
            reward += OUT_OF_TRACK_FINE
            done = True
        if self.step_cnt>self.max_steps:
            done = True
        return self.get_map(self.tname), reward, done, None
    def signal_handler(self,sig, frame):
        print ("Terminating")
        sys.exit(0)

if __name__ == "__main__":
    env=TurtlesimEnv(CAM_RES,GRID_RES,SEC_PER_STEP)
    env.load_routes('roads.csv')
    env.reset(10)
    rospy.sleep(1)
    env.step((.5,-.2),False)
    rospy.sleep(1)
    env.step((2.5,-4.2),True)
