#!/usr/bin/env python3
import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import GlobalAveragePooling2D, Conv3D, Permute, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
import env_3 as tse
import csv

REPLAY_MEMORY_SIZE = 3000 #???? (tysiace)
MIN_REPLAY_MEMORY_SIZE = 800 #????
MINIBATCH_SIZE = 32 #???? (16)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
EPISODES = 5000 #???? (tysiace)
DISCOUNT = 0.99
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
LEARNING_RATE = 0.1
CTL_DIM = 6
MAX_STEPS = 40 #???? (20?)          # liczba krokow w sekwencji uczenia

random.seed(1)
np.random.seed(1)

env=tse.TurtlesimEnv(tse.CAM_RES,tse.GRID_RES,tse.SEC_PER_STEP)
env.load_routes('test-roads-8ag.csv')

def ctl2act(decision):          #  predkosc\skret    -.1rad 0 .1rad
    v = .2                      #      0.2              0   1   2
    if decision>=3:             #      0.4              3   4   5
        v = .4
    w = .25*(decision%3-1)
    return [v,w]


def ctl2act_multiagent(decisions):
    actions = {}
    for tname, decision in decisions.items():  #  predkosc\skret    -.1rad 0 .1rad
        v = .2                                 #      0.2              0   1   2
        if decision>=3:                        #      0.4              3   4   5
            v = .4
        w = .25*(decision%3-1)
        actions[tname] = [v, w]
    return actions
    

def inp_stack(last, cur):       # przygotowanie wejscia dla sieci
    inp=np.stack([cur[2],cur[3],cur[4],cur[5],cur[6], last[2],last[3],last[4],last[5], last[6]],axis=-1)
    return inp

def decision(model,last,cur):   # predykcja sterowania na podst. biezacej i ostatniej sytuacji
    inp=np.expand_dims(inp_stack(last, cur), axis=-1)
    return model.predict(np.expand_dims(inp, axis=0)).flatten()

def create_model():             # wy: pozytek z CTL_DIM mozliwych decyzji
    input_shape = (tse.GRID_RES, tse.GRID_RES, 10, 1)  # warstwy (x,y,a,d,collisions) w chwili n oraz n-1
    model = Sequential()
    # ???? struktura
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 8), activation='relu', input_shape=input_shape))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Flatten())
    # model.add(Dropout(.1))
    model.add(Dense(CTL_DIM, activation="linear"))  # pozytek kazdej z CTL_DIM decyzji
    model.compile(loss="mse", optimizer=Adam(lr=0.002), metrics=["accuracy"])
    return model

# Start learning from the beginning
model=create_model()
target_model=create_model()
target_model.set_weights(model.get_weights())
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
target_update_counter = [0]


# Continue learning the model
'''model_json = ''
with open("modele/model1_375episodes.json",'r') as f:
    model_json=f.read()
    print(model_json)
model = model_from_json(model_json)
target_model = model_from_json(model_json)
model.load_weights("modele/model1_375episodes.h5")
model.compile(loss="mse", optimizer=Adam(lr=0.002), metrics=["accuracy"])
target_model.compile(loss="mse", optimizer=Adam(lr=0.002), metrics=["accuracy"])
target_model.set_weights(model.get_weights())
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
target_update_counter = [0]'''


# Q-learning na podst.
# https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
# oraz https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/?completed=/reinforcement-learning-agent-self-driving-autonomous-cars-carla-python/

def train(target_update_counter,terminal_state,learn_step):
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    Q0=np.zeros(((MINIBATCH_SIZE,CTL_DIM))) # nagrody krok n wg modelu doraznego
    Q1target = Q0.copy()                    # nagrody krok n+1 wg modelu docelowego
    for idx,(last_state, current_state, _, _, new_state, _) in enumerate(minibatch):
        Q0[idx]=decision(model,last_state,current_state)                # krok n   / model dorazny
        Q1target[idx]=decision(target_model,current_state,new_state)    # krok n+1 / model docelowy
    X = []                                  # sytuacje treningowe
    y = []                                  # decyzje treningowe
    for idx,(last_state, current_state, control, reward, new_state, done) in enumerate(minibatch):
        if done:
            new_q = reward      # nie ma juz stanu nastepnego, ucz sie otrzymywac faktyczna nagrode
        else:
            # nagroda uwzglednia nagrode za kolejny etap sterowania
            new_q = (1-LEARNING_RATE)*reward + LEARNING_RATE*(reward+DISCOUNT*np.max(Q1target[idx]))
        q0=Q0[idx].copy()
        q0[control]=new_q
        inp=inp_stack(last_state,current_state)
        X.append(np.expand_dims(inp, axis=-1))
        y.append(q0)
    model.fit(np.stack(X),np.stack(y),batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False)
    if terminal_state:                      # licznik rosnie tylko dla kroku koncowego
        target_update_counter[0]+=1
    if target_update_counter[0] > UPDATE_TARGET_EVERY:
        target_model.set_weights(model.get_weights())
        target_update_counter[0] = 0

def train_main():
    epsilon=1
    episode_cnt = 0
    agents = env.reset(MAX_STEPS)
    current_states = {tname:agent for tname, agent in agents.items()}
    tnames = set(current_states.keys())  # zbior agentow
    last_states = {tname:agent for tname, agent in agents.items()}   # zaczyna od postoju, poprz. stan taki jak obecny
    #print("Current states:\n", current_states)
    rewards = []
    while episode_cnt < EPISODES:
        episode_rwrd=0
        learn_step=1
        controls = {}
        for tname in tnames:
            if np.random.random() > epsilon:          # sterowanie wg reguly albo losowe
                controls[tname] = np.argmax(decision(model, last_states[tname], current_states[tname]))
            else:
                controls[tname] = np.random.randint(0, CTL_DIM)  # losowa predkosc pocz. i skret
        scene = env.step(ctl2act_multiagent(controls))  # wykonanie kroku symulacji
        terminal_state = {}
        for tname in tnames:
            terminal_state[tname] = False
            if scene[tname][2] or scene[tname][3]: # done or collision
                terminal_state[tname] = True
            episode_rwrd+=scene[tname][1]
            replay_memory.append((last_states[tname], current_states[tname], controls[tname], scene[tname][1], scene[tname][0], terminal_state[tname]))  # last 3 params: reward, new_state, done
        for tname in tnames:
            if len(replay_memory) >= MIN_REPLAY_MEMORY_SIZE:    # zgromadzono odpowiednio duzo probek
                #print("Collected enough samples to start learning")
                train(target_update_counter,terminal_state[tname],learn_step)
            learn_step+=1 
        for tname in tnames:
            if not terminal_state[tname]:
                last_states[tname]=current_states[tname]  # przejscie do nowego stanu z zapamietaniem poprzedniego
                current_states[tname]=scene[tname][0]  # new_state
            else:
                episode_cnt += 1
                print("Episode no. {}, total reward: {}".format(episode_cnt, episode_rwrd))
                current_states[tname] = env.reset_turtle(tname)
                last_states[tname] = [i.copy() for i in current_states[tname]] 
                rewards.append([episode_rwrd])

                # okresowy zapis modelu do plikow
                if episode_cnt>0 and episode_cnt % 50 == 0:
                    plik_json = "modele_wieloag/model1_" + str(episode_cnt) + "episodes.json"
                    plik_h5 = "modele_wieloag/model1_" + str(episode_cnt) + "episodes.h5"
                    model_json=model.to_json()
                    with open(plik_json,'w') as f:
                        f.write(model_json)
                    model.save_weights(plik_h5)
                    with open("nagrody.csv",'a',newline='') as ff:
                        writer = csv.writer(ff)
                        writer.writerows(rewards)
                    rewards.clear()

        if epsilon > MIN_EPSILON:       # rosnace p-stwo uczenia na podst. historii
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    # zapis modelu do plikow
    model_json=model.to_json()
    with open('model3.json','w') as f:
        f.write(model_json)
    model.save_weights('model3.h5')


# testowanie modelu
def test():
    # wczytaj model z plikow
    model_json = ''
    with open("modele_wieloag/model1_600episodes.json",'r') as f:
        model_json=f.read()
        #print(model_json)
    model = model_from_json(model_json)
    model.load_weights("modele_wieloag/model1_600episodes.h5")

    # Uwagi do skrytpu testowego:
    # tests = [[tname,cnt,0,x,0,y,goal_x,goal_y]], 
    # 'cnt' powinno byc zawsze 1, bo kazdy zolw ma inny punkt startowy
    # dwa zera sa tylko po to, by plik byl kompatybilny z funkcja reset, ktora dzialala dla scen. uczacego

    # wczytaj skrypt testowy
    fname = "test-roads-8ag.csv"
    tests = []
    with open(fname) as f:
        csv_reader=csv.reader(f, delimiter=';')
        for line in csv_reader:
            tests.append(list(line))

    accomplished_goals=0
    finished_turtles = 0

    # tylko dla celow inicjalizacji struktur danych
    agents = env.reset(80)
    current_states = {tname:agent for tname, agent in agents.items()}
    tnames = set(current_states.keys())  # zbior agentow
    last_states = {tname:agent for tname, agent in agents.items()} 

    id = 0
    for tname in tnames:
        current_states[tname] = env.set_goal(float(tests[id][3]),float(tests[id][5]),float(tests[id][6]),float(tests[id][7]),True, tname)
        last_states[tname] = env.set_goal(float(tests[id][3]),float(tests[id][5]),float(tests[id][6]),float(tests[id][7]),True, tname)
        id += 1

    while finished_turtles < len(tests):
        turtles_to_remove = []
        controls = {}
        for tname in tnames:
            controls[tname] = np.argmax(decision(model, last_states[tname], current_states[tname]))  # sterowanie modelem
        scene = env.step(ctl2act_multiagent(controls))  # wykonanie kroku symulacji
        terminal_state = {}
        for tname in tnames:
            terminal_state[tname] = False
            if scene[tname][2] or scene[tname][3]: # max_steps, ditch or collision
                terminal_state[tname] = True
        for tname in tnames:
            if not terminal_state[tname]:
                if(env.is_near_goal(tname)):
                    print("{} accomplished goal".format(tname))
                    accomplished_goals+=1
                    finished_turtles += 1
                    turtles_to_remove.append(tname)
                    break
                last_states[tname]=current_states[tname]  # przejscie do nowego stanu z zapamietaniem poprzedniego
                current_states[tname]=scene[tname][0]  # new_state
            else:
                print("{} failed to accomplish goal".format(tname))  
                turtles_to_remove.append(tname)
                finished_turtles += 1 
        for t in turtles_to_remove:
            tnames.remove(t)
         
    print('Accuracy:', accomplished_goals/len(tests) * 100, '%') 



if __name__ == "__main__":
    #train_main()
    test()

