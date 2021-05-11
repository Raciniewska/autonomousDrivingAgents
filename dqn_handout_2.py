import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import GlobalAveragePooling2D, Conv3D, Permute, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
import turtlesim_env as tse

REPLAY_MEMORY_SIZE = 2000 #???? (tysiace)
MIN_REPLAY_MEMORY_SIZE = 2000 #????
MINIBATCH_SIZE = 16 #???? (16)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
EPISODES = 200 #???? (tysiace)
DISCOUNT = 0.99
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
LEARNING_RATE = 0.1
CTL_DIM = 6
MAX_STEPS = 20 #???? (20?)          # liczba krokow w sekwencji uczenia

random.seed(1)
np.random.seed(1)

env=tse.TurtlesimEnv(tse.CAM_RES,tse.GRID_RES,tse.SEC_PER_STEP)
env.load_routes('roads.csv')

def ctl2act(decision):          #  predkosc\skret    -.1rad 0 .1rad
    v = .2                      #      0.2              0   1   2
    if decision>=3:             #      0.4              3   4   5
        v = .4
    w = .25*(decision%3-1)
    return [v,w]

def inp_stack(last, cur):       # przygotowanie wejscia dla sieci
    inp=np.stack([cur[2],cur[3],cur[4],cur[5],last[2],last[3],last[4],last[5]],axis=-1)
    return inp

def decision(model,last,cur):   # predykcja sterowania na podst. biezacej i ostatniej sytuacji
    inp=np.expand_dims(inp_stack(last, cur), axis=-1)
    return model.predict(np.expand_dims(inp, axis=0)).flatten()

def create_model():             # wy: pozytek z CTL_DIM mozliwych decyzji
    input_shape = (tse.GRID_RES, tse.GRID_RES, 8, 1)  # warstwy (x,y,a,d) w chwili n oraz n-1
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

model=create_model()
target_model=create_model()
target_model.set_weights(model.get_weights())
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
target_update_counter = [0]

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
    for episode in range(EPISODES):
        current_state = env.reset(MAX_STEPS)
        last_state = [i.copy() for i in current_state]  # zaczyna od postoju, poprz. stan taki jak obecny
        episode_rwrd=0
        learn_step=1
        while True:
            if np.random.random() > epsilon:                    # sterowanie wg reguly albo losowe
                control = np.argmax(decision(model, last_state, current_state))
            else:
                control=np.random.randint(0, CTL_DIM)           # losowa predkosc pocz. i skret
            new_state, reward, done, _ = env.step(ctl2act(control))     # wykonanie kroku symulacji
            episode_rwrd+=reward
            replay_memory.append((last_state, current_state, control, reward, new_state, done))
            if len(replay_memory) >= MIN_REPLAY_MEMORY_SIZE:    # zgromadzono odpowiednio duzo probek
                train(target_update_counter,done,learn_step)
            learn_step+=1
            if done:
                break
            last_state=current_state        # przejscie do nowego stanu z zapamietaniem poprzedniego
            current_state=new_state
            if epsilon > MIN_EPSILON:       # rosnace p-stwo uczenia na podst. historii
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
    # zapis modelu do plikow
    model_json=model.to_json()
    with open('nazwa_modelu.json','w') as f:
        f.write(model_json)
    model.save_weights('nazwa_modelu.h5')

def test():

    #model_json = ''
    #with open('model.json','r') as f:
    #    model_json=f.read()
    #    print(model_json)
    #
    #model = model_from_json(model_json)
    #model.load_weights('model.h5')

    max_moves = 30
    # tests (x,y, goal_x, goal_y, new_location) x,y - wspolrzedne poczatkowe, goal_x, goal_y - docelowe, new_location - informacja o tym, czy zolw ma byc zaladowany w nowym miejscu
    tests = [['turtle1', 9.6, 21.62, 9.2, 15.6, True], ['turtle1', 9.2, 15.6, 10.2, 10.6, False]]
    accomplished_goals=0
    fail=False

    for x in tests:
        current_state = env.set_goal(x[1],x[2],x[3],x[4],x[5] or fail, x[0])
        for c in range(0, max_moves):
            last_state = [i.copy() for i in current_state]
            control = np.argmax(decision(model, last_state, current_state))
            new_state, reward, done, _ = env.step(ctl2act(control))     # wykonanie kroku symulacji

            if(env.is_near_goal(x[0])):
                print('Goal accomplished')
                accomplished_goals+=1
                fail=False
                break

            if(env.is_outside_path(x[0]) or x==max_moves-1):
                print('Failed to accomplish goal')  
                fail=True
                break     
         
    print('Accuracy:', accomplished_goals/len(tests) * 100, '%') 

if __name__ == "__main__":
    #train_main()
    test()

    #model_json=model.to_json()
    #with open('model.json','w') as f:
    #    f.write(model_json)
