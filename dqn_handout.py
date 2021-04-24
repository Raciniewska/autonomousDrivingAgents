import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import GlobalAveragePooling2D, Conv3D, Permute, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
import turtlesim_env as tse

REPLAY_MEMORY_SIZE = ???? (tysiące)
MIN_REPLAY_MEMORY_SIZE = ????
MINIBATCH_SIZE = ???? (16)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
EPISODES = ???? (tysiące)
DISCOUNT = 0.99
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
LEARNING_RATE = 0.1
CTL_DIM = 6
MAX_STEPS = ???? (20?)          # liczba kroków w sekwencji uczenia

random.seed(1)
np.random.seed(1)

env=tse.TurtlesimEnv(tse.CAM_RES,tse.GRID_RES,tse.SEC_PER_STEP)
env.load_routes('routes.csv')

def ctl2act(decision):          #  prędkość\skręt    -.1rad 0 .1rad
    v = .2                      #      0.2              0   1   2
    if decision>=3:             #      0.4              3   4   5
        v = .4
    w = .25*(decision%3-1)
    return [v,w]

def inp_stack(last, cur):       # przygotowanie wejścia dla sieci
    inp=np.stack([cur[2],cur[3],cur[4],cur[5],last[2],last[3],last[4],last[5]],axis=-1)
    return inp

def decision(model,last,cur):   # predykcja sterowania na podst. bieżącej i ostatniej sytuacji
    inp=np.expand_dims(inp_stack(last, cur), axis=-1)
    return model.predict(np.expand_dims(inp, axis=0)).flatten()

def create_model():             # wy: pożytek z CTL_DIM możliwych decyzji
    input_shape = (tse.GRID_RES, tse.GRID_RES, 8, 1)  # warstwy (x,y,a,d) w chwili n oraz n-1
    model = Sequential()
    ???? struktura
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 8), activation='relu', input_shape=input_shape))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Permute((1, 2, 4, 3)))
    model.add(Conv3D(filters=16, kernel_size=(2, 2, 16), activation='relu'))
    model.add(Flatten())
    # model.add(Dropout(.1))
    model.add(Dense(CTL_DIM, activation="linear"))  # pożytek każdej z CTL_DIM decyzji
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
    Q0=np.zeros(((MINIBATCH_SIZE,CTL_DIM))) # nagrody krok n wg modelu doraźnego
    Q1target = Q0.copy()                    # nagrody krok n+1 wg modelu docelowego
    for idx,(last_state, current_state, _, _, new_state, _) in enumerate(minibatch):
        Q0[idx]=decision(model,last_state,current_state)                # krok n   / model doraźny
        Q1target[idx]=decision(target_model,current_state,new_state)    # krok n+1 / model docelowy
    X = []                                  # sytuacje treningowe
    y = []                                  # decyzje treningowe
    for idx,(last_state, current_state, control, reward, new_state, done) in enumerate(minibatch):
        if done:
            new_q = reward      # nie ma już stanu następnego, ucz się otrzymywać faktyczną nagrodę
        else:
            # nagroda uwzględnia nagrodę za kolejny etap sterowania
            new_q = (1-LEARNING_RATE)*reward + LEARNING_RATE*(reward+DISCOUNT*np.max(Q1target[idx]))
        q0=Q0[idx].copy()
        q0[control]=new_q
        inp=inp_stack(last_state,current_state)
        X.append(np.expand_dims(inp, axis=-1))
        y.append(q0)
    model.fit(np.stack(X),np.stack(y),batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False)
    if terminal_state:                      # licznik rośnie tylko dla kroku końcowego
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
            if np.random.random() > epsilon:                    # sterowanie wg reguły albo losowe
                control = np.argmax(decision(model, last_state, current_state))
            else:
                control=np.random.randint(0, CTL_DIM)           # losowa prędkość pocz. i skręt
            new_state, reward, done, _ = env.step(ctl2act(control))     # wykonanie kroku symulacji
            episode_rwrd+=reward
            replay_memory.append((last_state, current_state, control, reward, new_state, done))
            if len(replay_memory) >= MIN_REPLAY_MEMORY_SIZE:    # zgromadzono odpowiednio dużo próbek
                train(target_update_counter,done,learn_step)
            learn_step+=1
            if done:
                break
            last_state=current_state        # przejście do nowego stanu z zapamiętaniem poprzedniego
            current_state=new_state
            if epsilon > MIN_EPSILON:       # rosnące p-stwo uczenia na podst. historii
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
    # zapis modelu do plików
    model_json=model.to_json()
    with open(f'nazwa_modelu.json','w') as f:
        f.write(model_json)
    model.save_weights(f'nazwa_modelu.h5')