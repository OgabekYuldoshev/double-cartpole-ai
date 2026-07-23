/**
 * agent.js
 * ------------------------------------------------------------
 * DQNAgent - Deep Q-Learning agenti.
 * Tarkibiy qismlari:
 *   - Q-Network        (joriy baholovchi tarmoq)
 *   - Target Network    (barqaror maqsad qiymatlarini beruvchi tarmoq)
 *   - Epsilon-Greedy     (tadqiqot / o'zlashtirish muvozanati)
 *   - Replay Buffer      (tajribalarni saqlash)
 *
 * DIQQAT (kengaytirilishi uchun arxitektura):
 * Hozircha diskret action space (0,1,2) ishlatiladi. Kelajakda
 * continuous action (masalan DDPG/PPO orqali) qo'shish uchun
 * faqat `act()` va tarmoq chiqish qatlami almashtirilishi kifoya -
 * qolgan barcha qism (replay buffer, environment interfeysi)
 * o'zgarishsiz qoladi.
 * ------------------------------------------------------------
 */

class DQNAgent {
    /**
     * @param {number} stateSize
     * @param {number} actionSize
     */
    constructor(stateSize = RL_CONSTANTS.STATE_SIZE, actionSize = RL_CONSTANTS.ACTION_SIZE) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.layerSizes = [stateSize, ...RL_CONSTANTS.HIDDEN_LAYER_SIZES, actionSize];

        this.replayBuffer = new ReplayBuffer(RL_CONSTANTS.REPLAY_BUFFER_CAPACITY);
        this.gamma = RL_CONSTANTS.GAMMA;

        this.reset();
    }

    /** Agentni butunlay boshlang'ich holatga qaytaradi (yangi tarmoqlar, epsilon, hisoblagichlar) */
    reset() {
        this.qNetwork = new NeuralNetwork(this.layerSizes, RL_CONSTANTS.LEARNING_RATE);
        this.targetNetwork = new NeuralNetwork(this.layerSizes, RL_CONSTANTS.LEARNING_RATE);
        this.targetNetwork.copyWeightsFrom(this.qNetwork);

        this.epsilon = RL_CONSTANTS.EPSILON_START;

        this.trainingStepCount = 0;   // nechta marta trainOnBatch chaqirilgani
        this.targetUpdateCount = 0;   // nechta marta target tarmoq yangilangani ("generation")
        this.isTraining = true;       // UI orqali ON/OFF qilinadi
        this.lastLoss = 0;
    }

    /** Agentning to'liq holatini (tarmoqlar + hisoblagichlar) plain objectga chiqaradi */
    getState() {
        return {
            epsilon: this.epsilon,
            trainingStepCount: this.trainingStepCount,
            targetUpdateCount: this.targetUpdateCount,
            isTraining: this.isTraining,
            qNetwork: this.qNetwork.getState(),
            targetNetwork: this.targetNetwork.getState()
        };
    }

    /** getState() natijasidan agent holatini tiklaydi (davom ettirish uchun) */
    setState(state) {
        this.epsilon = state.epsilon;
        this.trainingStepCount = state.trainingStepCount;
        this.targetUpdateCount = state.targetUpdateCount;
        this.isTraining = state.isTraining;
        this.qNetwork.setState(state.qNetwork);
        this.targetNetwork.setState(state.targetNetwork);
    }

    /**
     * Epsilon-greedy siyosat asosida action tanlaydi.
     * @param {number[]} state
     * @returns {number} action indeksi (0, 1, yoki 2)
     */
    act(state) {
        if (this.isTraining && Math.random() < this.epsilon) {
            return randomInt(this.actionSize); // tasodifiy tadqiqot (exploration)
        }
        return this.getBestAction(state); // o'zlashtirish (exploitation)
    }

    /** Joriy Q-Network bo'yicha eng yaxshi (max Q qiymatli) actionni qaytaradi */
    getBestAction(state) {
        const qValues = this.qNetwork.predict(state);
        let bestAction = 0;
        let bestValue = qValues[0];
        for (let i = 1; i < qValues.length; i++) {
            if (qValues[i] > bestValue) {
                bestValue = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    /** Tajribani replay bufferga saqlaydi */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push(state, action, reward, nextState, done);
    }

    /**
     * Replay bufferdan mini-batch olib, Q-Networkni bitta qadam o'qitadi
     * (Bellman tenglamasi asosida): Q(s,a) <- r + gamma * max_a' Q_target(s', a')
     */
    trainStep() {
        if (!this.isTraining) return;
        if (!this.replayBuffer.hasEnough(RL_CONSTANTS.MIN_REPLAY_SIZE)) return;

        const batch = this.replayBuffer.sample(RL_CONSTANTS.BATCH_SIZE);

        const inputs = [];
        const targets = [];
        const masks = [];

        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            // Joriy Q-Network bashorati (faqat mask orqali tanlangan action yangilanadi)
            const currentQ = this.qNetwork.predict(state);

            let targetValue = reward;
            if (!done) {
                const nextQValues = this.targetNetwork.predict(nextState);
                const maxNextQ = Math.max(...nextQValues);
                targetValue = reward + this.gamma * maxNextQ;
            }

            const targetVector = currentQ.slice();
            targetVector[action] = targetValue;

            const mask = new Array(this.actionSize).fill(0);
            mask[action] = 1;

            inputs.push(state);
            targets.push(targetVector);
            masks.push(mask);
        }

        this.lastLoss = this.qNetwork.trainOnBatch(inputs, targets, masks);
        this.trainingStepCount += 1;

        if (this.trainingStepCount % RL_CONSTANTS.TARGET_UPDATE_INTERVAL === 0) {
            this.updateTargetNetwork();
        }

        this.updateEpsilon();
    }

    /** Target Networkni Q-Network og'irliklari bilan yangilaydi */
    updateTargetNetwork() {
        this.targetNetwork.copyWeightsFrom(this.qNetwork);
        this.targetUpdateCount += 1;
    }

    /** Epsilon qiymatini chiziqli ravishda minimal darajagacha kamaytiradi */
    updateEpsilon() {
        this.epsilon = Math.max(
            RL_CONSTANTS.EPSILON_MIN,
            this.epsilon - RL_CONSTANTS.EPSILON_DECAY
        );
    }

    /** Training rejimini yoqadi/o'chiradi (UI tugmasi uchun) */
    setTraining(enabled) {
        this.isTraining = enabled;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DQNAgent };
}
