/**
 * replayBuffer.js
 * ------------------------------------------------------------
 * Experience Replay Buffer - DQN algoritmining muhim qismi.
 * Agent har qadamda ko'rgan tajribasini (s, a, r, s', done) shu
 * bufferga saqlaydi, keyin training paytida tasodifiy mini-batch
 * tanlab oladi. Bu ketma-ket tajribalar orasidagi korrelyatsiyani
 * kamaytirib, o'qitishni barqarorlashtiradi.
 *
 * Doiraviy massiv (circular buffer) sifatida implementatsiya
 * qilingan - xotira sig'imi to'lganda eng eski tajriba almashtiriladi.
 * ------------------------------------------------------------
 */

class ReplayBuffer {
    /**
     * @param {number} capacity - buferning maksimal sig'imi
     */
    constructor(capacity = RL_CONSTANTS.REPLAY_BUFFER_CAPACITY) {
        this.capacity = capacity;
        this.buffer = new Array(capacity);
        this.size = 0;
        this.writeIndex = 0;
    }

    /**
     * Yangi tajribani buferga qo'shadi.
     * @param {number[]} state
     * @param {number} action
     * @param {number} reward
     * @param {number[]} nextState
     * @param {boolean} done
     */
    push(state, action, reward, nextState, done) {
        this.buffer[this.writeIndex] = { state, action, reward, nextState, done };
        this.writeIndex = (this.writeIndex + 1) % this.capacity;
        this.size = Math.min(this.size + 1, this.capacity);
    }

    /**
     * Tasodifiy mini-batch tanlab oladi.
     * @param {number} batchSize
     * @returns {Array} tajribalar massivi
     */
    sample(batchSize) {
        const indices = sampleIndices(this.size, batchSize);
        return indices.map((i) => this.buffer[i]);
    }

    /** Buferda yetarlicha tajriba to'planganini tekshiradi */
    hasEnough(minSize) {
        return this.size >= minSize;
    }

    /** Buferni tozalaydi (masalan, yangi eksperiment boshlash uchun) */
    clear() {
        this.buffer = new Array(this.capacity);
        this.size = 0;
        this.writeIndex = 0;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ReplayBuffer };
}
