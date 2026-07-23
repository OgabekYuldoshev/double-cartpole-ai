/**
 * network.js
 * ------------------------------------------------------------
 * To'liq noldan yozilgan Feedforward Neural Network (MLP).
 * Hech qanday tashqi ML kutubxona (TensorFlow.js va h.k.) ishlatilmagan.
 *
 * Arxitektura: kiruvchi qatlam -> yashirin qatlam(lar) [ReLU] -> chiquvchi qatlam [Linear]
 * DQN uchun chiquvchi qatlam har bir action uchun Q-qiymatini bildiradi.
 *
 * O'qitish uchun Adam optimizatori qo'llaniladi (SGD ga nisbatan
 * tezroq va barqarorroq yaqinlashish beradi).
 * ------------------------------------------------------------
 */

// Adam optimizer uchun standart giperparametrlar (umumiy qabul qilingan qiymatlar)
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;

class NeuralNetwork {
    /**
     * @param {number[]} layerSizes - masalan [6, 64, 64, 3]
     * @param {number} learningRate
     */
    constructor(layerSizes, learningRate = RL_CONSTANTS.LEARNING_RATE) {
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.length - 1; // og'irlik qatlamlari soni
        this.learningRate = learningRate;

        this.weights = [];  // weights[l] : [outSize][inSize]
        this.biases = [];   // biases[l]  : [outSize]

        // Adam optimizer holati (momentlar)
        this.mWeights = [];
        this.vWeights = [];
        this.mBiases = [];
        this.vBiases = [];
        this.adamTimeStep = 0;

        this._initializeParameters();
    }

    /** He initsializatsiyasi - ReLU tarmoqlar uchun mos keladi */
    _initializeParameters() {
        for (let l = 0; l < this.numLayers; l++) {
            const inSize = this.layerSizes[l];
            const outSize = this.layerSizes[l + 1];
            const stddev = Math.sqrt(2 / inSize);

            const W = [];
            const b = [];
            const mW = [];
            const vW = [];
            for (let i = 0; i < outSize; i++) {
                const row = [];
                const mRow = [];
                const vRow = [];
                for (let j = 0; j < inSize; j++) {
                    row.push(this._gaussianRandom() * stddev);
                    mRow.push(0);
                    vRow.push(0);
                }
                W.push(row);
                mW.push(mRow);
                vW.push(vRow);
                b.push(0);
            }
            this.weights.push(W);
            this.biases.push(b);
            this.mWeights.push(mW);
            this.vWeights.push(vW);
            this.mBiases.push(new Array(outSize).fill(0));
            this.vBiases.push(new Array(outSize).fill(0));
        }
    }

    /** Box-Muller transformatsiyasi orqali standart normal taqsimotdan tasodifiy son */
    _gaussianRandom() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    /** ReLU faollashtirish funksiyasi */
    _relu(z) {
        return z.map((value) => Math.max(0, value));
    }

    /** ReLU hosilasi (0 yoki 1) */
    _reluDerivative(z) {
        return z.map((value) => (value > 0 ? 1 : 0));
    }

    /**
     * Bitta namuna uchun to'g'ridan-to'g'ri (forward) hisoblash.
     * Backprop uchun barcha oraliq qiymatlarni (activations, zs) saqlaydi.
     *
     * @param {number[]} input - kirish vektori
     * @returns {{activations: number[][], zs: number[][]}}
     */
    forward(input) {
        let a = input;
        const activations = [input];
        const zs = [];

        for (let l = 0; l < this.numLayers; l++) {
            const W = this.weights[l];
            const b = this.biases[l];
            const z = new Array(W.length).fill(0);

            for (let i = 0; i < W.length; i++) {
                let sum = b[i];
                const row = W[i];
                for (let j = 0; j < row.length; j++) {
                    sum += row[j] * a[j];
                }
                z[i] = sum;
            }

            // Oxirgi qatlam chiziqli (linear) - Q-qiymatlar cheksiz oralig'da bo'lishi mumkin
            const isOutputLayer = l === this.numLayers - 1;
            a = isOutputLayer ? z.slice() : this._relu(z);

            zs.push(z);
            activations.push(a);
        }

        return { activations, zs };
    }

    /** Faqat chiquvchi qatlam natijasini qaytaradi (Q-qiymatlar massivi) */
    predict(input) {
        return this.forward(input).activations[this.activations_lastIndex()];
    }

    /** Yordamchi: oxirgi activation indeksini beradi */
    activations_lastIndex() {
        return this.numLayers;
    }

    /**
     * Mini-batch bo'yicha bitta o'qitish qadamini bajaradi.
     *
     * @param {number[][]} inputs - batch kirishlari
     * @param {number[][]} targets - har namuna uchun to'liq maqsad vektori (faqat mask=1 bo'lgan joy muhim)
     * @param {number[][]} masks - har namuna uchun qaysi chiqish neyroni yangilanishini bildiruvchi 0/1 vektor
     * @returns {number} o'rtacha MSE loss (monitoring uchun)
     */
    trainOnBatch(inputs, targets, masks) {
        const batchSize = inputs.length;

        // Gradientlarni to'plash uchun akkumulyatorlar (0 bilan boshlanadi)
        const gradW = this.weights.map((W) => W.map((row) => row.map(() => 0)));
        const gradB = this.biases.map((b) => b.map(() => 0));

        let totalLoss = 0;

        for (let n = 0; n < batchSize; n++) {
            const { activations, zs } = this.forward(inputs[n]);
            const output = activations[this.numLayers];

            // Chiquvchi qatlam xatoligi: faqat mask=1 bo'lgan neyronlar uchun (boshqalari 0)
            let delta = output.map((value, i) => {
                const diff = masks[n][i] * (value - targets[n][i]);
                totalLoss += masks[n][i] * diff * diff;
                return diff; // linear chiqish uchun dLoss/dz = dLoss/da
            });

            // Backpropagation: chiquvchi qatlamdan kiruvchi qatlamga tomon
            for (let l = this.numLayers - 1; l >= 0; l--) {
                const prevActivation = activations[l];
                const W = this.weights[l];

                // Og'irlik va bias gradientlarini to'plash
                for (let i = 0; i < W.length; i++) {
                    gradB[l][i] += delta[i];
                    const row = W[i];
                    for (let j = 0; j < row.length; j++) {
                        gradW[l][i][j] += delta[i] * prevActivation[j];
                    }
                }

                // Oldingi qatlam uchun delta hisoblash (agar kiruvchi qatlam bo'lmasa)
                if (l > 0) {
                    const newDelta = new Array(prevActivation.length).fill(0);
                    for (let j = 0; j < prevActivation.length; j++) {
                        let sum = 0;
                        for (let i = 0; i < W.length; i++) {
                            sum += W[i][j] * delta[i];
                        }
                        newDelta[j] = sum;
                    }
                    const reluDeriv = this._reluDerivative(zs[l - 1]);
                    delta = newDelta.map((value, j) => value * reluDeriv[j]);
                }
            }
        }

        // Gradientlarni batch hajmiga bo'lib o'rtachalashtiramiz
        for (let l = 0; l < this.numLayers; l++) {
            for (let i = 0; i < gradW[l].length; i++) {
                gradB[l][i] /= batchSize;
                for (let j = 0; j < gradW[l][i].length; j++) {
                    gradW[l][i][j] /= batchSize;
                }
            }
        }

        this._applyAdamUpdate(gradW, gradB);

        return totalLoss / batchSize;
    }

    /** Adam optimizatori orqali og'irlik va bias qiymatlarini yangilaydi */
    _applyAdamUpdate(gradW, gradB) {
        this.adamTimeStep += 1;
        const t = this.adamTimeStep;
        const biasCorrection1 = 1 - Math.pow(ADAM_BETA1, t);
        const biasCorrection2 = 1 - Math.pow(ADAM_BETA2, t);

        for (let l = 0; l < this.numLayers; l++) {
            for (let i = 0; i < this.weights[l].length; i++) {
                // Bias yangilash
                this.mBiases[l][i] = ADAM_BETA1 * this.mBiases[l][i] + (1 - ADAM_BETA1) * gradB[l][i];
                this.vBiases[l][i] = ADAM_BETA2 * this.vBiases[l][i] + (1 - ADAM_BETA2) * gradB[l][i] * gradB[l][i];
                const mHatB = this.mBiases[l][i] / biasCorrection1;
                const vHatB = this.vBiases[l][i] / biasCorrection2;
                this.biases[l][i] -= this.learningRate * mHatB / (Math.sqrt(vHatB) + ADAM_EPSILON);

                for (let j = 0; j < this.weights[l][i].length; j++) {
                    const g = gradW[l][i][j];
                    this.mWeights[l][i][j] = ADAM_BETA1 * this.mWeights[l][i][j] + (1 - ADAM_BETA1) * g;
                    this.vWeights[l][i][j] = ADAM_BETA2 * this.vWeights[l][i][j] + (1 - ADAM_BETA2) * g * g;
                    const mHat = this.mWeights[l][i][j] / biasCorrection1;
                    const vHat = this.vWeights[l][i][j] / biasCorrection2;
                    this.weights[l][i][j] -= this.learningRate * mHat / (Math.sqrt(vHat) + ADAM_EPSILON);
                }
            }
        }
    }

    /** Boshqa tarmoqning barcha og'irlik/bias qiymatlarini nusxalab oladi (Target Network uchun) */
    copyWeightsFrom(otherNetwork) {
        this.weights = otherNetwork.weights.map((W) => W.map((row) => row.slice()));
        this.biases = otherNetwork.biases.map((b) => b.slice());
    }

    /** Tarmoqning to'liq mustaqil nusxasini yaratadi */
    clone() {
        const copy = new NeuralNetwork(this.layerSizes.slice(), this.learningRate);
        copy.copyWeightsFrom(this);
        return copy;
    }

    /** Tarmoqning to'liq holatini (og'irliklar + Adam momentlari) plain objectga chiqaradi */
    getState() {
        return {
            weights: this.weights.map((W) => W.map((row) => row.slice())),
            biases: this.biases.map((b) => b.slice()),
            mWeights: this.mWeights.map((W) => W.map((row) => row.slice())),
            vWeights: this.vWeights.map((W) => W.map((row) => row.slice())),
            mBiases: this.mBiases.map((b) => b.slice()),
            vBiases: this.vBiases.map((b) => b.slice()),
            adamTimeStep: this.adamTimeStep
        };
    }

    /** getState() natijasidan tarmoq holatini tiklaydi */
    setState(state) {
        this.weights = state.weights.map((W) => W.map((row) => row.slice()));
        this.biases = state.biases.map((b) => b.slice());
        this.mWeights = state.mWeights.map((W) => W.map((row) => row.slice()));
        this.vWeights = state.vWeights.map((W) => W.map((row) => row.slice()));
        this.mBiases = state.mBiases.map((b) => b.slice());
        this.vBiases = state.vBiases.map((b) => b.slice());
        this.adamTimeStep = state.adamTimeStep;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralNetwork };
}
