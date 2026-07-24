/**
 * trainer.js
 * ------------------------------------------------------------
 * Bu fayl ikkita klassni o'z ichiga oladi:
 *
 * 1) CartPoleEnvironment - OpenAI Gym uslubidagi environment.
 *    Metodlari: reset(), step(force), getState(), reward(), isDone(), render()
 *    Bu klass physics.js (hisoblash), cart.js/pole.js (ma'lumot) va
 *    renderer.js (chizish) orasidagi "orkestrator" vazifasini bajaradi.
 *
 * 2) Trainer - Agent va Environment orasidagi RL sikli (training loop)ni
 *    boshqaradi: action tanlash -> environment qadam -> tajribani saqlash ->
 *    tarmoqni o'qitish -> statistikani yangilash.
 * ------------------------------------------------------------
 */

// ==================== ENVIRONMENT ====================

class CartPoleEnvironment {
    constructor() {
        this.cart = new Cart(PHYSICS_CONSTANTS.CART_MASS, PHYSICS_CONSTANTS.CART_POSITION_LIMIT);
        this.pole1 = new Pole(PHYSICS_CONSTANTS.POLE1_MASS, PHYSICS_CONSTANTS.POLE1_LENGTH);
        this.pole2 = new Pole(PHYSICS_CONSTANTS.POLE2_MASS, PHYSICS_CONSTANTS.POLE2_LENGTH);

        this.physics = new DoubleCartPolePhysics(this.cart, this.pole1, this.pole2);

        this.stepCount = 0;
        this.lastForce = 0;

        // "Markazga yaqin" bonusi uchun burchak chegarasi (radianda)
        this.centerZoneAngleRad = degToRad(PHYSICS_CONSTANTS.CENTER_ZONE_DEG);
    }

    /**
     * Environmentni boshlang'ich holatga qaytaradi.
     * @returns {number[]} boshlang'ich state vektori
     */
    reset() {
        this.cart.reset();
        this.pole1.reset();
        this.pole2.reset();
        this.stepCount = 0;
        this.lastForce = 0;
        return this.getState();
    }

    /**
     * Diskret action indeksini (0,1,2) haqiqiy kuchga (Newton) o'giradi.
     * Kelajakda continuous action qo'shish uchun shu metodni almashtirish kifoya:
     * masalan `actionToForce(action) { return action * MAX_FORCE; }`
     */
    actionToForce(action) {
        const F = PHYSICS_CONSTANTS.FORCE_MAGNITUDE;
        switch (action) {
            case 0: return -F;  // chap
            case 1: return 0;   // hech narsa
            case 2: return F;   // o'ng
            default: return 0;
        }
    }

    /**
     * Bitta simulyatsiya qadamini bajaradi.
     * @param {number} force - aravachaga qo'yiladigan kuch (Newton)
     * @returns {{state: number[], reward: number, done: boolean}}
     */
    step(force) {
        this.lastForce = force;
        this.physics.step(force, PHYSICS_CONSTANTS.DT);
        this.stepCount += 1;

        const done = this.isDone();
        const rewardValue = this.reward(done);

        return {
            state: this.getState(),
            reward: rewardValue,
            done
        };
    }

    /**
     * Joriy state vektorini qaytaradi:
     * [cartX, cartVelocity, pole1Angle, pole1AngularVelocity, pole2Angle, pole2AngularVelocity]
     */
    getState() {
        return [
            this.cart.position,
            this.cart.velocity,
            this.pole1.angle,
            this.pole1.angularVelocity,
            this.pole2.angle,
            this.pole2.angularVelocity
        ];
    }

    /**
     * Reward funksiyasi:
     *   - Agar tayoqlardan biri yiqilsa       -> -100
     *   - Agar aravacha chegaradan chiqsa      -> -1
     *   - Aks holda (ikkalasi tik)             -> +1, markazdan uzoqlikka qarab uzluksiz
     *     jarima ayiriladi, va markazga yaqin bo'lsa +0.1 qo'shiladi
     *
     * Uzluksiz pozitsiya jarimasi muhim: aks holda ±0.6m zonadan tashqarida
     * (lekin ±2.4m chegara ichida) agentga hech qanday signal berilmaydi va u
     * asta-sekin markazdan chetlashib, oxir-oqibat chegaradan chiqib ketadi.
     */
    reward(done) {
        const pole1Fallen = this.pole1.hasFallen();
        const pole2Fallen = this.pole2.hasFallen();
        const cartOutOfBounds = this.cart.isOutOfBounds();

        if (pole1Fallen || pole2Fallen) {
            return REWARD_CONSTANTS.FALL_PENALTY;
        }

        if (cartOutOfBounds) {
            return REWARD_CONSTANTS.OUT_OF_BOUNDS_PENALTY;
        }

        let totalReward = REWARD_CONSTANTS.ALIVE_REWARD;
        totalReward -= REWARD_CONSTANTS.POSITION_PENALTY * Math.abs(this.cart.position);

        const bothNearCenterPosition = Math.abs(this.cart.position) < PHYSICS_CONSTANTS.CENTER_ZONE_POSITION;
        const bothPolesNearVertical =
            this.pole1.isNearVertical(this.centerZoneAngleRad) &&
            this.pole2.isNearVertical(this.centerZoneAngleRad);

        if (bothNearCenterPosition && bothPolesNearVertical) {
            totalReward += REWARD_CONSTANTS.CENTER_BONUS;
        }

        return totalReward;
    }

    /**
     * Episode tugash shartlarini tekshiradi:
     *   - pole1 yoki pole2 burchagi ±36 gradusdan oshsa
     *   - aravacha chegaradan chiqsa
     *   - (xavfsizlik uchun) maksimal qadamlar soniga yetsa
     */
    isDone() {
        if (this.pole1.hasFallen() || this.pole2.hasFallen()) return true;
        if (this.cart.isOutOfBounds()) return true;
        if (this.stepCount >= RL_CONSTANTS.MAX_STEPS_PER_EPISODE) return true;
        return false;
    }

    /** Renderer orqali joriy holatni chizadi (fizikadan mustaqil chizish qatlami) */
    render(p, renderer, debugMode) {
        renderer.render(p, this, debugMode);
    }
}

// ==================== TRAINER ====================

const TRAINER_STORAGE_KEY = 'doubleCartPole_trainingState';

class Trainer {
    /**
     * @param {DQNAgent} agent
     * @param {CartPoleEnvironment} environment
     */
    constructor(agent, environment) {
        this.agent = agent;
        this.environment = environment;

        this.currentState = this.environment.reset();

        this.episodeCount = 0;
        this.episodeReward = 0;
        this.lastEpisodeReward = 0;
        this.totalReward = 0;
        this.bestEpisodeReward = -Infinity;

        this.loadState();
    }

    /**
     * localStorage'da saqlangan holat bo'lsa, agent (tarmoqlar, epsilon, h.k.) va
     * trainer statistikasini o'sha joydan tiklaydi. Reload qilinganda training
     * boshidan boshlanmasin uchun shu funksiya constructor'da chaqiriladi.
     * @returns {boolean} holat topilib tiklandimi
     */
    loadState() {
        try {
            const raw = localStorage.getItem(TRAINER_STORAGE_KEY);
            if (!raw) return false;

            const data = JSON.parse(raw);
            this.episodeCount = data.episodeCount ?? 0;
            this.totalReward = data.totalReward ?? 0;
            this.lastEpisodeReward = data.lastEpisodeReward ?? 0;
            this.bestEpisodeReward = data.bestEpisodeReward ?? -Infinity;
            if (data.agent) {
                this.agent.setState(data.agent);
            }
            return true;
        } catch (e) {
            return false;
        }
    }

    /** Joriy to'liq training holatini (agent + statistika) localStorage'ga yozadi */
    saveState() {
        try {
            const data = {
                episodeCount: this.episodeCount,
                totalReward: this.totalReward,
                lastEpisodeReward: this.lastEpisodeReward,
                bestEpisodeReward: this.bestEpisodeReward,
                agent: this.agent.getState()
            };
            localStorage.setItem(TRAINER_STORAGE_KEY, JSON.stringify(data));
        } catch (e) {
            // localStorage mavjud bo'lmasa yoki to'lib qolsa - jim o'tkazib yuboramiz
        }
    }

    /**
     * Bitta to'liq simulyatsiya + o'qitish qadamini bajaradi.
     * sketch.js har frame (yoki simulation speed ko'paytmasi bo'yicha bir necha marta) chaqiradi.
     */
    update() {
        const action = this.agent.act(this.currentState);
        const force = this.environment.actionToForce(action);

        const { state: nextState, reward, done } = this.environment.step(force);

        this.agent.remember(this.currentState, action, reward, nextState, done);
        this.agent.trainStep();

        this.episodeReward += reward;
        this.totalReward += reward;
        this.currentState = nextState;

        if (done) {
            this.lastEpisodeReward = this.episodeReward;
            this.bestEpisodeReward = Math.max(this.bestEpisodeReward, this.episodeReward);
            this.episodeCount += 1;
            this.episodeReward = 0;
            this.currentState = this.environment.reset();

            // Har episode tugaganda saqlaymiz - reload qilinsa shu joydan davom etadi
            this.saveState();
        }
    }

    /** Butun training jarayonini noldan boshlaydi (agent va environment birga) */
    fullReset() {
        this.currentState = this.environment.reset();
        this.agent.reset();
        this.episodeCount = 0;
        this.episodeReward = 0;
        this.lastEpisodeReward = 0;
        this.totalReward = 0;
        this.bestEpisodeReward = -Infinity;
        try {
            localStorage.removeItem(TRAINER_STORAGE_KEY);
        } catch (e) {
            // e'tiborsiz qoldiramiz
        }
    }

    /** UI panel uchun joriy statistikalarni qaytaradi */
    getStats() {
        return {
            episode: this.episodeCount,
            currentEpisodeReward: this.episodeReward,
            lastEpisodeReward: this.lastEpisodeReward,
            bestEpisodeReward: this.bestEpisodeReward === -Infinity ? 0 : this.bestEpisodeReward,
            totalReward: this.totalReward,
            epsilon: this.agent.epsilon,
            generation: this.agent.targetUpdateCount,
            stepCount: this.environment.stepCount,
            isTraining: this.agent.isTraining,
            loss: this.agent.lastLoss
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CartPoleEnvironment, Trainer };
}
