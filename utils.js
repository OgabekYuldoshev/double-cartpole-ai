/**
 * utils.js
 * ------------------------------------------------------------
 * Loyihaning barcha "magic number"lari shu yerda konstanta
 * sifatida to'planadi, shuningdek fizik va matematik hisob-kitoblar
 * uchun umumiy yordamchi funksiyalar joylashgan.
 * Bu fayl boshqa hech qanday modulga bog'liq emas.
 * ------------------------------------------------------------
 */

// ==================== FIZIK KONSTANTALAR ====================
const PHYSICS_CONSTANTS = Object.freeze({
    GRAVITY: 9.81,              // m/s^2

    CART_MASS: 1.0,             // kg
    POLE1_MASS: 0.1,            // kg
    POLE2_MASS: 0.1,            // kg

    POLE1_LENGTH: 0.6,          // metr (to'liq uzunlik)
    POLE2_LENGTH: 0.6,          // metr (to'liq uzunlik)

    CART_FRICTION: 0.10,        // aravacha va rels orasidagi ishqalanish koeffitsienti
    POLE1_FRICTION: 0.008,      // pastki tayoq shsarnirdagi ishqalanish (damping)
    POLE2_FRICTION: 0.008,      // yuqori tayoq sharnirdagi ishqalanish (damping)

    FORCE_MAGNITUDE: 12.0,      // N, diskret action uchun kuch qiymati
    CART_POSITION_LIMIT: 2.4,   // metr, aravacha markazdan chekka masofasi

    ANGLE_LIMIT_DEG: 36,        // gradus, bu burchakdan oshsa episode tugaydi
    CENTER_ZONE_DEG: 6,         // gradus, "markazga yaqin" bonus uchun chegaraviy burchak
    CENTER_ZONE_POSITION: 0.6,  // metr, aravachaning markazga yaqinlik chegarasi

    DT: 0.02,                   // sekund, simulyatsiya vaqt qadami (50 Hz)
    RK4_SUBSTEPS: 1,            // har frame nechta RK4 integratsiya bajarilishi

    PIXELS_PER_METER: 150       // render uchun metrdan pikselga o'tkazish koeffitsienti
});

// ==================== REWARD KONSTANTALARI ====================
const REWARD_CONSTANTS = Object.freeze({
    ALIVE_REWARD: 1.0,          // har frame ikkala tayoq tik bo'lsa
    CENTER_BONUS: 0.1,          // ikkala tayoq markazga yaqin bo'lsa qo'shimcha bonus
    POSITION_PENALTY: 0.05,     // markazdan uzoqlik uchun uzluksiz jarima koeffitsienti (drift'ni oldini oladi)
    FALL_PENALTY: -100.0,       // tayoqlardan biri yiqilsa
    OUT_OF_BOUNDS_PENALTY: -1.0 // aravacha chegaradan chiqsa
});

// ==================== RL KONSTANTALARI ====================
const RL_CONSTANTS = Object.freeze({
    STATE_SIZE: 6,              // [cartX, cartVel, theta1, w1, theta2, w2]
    ACTION_SIZE: 3,              // 0 = chap, 1 = hech narsa, 2 = o'ng

    HIDDEN_LAYER_SIZES: [64, 64],

    LEARNING_RATE: 0.001,
    GAMMA: 0.99,                 // discount factor

    EPSILON_START: 1.0,
    EPSILON_MIN: 0.05,
    EPSILON_DECAY: 0.0001,       // har o'yin qadamida chiziqli kamayish

    REPLAY_BUFFER_CAPACITY: 20000,
    BATCH_SIZE: 64,
    MIN_REPLAY_SIZE: 500,        // shu miqdorda tajriba to'planmaguncha train boshlanmaydi

    TARGET_UPDATE_INTERVAL: 400, // necha training qadamida target tarmoq yangilanadi

    MAX_STEPS_PER_EPISODE: 5000
});

// ==================== UMUMIY MATEMATIK FUNKSIYALAR ====================

/** Gradusni radianga o'tkazadi */
function degToRad(deg) {
    return (deg * Math.PI) / 180;
}

/** Radianni gradusga o'tkazadi */
function radToDeg(rad) {
    return (rad * 180) / Math.PI;
}

/** Qiymatni [min, max] oralig'ida ushlab turadi */
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/** [min, max) oralig'ida tasodifiy son qaytaradi */
function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

/** [0, n) oralig'ida tasodifiy butun son qaytaradi */
function randomInt(n) {
    return Math.floor(Math.random() * n);
}

/**
 * Massivdan tasodifiy indekslarni takrorlanmasdan tanlab olish
 * (Replay Buffer uchun mini-batch tanlashda ishlatiladi)
 */
function sampleIndices(populationSize, sampleSize) {
    const indices = [];
    const used = new Set();
    while (indices.length < sampleSize && used.size < populationSize) {
        const idx = randomInt(populationSize);
        if (!used.has(idx)) {
            used.add(idx);
            indices.push(idx);
        }
    }
    return indices;
}

/**
 * 3x3 chiziqli tenglamalar tizimini Cramer qoidasi orqali yechadi: A * x = b
 * A - 3x3 massiv (matritsa), b - uzunligi 3 bo'lgan vektor
 * Qaytaradi: uzunligi 3 bo'lgan yechim vektori [x0, x1, x2]
 */
function solveLinearSystem3x3(A, b) {
    const det3 = (m) =>
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    const detA = det3(A);

    // Agar matritsa singular bo'lsa (amalda deyarli sodir bo'lmaydi),
    // xavfsizlik uchun nolinchi tezlanishlarni qaytaramiz.
    if (Math.abs(detA) < 1e-12) {
        return [0, 0, 0];
    }

    const result = [0, 0, 0];
    for (let col = 0; col < 3; col++) {
        // A ustunini b bilan almashtirib, determinantni hisoblaymiz
        const Ai = [
            [A[0][0], A[0][1], A[0][2]],
            [A[1][0], A[1][1], A[1][2]],
            [A[2][0], A[2][1], A[2][2]]
        ];
        Ai[0][col] = b[0];
        Ai[1][col] = b[1];
        Ai[2][col] = b[2];
        result[col] = det3(Ai) / detA;
    }
    return result;
}

/** Ikki burchak orasidagi farqni [-PI, PI] oralig'iga normallashtiradi */
function normalizeAngle(angle) {
    let a = angle % (2 * Math.PI);
    if (a > Math.PI) a -= 2 * Math.PI;
    if (a < -Math.PI) a += 2 * Math.PI;
    return a;
}

// Node.js muhitida test qilish uchun (brauzerda bu bloк e'tiborsiz qoladi)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PHYSICS_CONSTANTS,
        REWARD_CONSTANTS,
        RL_CONSTANTS,
        degToRad,
        radToDeg,
        clamp,
        randomRange,
        randomInt,
        sampleIndices,
        solveLinearSystem3x3,
        normalizeAngle
    };
}
