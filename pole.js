/**
 * pole.js
 * ------------------------------------------------------------
 * Pole klassi bitta tayoqning fizik parametrlarini (massa, uzunlik,
 * inersiya momenti) va joriy holatini (burchak, burchak tezligi,
 * burchak tezlanishi) saqlaydi.
 * Bu klass ham cart.js kabi faqat ma'lumot konteyneri - hisob-kitob
 * qilmaydi. Uzunlikni ikkiga bo'lib, massa markazigacha bo'lgan
 * masofa va bir jinsli sterjen uchun inersiya momentini o'zi hisoblab
 * qo'yadi, chunki bular o'zgarmas geometrik xossalar.
 * ------------------------------------------------------------
 */

class Pole {
    /**
     * @param {number} mass - tayoq massasi (kg)
     * @param {number} length - tayoqning to'liq uzunligi (metr)
     * @param {number} angleLimitDeg - vertikaldan og'ish ruxsat etilgan maksimal burchak (gradus)
     */
    constructor(mass, length, angleLimitDeg = PHYSICS_CONSTANTS.ANGLE_LIMIT_DEG) {
        this.mass = mass;
        this.length = length;                      // to'liq uzunlik (L)
        this.centerOfMassDistance = length / 2;     // sharnirdan massa markazigacha masofa (l)

        // Bir jinsli sterjen (uniform rod) uchun massa markazi atrofidagi
        // inersiya momenti: I = (1/12) * m * L^2
        this.momentOfInertia = (1 / 12) * mass * length * length;

        this.angleLimitRad = degToRad(angleLimitDeg);

        this.reset();
    }

    /**
     * Tayoqni deyarli vertikal holatga qaytaradi.
     * @param {number} initialAngleNoise - radianlarda kichik tasodifiy boshlang'ich og'ish
     */
    reset(initialAngleNoise = 0.02) {
        this.angle = randomRange(-initialAngleNoise, initialAngleNoise); // radian, 0 = tik turgan holat
        this.angularVelocity = 0;      // rad/s
        this.angularAcceleration = 0;  // rad/s^2
    }

    /**
     * Tayoqning yangi holatini o'rnatadi.
     * Bu metodni faqat physics.js chaqiradi (integratsiyadan keyin).
     */
    setState(angle, angularVelocity, angularAcceleration) {
        this.angle = angle;
        this.angularVelocity = angularVelocity;
        this.angularAcceleration = angularAcceleration;
    }

    /** Tayoq ruxsat etilgan burchak chegarasidan chiqib ketganini tekshiradi */
    hasFallen() {
        return Math.abs(this.angle) > this.angleLimitRad;
    }

    /** Tayoq markazga (vertikalga) qanchalik yaqinligini tekshirish uchun */
    isNearVertical(thresholdRad) {
        return Math.abs(this.angle) < thresholdRad;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Pole };
}
