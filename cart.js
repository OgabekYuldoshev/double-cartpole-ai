/**
 * cart.js
 * ------------------------------------------------------------
 * Cart klassi aravachaning fizik parametrlarini (massa, o'lcham)
 * va joriy holatini (pozitsiya, tezlik, tezlanish) saqlaydi.
 * Bu klass HECH QANDAY fizika hisoblamaydi - u faqat ma'lumot
 * konteyneri (data container) sifatida ishlaydi.
 * Barcha dinamika hisob-kitoblari physics.js da bajariladi.
 * ------------------------------------------------------------
 */

class Cart {
    /**
     * @param {number} mass - aravacha massasi (kg)
     * @param {number} positionLimit - markazdan chekkagacha ruxsat etilgan masofa (metr)
     */
    constructor(mass = PHYSICS_CONSTANTS.CART_MASS, positionLimit = PHYSICS_CONSTANTS.CART_POSITION_LIMIT) {
        this.mass = mass;
        this.positionLimit = positionLimit;

        // Render uchun vizual o'lchamlar (metr birligida, keyin pikselga o'giriladi)
        this.width = 0.5;
        this.height = 0.3;

        this.reset();
    }

    /** Aravachani boshlang'ich holatga qaytaradi (kichik tasodifiy tebranish bilan) */
    reset() {
        this.position = randomRange(-0.02, 0.02); // metr
        this.velocity = 0;                        // m/s
        this.acceleration = 0;                    // m/s^2
    }

    /**
     * Aravachaning yangi holatini o'rnatadi.
     * Bu metodni faqat physics.js chaqiradi (integratsiyadan keyin).
     */
    setState(position, velocity, acceleration) {
        this.position = position;
        this.velocity = velocity;
        this.acceleration = acceleration;
    }

    /** Aravacha ruxsat etilgan chegaradan chiqib ketganini tekshiradi */
    isOutOfBounds() {
        return Math.abs(this.position) > this.positionLimit;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Cart };
}
