/**
 * physics.js
 * ------------------------------------------------------------
 * Double Inverted Pendulum on a Cart uchun TO'LIQ NOCHIZIQLI
 * dinamika. Tenglamalar Lagrange mexanikasi asosida noldan
 * chiqarilgan (soddalashtirilgan emas).
 *
 * Umumlashgan koordinatalar: q = [x, theta1, theta2]
 * Bu yerda:
 *   x       - aravacha pozitsiyasi
 *   theta1  - pastki tayoqning vertikaldan og'ish burchagi
 *   theta2  - yuqori tayoqning vertikaldan og'ish burchagi
 *             (pastki tayoq uchiga sharnirlangan)
 *
 * Harakat tenglamalari matritsa ko'rinishida:
 *
 *   M(q) * qddot = f(q, qdot, F)
 *
 * M(q) - 3x3 massa (inersiya) matritsasi:
 *   [ A         B1*cos(t1)         B2*cos(t2)        ]
 *   [ B1*cos(t1)  D1                D12*cos(t1-t2)   ]
 *   [ B2*cos(t2)  D12*cos(t1-t2)    D2               ]
 *
 * Bu yerda (m1,m2 - tayoq massalari, M - aravacha massasi,
 * l1,l2 - sharnirdan massa markazigacha masofa, L1 - pole1
 * to'liq uzunligi, I1,I2 - inersiya momentlari):
 *
 *   A   = M + m1 + m2
 *   B1  = m1*l1 + m2*L1
 *   B2  = m2*l2
 *   D1  = m1*l1^2 + m2*L1^2 + I1
 *   D2  = m2*l2^2 + I2
 *   D12 = m2*L1*l2
 *
 * Fizik hisob-kitobning to'liq matematik chiqarilishi Lagranjian
 * L = T - V ni umumlashgan koordinatalar bo'yicha differensiallash
 * orqali olingan (kinetik va potensial energiya kartezian
 * koordinatalarda yozilib, keyin q, qdot orqali ifodalangan).
 *
 * Integratsiya usuli: 4-tartibli Runge-Kutta (RK4), bu Euler
 * usuliga qaraganda ancha barqaror va aniqroq natija beradi,
 * ayniqsa notekis (stiff) tayoq dinamikasi uchun muhim.
 * ------------------------------------------------------------
 */

class DoubleCartPolePhysics {
    /**
     * @param {Cart} cart
     * @param {Pole} pole1 - pastki tayoq (aravachaga ulangan)
     * @param {Pole} pole2 - yuqori tayoq (pole1 uchiga ulangan)
     */
    constructor(cart, pole1, pole2) {
        this.cart = cart;
        this.pole1 = pole1;
        this.pole2 = pole2;

        this.gravity = PHYSICS_CONSTANTS.GRAVITY;
        this.cartFriction = PHYSICS_CONSTANTS.CART_FRICTION;
        this.pole1Friction = PHYSICS_CONSTANTS.POLE1_FRICTION;
        this.pole2Friction = PHYSICS_CONSTANTS.POLE2_FRICTION;
    }

    /**
     * Joriy to'liq holatni tekis massiv ko'rinishida qaytaradi:
     * [x, xdot, theta1, w1, theta2, w2]
     */
    getStateVector() {
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
     * Berilgan holat vektori va kuch uchun tizimning hosilasini hisoblaydi.
     * Qaytadigan qiymat: [xdot, xddot, w1, a1, w2, a2]
     *
     * @param {number[]} state - [x, xdot, theta1, w1, theta2, w2]
     * @param {number} force - aravachaga qo'yilgan gorizontal kuch (N)
     */
    computeDerivatives(state, force) {
        const [x, xdot, theta1, w1, theta2, w2] = state;

        const M = this.cart.mass;
        const m1 = this.pole1.mass;
        const m2 = this.pole2.mass;

        const l1 = this.pole1.centerOfMassDistance;
        const l2 = this.pole2.centerOfMassDistance;
        const L1 = this.pole1.length;

        const I1 = this.pole1.momentOfInertia;
        const I2 = this.pole2.momentOfInertia;

        const g = this.gravity;

        // --- Massa matritsasi koeffitsientlari (Lagranjiandan olingan) ---
        const A = M + m1 + m2;
        const B1 = m1 * l1 + m2 * L1;
        const B2 = m2 * l2;
        const D1 = m1 * l1 * l1 + m2 * L1 * L1 + I1;
        const D2 = m2 * l2 * l2 + I2;
        const D12 = m2 * L1 * l2;

        const cos1 = Math.cos(theta1);
        const sin1 = Math.sin(theta1);
        const cos2 = Math.cos(theta2);
        const sin2 = Math.sin(theta2);
        const cosDiff = Math.cos(theta1 - theta2);
        const sinDiff = Math.sin(theta1 - theta2);

        // Massa (inersiya) matritsasi M(q)
        const massMatrix = [
            [A, B1 * cos1, B2 * cos2],
            [B1 * cos1, D1, D12 * cosDiff],
            [B2 * cos2, D12 * cosDiff, D2]
        ];

        // O'ng tomon vektori f(q, qdot, F) - markazdan qochma (Coriolis)
        // va gravitatsiya hadlarini o'z ichiga oladi.
        const rhs = [
            force - this.cartFriction * xdot + B1 * sin1 * w1 * w1 + B2 * sin2 * w2 * w2,
            B1 * g * sin1 - D12 * sinDiff * w2 * w2 - this.pole1Friction * w1,
            B2 * g * sin2 + D12 * sinDiff * w1 * w1 - this.pole2Friction * w2
        ];

        // 3x3 chiziqli tizimni yechib, tezlanishlarni topamiz: M * qddot = rhs
        const [xddot, a1, a2] = solveLinearSystem3x3(massMatrix, rhs);

        return [xdot, xddot, w1, a1, w2, a2];
    }

    /**
     * Ikkita holat vektorini elementma-element qo'shadi: a + scale*b
     */
    static addScaled(a, b, scale) {
        return a.map((value, i) => value + b[i] * scale);
    }

    /**
     * Bitta vaqt qadamini 4-tartibli Runge-Kutta (RK4) usulida integratsiya qiladi
     * va cart/pole1/pole2 obyektlarining holatini yangilaydi.
     *
     * @param {number} force - joriy qadamda qo'llaniladigan kuch (N)
     * @param {number} dt - vaqt qadami (sekund)
     */
    step(force, dt) {
        const s0 = this.getStateVector();

        const k1 = this.computeDerivatives(s0, force);
        const s1 = DoubleCartPolePhysics.addScaled(s0, k1, dt / 2);

        const k2 = this.computeDerivatives(s1, force);
        const s2 = DoubleCartPolePhysics.addScaled(s0, k2, dt / 2);

        const k3 = this.computeDerivatives(s2, force);
        const s3 = DoubleCartPolePhysics.addScaled(s0, k3, dt);

        const k4 = this.computeDerivatives(s3, force);

        // Yakuniy holat: s0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        const finalState = s0.map((value, i) => {
            return value + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        });

        const [x, xdot, theta1, w1, theta2, w2] = finalState;

        // Tezlanishlarni (render/debug uchun) joriy kuch asosida qayta hisoblaymiz
        const derivativesAtFinal = this.computeDerivatives(finalState, force);

        this.cart.setState(x, xdot, derivativesAtFinal[1]);
        this.pole1.setState(theta1, w1, derivativesAtFinal[3]);
        this.pole2.setState(theta2, w2, derivativesAtFinal[5]);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DoubleCartPolePhysics };
}
