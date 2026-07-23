/**
 * renderer.js
 * ------------------------------------------------------------
 * Renderer klassi FAQAT chizish bilan shug'ullanadi.
 * U hech qanday fizik hisob-kitob qilmaydi - faqat berilgan
 * holat (state) qiymatlarini ekranga chiroyli tarzda tushiradi.
 * p5.js instance-mode funksiyalaridan foydalanadi (p parametri
 * orqali uzatiladi), shuning uchun global p5 funksiyalariga
 * bog'liq emas va boshqa loyihalarga ham ko'chirish oson.
 * ------------------------------------------------------------
 */

class Renderer {
    /**
     * @param {number} canvasWidth
     * @param {number} canvasHeight
     * @param {number} pixelsPerMeter
     */
    constructor(canvasWidth, canvasHeight, pixelsPerMeter = PHYSICS_CONSTANTS.PIXELS_PER_METER) {
        this.width = canvasWidth;
        this.height = canvasHeight;
        this.pixelsPerMeter = pixelsPerMeter;

        // Rels (track) qayerda joylashishini belgilaymiz (ekranning pastroq qismi)
        this.trackY = this.height * 0.62;

        // Rang sxemasi (bitta joyda saqlanadi - dizaynni o'zgartirish oson bo'lishi uchun)
        this.colors = {
            background: [18, 20, 26],
            track: [60, 64, 76],
            cart: [90, 170, 255],
            cartStroke: [140, 200, 255],
            pole1: [255, 120, 100],
            pole2: [120, 255, 160],
            joint: [230, 230, 240],
            text: [220, 220, 230],
            comMarker: [255, 210, 60],
            forceVector: [255, 90, 90],
            velocityVector: [90, 220, 255],
            limitLine: [255, 70, 70],
            centerZone: [80, 90, 60]
        };
    }

    /** Metr koordinatasini ekran (piksel) X koordinatasiga o'giradi */
    worldToScreenX(worldX) {
        return this.width / 2 + worldX * this.pixelsPerMeter;
    }

    /** Asosiy chizish funksiyasi - har frame chaqiriladi */
    render(p, environment, debugMode) {
        const { cart, pole1, pole2 } = environment;

        p.background(...this.colors.background);

        this._drawTrack(p, environment.cart.positionLimit);
        this._drawCenterZone(p);

        const cartScreenX = this.worldToScreenX(cart.position);
        const cartScreenY = this.trackY;

        this._drawCart(p, cartScreenX, cartScreenY, cart);

        // Pastki tayoq (pole1) sharniri aravacha markazida
        const joint1 = { x: cartScreenX, y: cartScreenY };
        const tip1 = this._drawPole(p, joint1, pole1, this.colors.pole1);

        // Yuqori tayoq (pole2) sharniri pole1 uchida
        const tip2 = this._drawPole(p, tip1, pole2, this.colors.pole2);

        this._drawJoint(p, joint1);
        this._drawJoint(p, tip1);
        this._drawJoint(p, tip2, true);

        if (debugMode) {
            this._drawDebugOverlay(p, environment, joint1, tip1, tip2);
        }
    }

    /** Rels chizig'ini va chegara belgilarini chizadi */
    _drawTrack(p, positionLimit) {
        p.stroke(...this.colors.track);
        p.strokeWeight(4);
        p.line(0, this.trackY, this.width, this.trackY);

        // Chegara chiziqlari (cart limit)
        const leftLimitX = this.worldToScreenX(-positionLimit);
        const rightLimitX = this.worldToScreenX(positionLimit);

        p.stroke(...this.colors.limitLine);
        p.strokeWeight(2);
        p.line(leftLimitX, this.trackY - 40, leftLimitX, this.trackY + 40);
        p.line(rightLimitX, this.trackY - 40, rightLimitX, this.trackY + 40);
    }

    /** "Markazga yaqin" bonus zonasini fon sifatida ko'rsatadi */
    _drawCenterZone(p) {
        const zoneWidth = PHYSICS_CONSTANTS.CENTER_ZONE_POSITION * 2 * this.pixelsPerMeter;
        const zoneX = this.width / 2 - zoneWidth / 2;
        p.noStroke();
        p.fill(...this.colors.centerZone, 60);
        p.rect(zoneX, this.trackY - 200, zoneWidth, 210);
    }

    /** Aravachani to'rtburchak sifatida chizadi */
    _drawCart(p, screenX, screenY, cart) {
        const w = cart.width * this.pixelsPerMeter;
        const h = cart.height * this.pixelsPerMeter;

        p.push();
        p.translate(screenX, screenY);
        p.stroke(...this.colors.cartStroke);
        p.strokeWeight(2);
        p.fill(...this.colors.cart);
        p.rectMode(p.CENTER);
        p.rect(0, 0, w, h, 6);

        // Ikkita g'ildirak (faqat vizual, fizikaga ta'sir qilmaydi)
        p.fill(30);
        p.noStroke();
        p.circle(-w / 3, h / 2, 10);
        p.circle(w / 3, h / 2, 10);
        p.pop();
    }

    /**
     * Bitta tayoqni chizadi (sharnirdan boshlanib, burchak bo'yicha yuqoriga qarab).
     * @returns {{x:number, y:number}} tayoqning uchki nuqtasi (keyingi sharnir uchun)
     */
    _drawPole(p, joint, pole, color) {
        const lengthPx = pole.length * this.pixelsPerMeter;

        // theta = 0 -> tik yuqoriga; musbat theta -> o'ngga og'ish
        const tipX = joint.x + Math.sin(pole.angle) * lengthPx;
        const tipY = joint.y - Math.cos(pole.angle) * lengthPx;

        p.stroke(...color);
        p.strokeWeight(8);
        p.strokeCap(p.ROUND);
        p.line(joint.x, joint.y, tipX, tipY);

        return { x: tipX, y: tipY };
    }

    /** Sharnir nuqtasini kichik doira sifatida chizadi */
    _drawJoint(p, point, isTip = false) {
        p.noStroke();
        p.fill(...this.colors.joint);
        p.circle(point.x, point.y, isTip ? 10 : 12);
    }

    /**
     * Debug rejimida qo'shimcha ma'lumotlarni chizadi:
     * massa markazlari, kuch va tezlik vektorlari, burchak yoylari, state qiymatlari.
     */
    _drawDebugOverlay(p, environment, joint1, tip1, tip2) {
        const { cart, pole1, pole2 } = environment;

        // ---- Massa markazlari (Center of Mass) ----
        const com1 = {
            x: joint1.x + Math.sin(pole1.angle) * pole1.centerOfMassDistance * this.pixelsPerMeter,
            y: joint1.y - Math.cos(pole1.angle) * pole1.centerOfMassDistance * this.pixelsPerMeter
        };
        const com2 = {
            x: tip1.x + Math.sin(pole2.angle) * pole2.centerOfMassDistance * this.pixelsPerMeter,
            y: tip1.y - Math.cos(pole2.angle) * pole2.centerOfMassDistance * this.pixelsPerMeter
        };

        p.noStroke();
        p.fill(...this.colors.comMarker);
        p.circle(com1.x, com1.y, 8);
        p.circle(com2.x, com2.y, 8);

        // ---- Kuch vektori (aravachaga qo'yilgan gorizontal kuch) ----
        const forceScale = 3.0; // vizual masshtab
        const cartScreenX = this.worldToScreenX(cart.position);
        p.stroke(...this.colors.forceVector);
        p.strokeWeight(3);
        const forceEndX = cartScreenX + environment.lastForce * forceScale;
        this._drawArrow(p, cartScreenX, this.trackY, forceEndX, this.trackY);

        // ---- Tezlik vektorlari (tayoq uchlari uchun) ----
        const velScale = 15.0;
        p.stroke(...this.colors.velocityVector);
        p.strokeWeight(2);
        const tip1VelX = tip1.x + Math.cos(pole1.angle) * pole1.angularVelocity * pole1.length * this.pixelsPerMeter * velScale * 0.01;
        const tip1VelY = tip1.y + Math.sin(pole1.angle) * pole1.angularVelocity * pole1.length * this.pixelsPerMeter * velScale * 0.01;
        this._drawArrow(p, tip1.x, tip1.y, tip1VelX, tip1VelY);

        // ---- Burchak yoylari (vertikaldan og'ish) ----
        p.noFill();
        p.stroke(...this.colors.pole1, 180);
        p.strokeWeight(1.5);
        p.arc(joint1.x, joint1.y, 60, 60, -Math.PI / 2, -Math.PI / 2 + pole1.angle);

        p.stroke(...this.colors.pole2, 180);
        p.arc(tip1.x, tip1.y, 40, 40, -Math.PI / 2, -Math.PI / 2 + pole2.angle);

        // ---- State qiymatlari matn ko'rinishida ----
        p.noStroke();
        p.fill(...this.colors.text);
        p.textFont('monospace');
        p.textSize(12);
        p.textAlign(p.LEFT, p.TOP);

        const lines = [
            `x       = ${cart.position.toFixed(3)} m`,
            `xdot    = ${cart.velocity.toFixed(3)} m/s`,
            `xddot   = ${cart.acceleration.toFixed(3)} m/s^2`,
            `theta1  = ${radToDeg(pole1.angle).toFixed(2)} deg`,
            `w1      = ${radToDeg(pole1.angularVelocity).toFixed(2)} deg/s`,
            `a1      = ${radToDeg(pole1.angularAcceleration).toFixed(2)} deg/s^2`,
            `theta2  = ${radToDeg(pole2.angle).toFixed(2)} deg`,
            `w2      = ${radToDeg(pole2.angularVelocity).toFixed(2)} deg/s`,
            `a2      = ${radToDeg(pole2.angularAcceleration).toFixed(2)} deg/s^2`,
            `force   = ${environment.lastForce.toFixed(2)} N`
        ];

        const panelX = 10;
        const panelY = 10;
        p.fill(0, 0, 0, 140);
        p.rect(panelX - 6, panelY - 6, 210, lines.length * 16 + 10, 4);

        p.fill(...this.colors.text);
        lines.forEach((line, i) => {
            p.text(line, panelX, panelY + i * 16);
        });
    }

    /** Ikki nuqta orasida o'q (arrow) chizadi - vektorlarni ko'rsatish uchun */
    _drawArrow(p, x1, y1, x2, y2) {
        p.line(x1, y1, x2, y2);
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const headLength = 8;
        p.push();
        p.translate(x2, y2);
        p.rotate(angle);
        p.line(0, 0, -headLength, headLength / 2);
        p.line(0, 0, -headLength, -headLength / 2);
        p.pop();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Renderer };
}
