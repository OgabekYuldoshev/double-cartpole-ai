/**
 * sketch.js
 * ------------------------------------------------------------
 * Loyihaning kirish nuqtasi (entry point). p5.js "instance mode"da
 * ishlatilgan - bu global funksiyalar (setup/draw) bilan
 * to'qnashuvlarning oldini oladi va kodni modulli qiladi.
 *
 * Bu fayl:
 *   - Environment, Agent, Trainer, Renderer obyektlarini yaratadi
 *   - HTML UI elementlarini (tugmalar, sliderlar) RL sikliga bog'laydi
 *   - Har frame kerakli miqdorda simulyatsiya qadamlarini bajaradi
 *     (Simulation Speed slideriga qarab) va bitta marta chizadi
 * ------------------------------------------------------------
 */

const sketch = (p) => {
    // ---- Asosiy obyektlar ----
    let environment;
    let agent;
    let trainer;
    let renderer;

    // ---- UI holati ----
    let debugMode = false;
    let simulationSpeed = 20; // har frame nechta trainer.update() chaqirilishi

    const CANVAS_WIDTH = 900;
    const CANVAS_HEIGHT = 520;

    p.setup = () => {
        const canvas = p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.parent('canvas-holder');

        environment = new CartPoleEnvironment();
        agent = new DQNAgent();
        trainer = new Trainer(agent, environment);
        renderer = new Renderer(CANVAS_WIDTH, CANVAS_HEIGHT);

        p.frameRate(60);

        setupUIHandlers();

        // Sahifa yopilishidan oldin joriy holatni saqlaymiz - episode oxirigacha
        // kutmasdan, mid-episode holatda ham reload/tab close paytida yo'qotmaslik uchun
        window.addEventListener('beforeunload', () => {
            trainer.saveState();
        });
    };

    p.draw = () => {
        // Simulation Speed bo'yicha bir necha marta RL qadamini bajaramiz,
        // lekin ekranga faqat bitta marta chizamiz (tezlashtirilgan training uchun)
        for (let i = 0; i < simulationSpeed; i++) {
            trainer.update();
        }

        environment.render(p, renderer, debugMode);
        updateStatsPanel();
    };

    /** HTML UI elementlariga event listenerlarni ulaydi */
    function setupUIHandlers() {
        const trainingToggleBtn = document.getElementById('training-toggle-btn');
        trainingToggleBtn.addEventListener('click', () => {
            const newState = !agent.isTraining;
            agent.setTraining(newState);
            trainingToggleBtn.textContent = newState ? 'Training: ON' : 'Training: OFF';
            trainingToggleBtn.classList.toggle('active', newState);
        });

        const debugToggleBtn = document.getElementById('debug-toggle-btn');
        debugToggleBtn.addEventListener('click', () => {
            debugMode = !debugMode;
            debugToggleBtn.textContent = debugMode ? 'Debug: ON' : 'Debug: OFF';
            debugToggleBtn.classList.toggle('active', debugMode);
        });

        const resetBtn = document.getElementById('reset-btn');
        resetBtn.addEventListener('click', () => {
            trainer.fullReset();
        });

        const speedSlider = document.getElementById('speed-slider');
        const speedValueLabel = document.getElementById('speed-value');
        speedSlider.addEventListener('input', (e) => {
            simulationSpeed = parseInt(e.target.value, 10);
            speedValueLabel.textContent = `${simulationSpeed}x`;
        });
    }

    /** Statistika panelini (FPS, Episode, Reward va h.k.) yangilaydi */
    function updateStatsPanel() {
        const stats = trainer.getStats();

        document.getElementById('stat-fps').textContent = p.frameRate().toFixed(1);
        document.getElementById('stat-episode').textContent = stats.episode;
        document.getElementById('stat-reward').textContent = stats.currentEpisodeReward.toFixed(2);
        document.getElementById('stat-last-reward').textContent = stats.lastEpisodeReward.toFixed(2);
        document.getElementById('stat-best-reward').textContent = stats.bestEpisodeReward.toFixed(2);
        document.getElementById('stat-total-reward').textContent = stats.totalReward.toFixed(1);
        document.getElementById('stat-epsilon').textContent = stats.epsilon.toFixed(3);
        document.getElementById('stat-generation').textContent = stats.generation;
        document.getElementById('stat-step').textContent = stats.stepCount;
        document.getElementById('stat-loss').textContent = stats.loss.toFixed(4);
    }
};

// p5.js instance yaratish - global window.p5 konstruktoridan foydalaniladi
new p5(sketch);
