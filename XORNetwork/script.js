/* ============================================================
    XOR NEURAL NETWORK — Silicon Zen edition
    ============================================================ */

// ── XOR DATA ──
const data = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 0 }
];

// ── DOM REFS ──
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logDiv = document.getElementById('log');
const epochSpan = document.getElementById('epochDisplay');
const lossSpan = document.getElementById('lossDisplay');
const statusLabel = document.getElementById('statusLabel');

// ── ACTIVATION & DERIVATIVES ──
function tanh(x) { return Math.tanh(x); }
function tanhDerivative(x) { return 1 - x * x; }

// ── NEURAL NETWORK CLASS (XOR) ──
class XORNetwork {
    constructor() {
        // Random weights between -1 and 1
        this.w11 = Math.random() * 2 - 1;
        this.w12 = Math.random() * 2 - 1;
        this.w21 = Math.random() * 2 - 1;
        this.w22 = Math.random() * 2 - 1;
        this.wo1 = Math.random() * 2 - 1;
        this.wo2 = Math.random() * 2 - 1;
        // Biases
        this.bh1 = Math.random() * 2 - 1;
        this.bh2 = Math.random() * 2 - 1;
        this.bout = Math.random() * 2 - 1;
        this.lr = 0.1;
    }

    // Forward pass
    forward(x1, x2) {
        const z1 = x1 * this.w11 + x2 * this.w12 + this.bh1;
        const z2 = x1 * this.w21 + x2 * this.w22 + this.bh2;
        const h1 = tanh(z1);
        const h2 = tanh(z2);
        const z_out = h1 * this.wo1 + h2 * this.wo2 + this.bout;
        const out = tanh(z_out);
        return { z1, z2, h1, h2, out };
    }

    // Train on a single sample
    train(x1, x2, target) {
        const { z1, z2, h1, h2, out } = this.forward(x1, x2);
        const loss = (out - target) ** 2;

        // Gradients (backprop)
        const dLoss_dOut = 2 * (out - target);
        const dOut_dZout = tanhDerivative(out);
        const dLoss_dZout = dLoss_dOut * dOut_dZout;

        const dW_o1 = dLoss_dZout * h1;
        const dW_o2 = dLoss_dZout * h2;
        const dB_out = dLoss_dZout;

        const dLoss_dH1 = dLoss_dZout * this.wo1;
        const dLoss_dH2 = dLoss_dZout * this.wo2;
        const dH1_dZ1 = tanhDerivative(h1);
        const dH2_dZ2 = tanhDerivative(h2);

        const dW_11 = dLoss_dH1 * dH1_dZ1 * x1;
        const dW_12 = dLoss_dH1 * dH1_dZ1 * x2;
        const dW_21 = dLoss_dH2 * dH2_dZ2 * x1;
        const dW_22 = dLoss_dH2 * dH2_dZ2 * x2;
        const dB_h1 = dLoss_dH1 * dH1_dZ1;
        const dB_h2 = dLoss_dH2 * dH2_dZ2;

        // Update weights
        this.w11 -= this.lr * dW_11;
        this.w12 -= this.lr * dW_12;
        this.w21 -= this.lr * dW_21;
        this.w22 -= this.lr * dW_22;
        this.wo1 -= this.lr * dW_o1;
        this.wo2 -= this.lr * dW_o2;
        this.bh1 -= this.lr * dB_h1;
        this.bh2 -= this.lr * dB_h2;
        this.bout -= this.lr * dB_out;

        return {
            weights: [this.w11, this.w12, this.w21, this.w22, this.wo1, this.wo2, this.bh1, this.bh2, this.bout],
            h1, h2, out, loss
        };
    }
}

// ── NETWORK INSTANCE & STATE ──
let net = new XORNetwork();
let epoch = 0;
let running = false;
let timer = null;
const START_EPOCHS = 1500;
let MAX_EPOCHS = START_EPOCHS;

// ── DRAWING NETWORK (adapted for XOR architecture) ──
function drawNetwork(weights, loss) {
    const W = canvas.width,
        H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Node positions (input, hidden, output)
    const x1Pos = { x: 100, y: 100 };
    const x2Pos = { x: 100, y: 200 };
    const h1Pos = { x: 300, y: 75 };
    const h2Pos = { x: 300, y: 225 };
    const outPos = { x: 550, y: 150 };

    // Helper: draw a weighted connection
    function drawLine(x1, y1, x2, y2, weight, label) {
        const thickness = Math.min(6, Math.abs(weight) * 4 + 1);
        const hue = weight > 0 ? 120 : 0;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${Math.min(1, Math.abs(weight) + 0.2)})`;
        ctx.lineWidth = thickness;
        ctx.stroke();

        const mx = (x1 + x2) / 2 - 15;
        const my = (y1 + y2) / 2 - 14;
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(label + '=' + weight.toFixed(2), mx, my);
    }

    // Extract weights
    const [w11, w12, w21, w22, wo1, wo2, bh1, bh2, bout] = weights;

    // Input → Hidden
    drawLine(x1Pos.x, x1Pos.y, h1Pos.x, h1Pos.y, w11, 'w11');
    drawLine(x1Pos.x, x1Pos.y, h2Pos.x, h2Pos.y, w12, 'w12');
    drawLine(x2Pos.x, x2Pos.y, h1Pos.x, h1Pos.y, w21, 'w21');
    drawLine(x2Pos.x, x2Pos.y, h2Pos.x, h2Pos.y, w22, 'w22');

    // Hidden → Output
    drawLine(h1Pos.x, h1Pos.y, outPos.x, outPos.y, wo1, 'wo1');
    drawLine(h2Pos.x, h2Pos.y, outPos.x, outPos.y, wo2, 'wo2');

    // Bias labels
    ctx.fillStyle = '#ff7b72';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('b1=' + bh1.toFixed(2), h1Pos.x + 40, h1Pos.y - 20);
    ctx.fillText('b2=' + bh2.toFixed(2), h2Pos.x + 40, h2Pos.y - 20);
    ctx.fillText('bo=' + bout.toFixed(2), outPos.x + 40, outPos.y - 20);

    // Helper: draw a node
    function drawNode(x, y, label, sublabel = '', borderColor = '#58a6ff', fill = '#1f2937') {
        const grad = ctx.createRadialGradient(x, y, 0, x, y, 30);
        grad.addColorStop(0, `rgba(88,166,255,0.1)`);
        grad.addColorStop(1, 'rgba(88,166,255,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, 30, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = fill;
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 22, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f0f6fc';
        ctx.font = 'bold 13px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y - 3);
        if (sublabel) {
            ctx.fillStyle = '#8b949e';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillText(sublabel, x, y + 16);
        }
    }

    drawNode(x1Pos.x, x1Pos.y, 'X1', '');
    drawNode(x2Pos.x, x2Pos.y, 'X2', '');
    drawNode(h1Pos.x, h1Pos.y, 'H1', '', '#ff7b72');
    drawNode(h2Pos.x, h2Pos.y, 'H2', '', '#ff7b72');

    // Output: show sigmoid on sample (1,1)
    const zDemo = 1 * wo1 + 1 * wo2 + bout;
    const outDemo = tanh(zDemo);
    drawNode(outPos.x, outPos.y, 'OUT', outDemo.toFixed(2), '#3fb950');

    // Loss on canvas
    ctx.fillStyle = '#f0883e';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${loss.toFixed(6)}`, 20, 280);
}

// ── LOG HELPER ──
function logMessage(html, cls = '') {
    const line = document.createElement('div');
    line.innerHTML = html;
    if (cls) line.className = cls;
    logDiv.appendChild(line);
    logDiv.scrollTop = logDiv.scrollHeight;
}

// ── TRAINING LOOP ──
function trainLoop() {
    if (!running) return;

    // Train on a random sample
    const sample = data[Math.floor(Math.random() * data.length)];
    const result = net.train(sample.x1, sample.x2, sample.y);
    epoch++;

    epochSpan.textContent = epoch;
    lossSpan.textContent = result.loss.toFixed(6);
    drawNetwork(result.weights, result.loss);

    if (epoch % 10 === 0) {
        logMessage(
            `<span class="weight">[Epoch ${epoch}]</span> Loss: <span class="loss">${result.loss.toFixed(6)}</span> | w11: ${result.weights[0].toFixed(3)}, w12: ${result.weights[1].toFixed(3)}, wo1: ${result.weights[4].toFixed(3)}`
        );
    }

    if (epoch < MAX_EPOCHS) {
        timer = setTimeout(trainLoop, 10);
    } else {
        running = false;
        statusLabel.textContent = 'done ✓';
        logMessage(`<span class="done">DONE!</span> Testing XOR gate:`);

        let passed = true;
        data.forEach(d => {
            const f = net.forward(d.x1, d.x2);
            const pred = Math.round(f.out);
            logMessage(
                `  ${d.x1} XOR ${d.x2} = ${pred} (raw: ${f.out.toFixed(4)})`
            );
            if (pred !== d.y) passed = false;
        });

        if (passed) {
            logMessage(`<span style="color: #3fb950;"> XOR Neural Network PASSED</span>`);
        } else {
            logMessage(`<span style="color: #b93f3f;"> XOR Neural Network FAILED</span>`);
        }

        logMessage(
            `<span class="weight">Final Weights:</span> [${result.weights.map(w=>w.toFixed(3)).join(', ')}]`
        );
    }
}

// ── CONTROLS ──
function resetPerceptron() {
    if (timer) { clearTimeout(timer);
        timer = null; }
    running = false;
    MAX_EPOCHS = START_EPOCHS;
    epoch = 0;
    net = new XORNetwork();
    epochSpan.textContent = '0';
    lossSpan.textContent = '1.000000';
    statusLabel.textContent = 'reset';
    logDiv.innerHTML = '';
    logMessage(`<span class="prompt">■</span> <span class="muted">Reset. Ready to train.</span>`);
    drawNetwork([
        net.w11, net.w12, net.w21, net.w22,
        net.wo1, net.wo2,
        net.bh1, net.bh2, net.bout
    ], 1.0);
}

function startTraining() {
    if (running) return;
    if (epoch >= MAX_EPOCHS) {
        MAX_EPOCHS += 500; // extend training if already done
    }
    running = true;
    statusLabel.textContent = 'training…';
    logMessage(`<span class="prompt">▶</span> <span class="ok">Training started.</span>`);
    trainLoop();
}

// ── INIT ──
resetPerceptron();

document.getElementById('resetBtn').addEventListener('click', resetPerceptron);
document.getElementById('trainBtn').addEventListener('click', startTraining);