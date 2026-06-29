/* ============================================================
    AND GATE PERCEPTRON
    ============================================================ */

// ── DATA ──
const data = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 0 },
    { x1: 1, x2: 0, y: 0 },
    { x1: 1, x2: 1, y: 1 }
];

// ── NETWORK PARAMETERS ──
let w1 = Math.random() * 2 - 1;
let w2 = Math.random() * 2 - 1;
let b = Math.random() * 2 - 1;
const lr = 0.1;

// ── DOM REFS ──
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logDiv = document.getElementById('log');
const epochSpan = document.getElementById('epochDisplay');
const lossSpan = document.getElementById('lossDisplay');
const statusLabel = document.getElementById('statusLabel');

// ── ACTIVATION ──
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ── DRAWING THE NETWORK ──
function drawNetwork(currentW1, currentW2, currentB, loss) {
    const W = canvas.width,
        H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Node positions
    const x1Pos = { x: 80, y: 70 };
    const x2Pos = { x: 80, y: 150 };
    const sumPos = { x: 280, y: 110 };
    const outPos = { x: 500, y: 110 };

    // ── Helper: draw a weighted connection ──
    function drawLine(x1, y1, x2, y2, weight, label, color = '#58a6ff') {
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

    // ── Connections ──
    drawLine(x1Pos.x, x1Pos.y, sumPos.x, sumPos.y - 20, currentW1, 'w1');
    drawLine(x2Pos.x, x2Pos.y, sumPos.x, sumPos.y + 20, currentW2, 'w2');
    // Σ → OUT (fixed weight 1)
    const activationWeight = 1.0;
    drawLine(sumPos.x, sumPos.y, outPos.x, outPos.y, activationWeight, 'σ', '#58a6ff');

    // Bias label
    ctx.fillStyle = '#ff7b72';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('b=' + currentB.toFixed(2), sumPos.x + 40, sumPos.y - 20);

    // ── Draw nodes ──
    function drawNode(x, y, label, sublabel = '', borderColor = '#58a6ff', fill = '#1f2937') {
        // Glow
        const grad = ctx.createRadialGradient(x, y, 0, x, y, 30);
        grad.addColorStop(0, `rgba(88,166,255,0.1)`);
        grad.addColorStop(1, 'rgba(88,166,255,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, 30, 0, Math.PI * 2);
        ctx.fill();

        // Circle
        ctx.fillStyle = fill;
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 22, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Label
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
    drawNode(sumPos.x, sumPos.y, 'Σ', "", '#ff7b72');

    // Output: show sigmoid on sample (1,1)
    const zDemo = 1 * currentW1 + 1 * currentW2 + currentB;
    const outDemo = sigmoid(zDemo);
    drawNode(outPos.x, outPos.y, 'OUT', outDemo.toFixed(2), '#3fb950');

    // Loss on canvas
    ctx.fillStyle = '#f0883e';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${loss.toFixed(6)}`, 20, 200);
}

// ── TRAINING STEP ──
function trainStep() {
    let totalLoss = 0;
    data.forEach(sample => {
        const z = sample.x1 * w1 + sample.x2 * w2 + b;
        const pred = sigmoid(z);
        const loss = (pred - sample.y) ** 2;
        totalLoss += loss;

        const dLoss_dPred = 2 * (pred - sample.y);
        const dPred_dZ = pred * (1 - pred);
        const dZ_dW1 = sample.x1;
        const dZ_dW2 = sample.x2;
        const dZ_dB = 1;

        w1 -= lr * dLoss_dPred * dPred_dZ * dZ_dW1;
        w2 -= lr * dLoss_dPred * dPred_dZ * dZ_dW2;
        b -= lr * dLoss_dPred * dPred_dZ * dZ_dB;
    });
    return totalLoss / data.length;
}

// ── STATE ──
let epochs = 0;
let running = false;
let timer = null;
const START_EPOCHS = 100
let MAX_EPOCHS = START_EPOCHS;


// ── LOG HELPER (Silicon Zen CLI style) ──
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

    const avgLoss = trainStep();
    epochs++;

    epochSpan.textContent = epochs;
    lossSpan.textContent = avgLoss.toFixed(6);
    drawNetwork(w1, w2, b, avgLoss);

    if (epochs % 5 === 0) {
        logMessage(
            `<span class="weight">[Epoch ${epochs}]</span> Loss: <span class="loss">${avgLoss.toFixed(6)}</span> | w1: ${w1.toFixed(3)}, w2: ${w2.toFixed(3)}, b: ${b.toFixed(3)}`
            );
    }

    if (epochs < MAX_EPOCHS) {
        timer = setTimeout(trainLoop, 100);
    } else {
        running = false;
        statusLabel.textContent = 'done ✓';
        logMessage(`<span class="done">DONE!</span> Testing AND gate:`);

        let passed = true;

        data.forEach(d => {
            const z = d.x1 * w1 + d.x2 * w2 + b;
            const p = sigmoid(z);
            const prediction = Math.round(p);
            logMessage(
                `  ${d.x1} AND ${d.x2} = ${prediction} (raw: ${p.toFixed(4)})`
                );

            if (prediction !== d.y) {
                passed = false;
            }
        });

        if (passed) {
            logMessage(`<span style="color: #3fb950;"> AND Perceptron PASSED</span>`)
        } else {
            logMessage(`<span style="color: #b93f3f;"> AND Perceptron FAILED</span>`)
        }

        logMessage(
            `<span class="weight">Final Weights:</span> [${w1.toFixed(3)}, ${w2.toFixed(3)}, ${b.toFixed(3)}]`
            );
    }
}

// ── CONTROLS ──
function resetPerceptron() {
    if (timer) { clearTimeout(timer);
        timer = null; }
    running = false;
    MAX_EPOCHS = START_EPOCHS;
    epochs = 0;
    w1 = Math.random() * 2 - 1;
    w2 = Math.random() * 2 - 1;
    b = Math.random() * 2 - 1;
    epochSpan.textContent = '0';
    lossSpan.textContent = '1.000000';
    statusLabel.textContent = 'reset';
    logMessage(`<span class="prompt">■</span> <span class="muted">Reset. Ready to train.</span><br/><br/>`);
    drawNetwork(w1, w2, b, 1.0);
}

function startTraining() {
    if (running) return;
    if (epochs >= MAX_EPOCHS) {
        MAX_EPOCHS += 100
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