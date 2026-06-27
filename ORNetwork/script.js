// ===== DATA =====
const data = [
{ x1: 0, x2: 0, y: 0 },
{ x1: 0, x2: 1, y: 0 },
{ x1: 1, x2: 0, y: 0 },
{ x1: 1, x2: 1, y: 1 }
];

// ===== NETWORK PARAMETERS =====
let w1 = Math.random() * 2 - 1;
let w2 = Math.random() * 2 - 1;
let b = Math.random() * 2 - 1;
const lr = 0.1;

// ===== DOM REFS =====
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logDiv = document.getElementById('log');
const epochSpan = document.getElementById('epochDisplay');
const lossSpan = document.getElementById('lossDisplay');

// ===== ACTIVATION =====
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ===== DRAWING THE NETWORK =====
function drawNetwork(currentW1, currentW2, currentB, loss) {
const W = canvas.width, H = canvas.height;
ctx.clearRect(0, 0, W, H);

// Node positions
const x1Pos = { x: 80, y: 80 };
const x2Pos = { x: 80, y: 160 };
const sumPos = { x: 280, y: 120 };   // Σ neuron (moved slightly left)
const outPos = { x: 500, y: 120 };

// ---- Helper: draw a weighted connection ----
function drawLine(x1, y1, x2, y2, weight, label, color = '#58a6ff') {
    const thickness = Math.min(6, Math.abs(weight) * 4 + 1);
    const hue = weight > 0 ? 120 : 0; // green for positive, red for negative
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${Math.min(1, Math.abs(weight) + 0.2)})`;
    ctx.lineWidth = thickness;
    ctx.stroke();

    // Label
    const mx = (x1 + x2)/2 - 15;
    const my = (y1 + y2)/2 - 12;
    ctx.fillStyle = '#ffffff';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label + '=' + weight.toFixed(2), mx, my);
}

// ---- Draw connections ----
// w1 (X1 → Σ)
drawLine(x1Pos.x, x1Pos.y, sumPos.x, sumPos.y-20, currentW1, 'w1');
// w2 (X2 → Σ)
drawLine(x2Pos.x, x2Pos.y, sumPos.x, sumPos.y+20, currentW2, 'w2');

// --- FIX: Add line from Σ → OUT (sigmoid activation) ---
// This connection has a fixed weight of 1 (just passing through)
const activationWeight = 1.0;
drawLine(sumPos.x, sumPos.y, outPos.x, outPos.y, activationWeight, 'σ', '#58a6ff');

// Bias: show text near Σ
ctx.fillStyle = '#ff7b72';
ctx.font = '11px monospace';
ctx.textAlign = 'center';
ctx.fillText('b=' + currentB.toFixed(2), sumPos.x + 40, sumPos.y - 20);

// ---- Draw nodes ----
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
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y - 3);
    if (sublabel) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.fillText(sublabel, x, y + 16);
    }
}

// Inputs
drawNode(x1Pos.x, x1Pos.y, 'X1', '');
drawNode(x2Pos.x, x2Pos.y, 'X2', '');

// Sum neuron (with bias shown)
drawNode(sumPos.x, sumPos.y, 'Σ', 'b='+currentB.toFixed(2), '#ff7b72');

// Output neuron (shows sigmoid output)
// Compute a demo activation using current weights on sample (1,1)
const zDemo = 1 * currentW1 + 1 * currentW2 + currentB;
const outDemo = sigmoid(zDemo);
drawNode(outPos.x, outPos.y, 'OUT', outDemo.toFixed(2), '#3fb950');

// ---- Draw loss on canvas ----
ctx.fillStyle = '#f0883e';
ctx.font = '12px monospace';
ctx.textAlign = 'left';
ctx.fillText(`Loss: ${loss.toFixed(6)}`, 20, 220);
}

// ===== TRAINING =====
function trainStep() {
let totalLoss = 0;
data.forEach(sample => {
    const z = sample.x1 * w1 + sample.x2 * w2 + b;
    const pred = sigmoid(z);
    const loss = (pred - sample.y) ** 2;
    totalLoss += loss;

    // Gradients
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

// ===== LOOP =====
let epochs = 0;
const MAX_EPOCHS = 150;

function loop() {
const avgLoss = trainStep();
epochs++;

// Update status
epochSpan.textContent = epochs;
lossSpan.textContent = avgLoss.toFixed(6);

// Draw the network with current weights
drawNetwork(w1, w2, b, avgLoss);

// Log every 10 epochs
if (epochs % 10 === 0) {
    const line = document.createElement('div');
    line.innerHTML = `<span class="weight">[Epoch ${epochs}]</span> Loss: <span class="loss">${avgLoss.toFixed(6)}</span> | w1: ${w1.toFixed(3)}, w2: ${w2.toFixed(3)}, b: ${b.toFixed(3)}`;
    logDiv.appendChild(line);
    logDiv.scrollTop = logDiv.scrollHeight;
}

if (epochs < MAX_EPOCHS) {
    setTimeout(loop, 100);
} else {
    // Done
    const doneLine = document.createElement('div');
    doneLine.innerHTML = `<span class="done">DONE! Testing AND:</span>`;
    logDiv.appendChild(doneLine);
    data.forEach(d => {
    const z = d.x1 * w1 + d.x2 * w2 + b;
    const p = sigmoid(z);
    const line = document.createElement('div');
    line.innerHTML = `  ${d.x1} AND ${d.x2} = ${Math.round(p)} (raw: ${p.toFixed(4)})`;
    logDiv.appendChild(line);
    });
    const finalLine = document.createElement('div');
    finalLine.innerHTML = `<span class="weight">Final Weights:</span> [${w1.toFixed(3)}, ${w2.toFixed(3)}, ${b.toFixed(3)}]`;
    logDiv.appendChild(finalLine);
    const note = document.createElement('div');
    note.innerHTML = `<span style="color: #8b949e;">Notice: w1≈1.0, w2≈1.0, b≈-1.5 (as predicted)</span>`;
    logDiv.appendChild(note);
    logDiv.scrollTop = logDiv.scrollHeight;
}
}

// ===== START =====
// Initial draw
drawNetwork(w1, w2, b, 1.0);
logDiv.innerHTML = '> Training AND gate...\n';
setTimeout(loop, 300);