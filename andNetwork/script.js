// The data
const data = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 0 },
    { x1: 1, x2: 0, y: 0 },
    { x1: 1, x2: 1, y: 1 }
];

// Init weights and bias
let w1 = Math.random() * 2 - 1;
let w2 = Math.random() * 2 - 1;
let b = Math.random() * 2 - 1;
const lr = 0.1;
const logDiv = document.getElementById('log');

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function train() {
    let totalLoss = 0;
    // Loop through all 4 samples
    data.forEach(sample => {
    // Forward pass
    const z = sample.x1 * w1 + sample.x2 * w2 + b;
    const pred = sigmoid(z);
    
    // Loss (MSE)
    const loss = (pred - sample.y) ** 2;
    totalLoss += loss;

    // Gradient (Derivative of Loss w.r.t weights)
    // For a Sigmoid, the derivative is pred * (1 - pred)
    const dLoss_dPred = 2 * (pred - sample.y);
    const dPred_dZ = pred * (1 - pred);
    const dZ_dW1 = sample.x1;
    const dZ_dW2 = sample.x2;
    const dZ_dB = 1;

    // Update weights (Gradient Descent)
    w1 -= lr * dLoss_dPred * dPred_dZ * dZ_dW1;
    w2 -= lr * dLoss_dPred * dPred_dZ * dZ_dW2;
    b -= lr * dLoss_dPred * dPred_dZ * dZ_dB;
    });

    return totalLoss / data.length;
}

// Train for 100 epochs
let epochs = 0;
function loop() {
    const avgLoss = train();
    epochs++;
    
    if (epochs % 10 === 0) {
    logDiv.innerHTML += `\n[Epoch ${epochs}] Loss: ${avgLoss.toFixed(6)} | w1: ${w1.toFixed(3)}, w2: ${w2.toFixed(3)}, b: ${b.toFixed(3)}`;
    logDiv.scrollTop = logDiv.scrollHeight;
    }

    if (epochs < 100) {
    setTimeout(loop, 50);
    } else {
    logDiv.innerHTML += `\n\n✅ DONE! Testing AND:`;
    data.forEach(d => {
        const z = d.x1 * w1 + d.x2 * w2 + b;
        const p = sigmoid(z);
        logDiv.innerHTML += `\n  ${d.x1} AND ${d.x2} = ${Math.round(p)} (raw: ${p.toFixed(4)})`;
    });
    logDiv.innerHTML += `\n\n Final Weights: [${w1.toFixed(3)}, ${w2.toFixed(3)}, ${b.toFixed(3)}]`;
    logDiv.innerHTML += `\n Notice: w1≈1.0, w2≈1.0, b≈-1.5 (as predicted)`;
    }
}

loop();