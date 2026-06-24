const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = 700, H = 300;

// Network Architecture Layout
const layers = {
    input: [{x: 100, y: 100}, {x: 100, y: 200}],
    hidden: [{x: 300, y: 75}, {x: 300, y: 225}],
    output: [{x: 550, y: 150}]
};

function drawNetwork(weights, loss) {
    ctx.clearRect(0, 0, W, H);
    
    // 1. Draw Lines (Weights)
    // Weights array: [w11, w12, w21, w22, wo1, wo2, bh1, bh2, bout]
    const w11 = weights[0], w12 = weights[1], w21 = weights[2], w22 = weights[3];
    const wo1 = weights[4], wo2 = weights[5];
    
    // Helper to draw a weight line
    function drawLine(x1, y1, x2, y2, weight, label) {
        const val = Math.round(weight * 100) / 100;
        const thickness = Math.min(5, Math.abs(weight) * 4 + 1);
        const hue = weight > 0 ? 120 : 0; // Green for positive, Red for negative
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${Math.min(1, Math.abs(weight) + 0.2)})`;
        ctx.lineWidth = thickness;
        ctx.stroke();
        
        // Draw weight value in the middle
        const midX = (x1 + x2)/2 - 15;
        const midY = (y1 + y2)/2 - 10;
        ctx.fillStyle = '#fff';
        ctx.font = '11px monospace';
        ctx.fillText(val, midX, midY);
    }

    // Input -> Hidden
    drawLine(100, 100, 300, 75, w11, 'w11');
    drawLine(100, 100, 300, 225, w12, 'w12');
    drawLine(100, 200, 300, 75, w21, 'w21');
    drawLine(100, 200, 300, 225, w22, 'w22');
    
    // Hidden -> Output
    drawLine(300, 75, 550, 150, wo1, 'wo1');
    drawLine(300, 225, 550, 150, wo2, 'wo2');

    // 2. Draw Nodes (Neurons)
    function drawNode(x, y, label, activation = 0) {
        // Glow effect
        const grad = ctx.createRadialGradient(x, y, 0, x, y, 30);
        grad.addColorStop(0, `rgba(88, 166, 255, ${0.1 + activation * 0.4})`);
        grad.addColorStop(1, 'rgba(88,166,255,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, 30, 0, Math.PI * 2);
        ctx.fill();

        // Main circle
        ctx.fillStyle = '#1f2937';
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 22, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Activation text inside
        ctx.fillStyle = '#f0f6fc';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y - 3);
        ctx.fillStyle = '#8b949e';
        ctx.font = '9px monospace';
        ctx.fillText((activation).toFixed(2), x, y + 15);
    }

    // Inputs (we pass dummy activations for drawing, will be updated in loop)
    drawNode(100, 100, 'X1', 0);
    drawNode(100, 200, 'X2', 0);
    drawNode(300, 75, 'H1', 0);
    drawNode(300, 225, 'H2', 0);
    drawNode(550, 150, 'OUT', 0);

    // Draw Loss on canvas
    ctx.fillStyle = '#f0883e';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${loss.toFixed(6)}`, 20, 280);
}

// Softplus and its derivative
function softplus(x) { 
    // Prevents Math.exp from exploding for very large numbers
    if (x > 100) return x; 
    return Math.log(1 + Math.exp(x)); 
}
function softplusDerivative(x) { 
    // The derivative of ln(1+e^x) is the Sigmoid function. 1 / (1 + e^-x)
    return 1 / (1 + Math.exp(-x)); 
}

// Sigmoid and its derivative
function sigmoid(x) {
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    } else {
        const z = Math.exp(x);
        return z / (1 + z);
    }
}
function sigmoidDerivative(x) { return x * (1 - x); }

// Tanh and its derivative
function tanh(x) {
    return Math.tanh(x);
}
function tanhDerivative(tanhOutput) {
    return 1 - tanhOutput * tanhOutput;
}

// The Neural Network class
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
        
        this.lr = 0.1; // Learning Rate
    }

    // Forward pass: returns {hidden1, hidden2, output}
    forward(x1, x2) {
        // Hidden layer
        const z1 = x1 * this.w11 + x2 * this.w12 + this.bh1;
        const z2 = x1 * this.w21 + x2 * this.w22 + this.bh2;
        const h1 = tanh(z1);
        const h2 = tanh(z2);
        
        // Output layer
        const z_out = h1 * this.wo1 + h2 * this.wo2 + this.bout;
        const out = tanh(z_out); // Output should be between 0 and 1
        
        return { z1, z2, h1, h2, out };
    }

    // Train on a single sample (x1, x2, target)
    train(x1, x2, target) {
        // --- Forward Pass ---
        const {z1, z2, h1, h2, out} = this.forward(x1, x2);
        
        // --- Calculate Loss ---
        const loss = (out - target) ** 2;
        
        // --- Backward Pass (The "Blame Game") ---
        
        // 1. Output layer gradients
        const dLoss_dOut = 2 * (out - target);
        const dOut_dZout = tanhDerivative(out);
        const dLoss_dZout = dLoss_dOut * dOut_dZout;
        
        // Gradients for output weights
        const dW_o1 = dLoss_dZout * h1;
        const dW_o2 = dLoss_dZout * h2;
        const dB_out = dLoss_dZout * 1;
        
        // 2. Hidden layer gradients (Backpropagating through output)
        const dLoss_dH1 = dLoss_dZout * this.wo1;
        const dLoss_dH2 = dLoss_dZout * this.wo2;
        const dH1_dZ1 = tanhDerivative(h1);
        const dH2_dZ2 = tanhDerivative(h2);
        
        // Gradients for hidden weights (Layer 1)
        const dW_11 = dLoss_dH1 * dH1_dZ1 * x1;
        const dW_12 = dLoss_dH1 * dH1_dZ1 * x2;
        const dW_21 = dLoss_dH2 * dH2_dZ2 * x1;
        const dW_22 = dLoss_dH2 * dH2_dZ2 * x2;
        const dB_h1 = dLoss_dH1 * dH1_dZ1 * 1;
        const dB_h2 = dLoss_dH2 * dH2_dZ2 * 1;
        
        // --- Update Weights (Gradient Descent) ---
        this.w11 -= this.lr * dW_11;
        this.w12 -= this.lr * dW_12;
        this.w21 -= this.lr * dW_21;
        this.w22 -= this.lr * dW_22;
        this.wo1 -= this.lr * dW_o1;
        this.wo2 -= this.lr * dW_o2;
        this.bh1 -= this.lr * dB_h1;
        this.bh2 -= this.lr * dB_h2;
        this.bout -= this.lr * dB_out;

        [
            this.w11,
            this.w12,
            this.w21,
            this.w22,
            this.wo1,
            this.wo2
        ].forEach(w => {
            if (!Number.isFinite(w)){
                console.log("BAD WEIGHT", w);
                if(!isNaN(w)){
                    console.log("dLoss_dZout", dLoss_dZout);
                    console.log("dW_o1", dW_o1);
                    console.log("dW_o2", dW_o2);
                    console.log("dB_out", dB_out);
                    console.log("dW_11", dW_11);
                    console.log("dW_12", dW_12);
                    console.log("dW_21", dW_21);
                    console.log("dW_22", dW_22);
                    console.log("dB_h1", dB_h1);
                    console.log("dB_h2", dB_h2);
                    console.log("dLoss_dH1", dLoss_dH1);
                    console.log("dH1_dZ1", dH1_dZ1);
                    console.log("z1", z1);
                    console.log("\n");   
                }
            }
        });
        
        // Return details for logging/visualization
        return {
            weights: [this.w11, this.w12, this.w21, this.w22, this.wo1, this.wo2, this.bh1, this.bh2, this.bout],
            h1, h2, out, loss
        };
    }
}

let net = new XORNetwork();
let epoch = 0;
const consoleDiv = document.getElementById('console');

// // The XOR truth table
// const data = [
//     { x1: 0, x2: 0, y: 0 },
//     { x1: 0, x2: 1, y: 1 },
//     { x1: 1, x2: 0, y: 1 },
//     { x1: 1, x2: 1, y: 0 }
// ];

function logToConsole(message, type = '') {
    const line = document.createElement('div');
    line.innerHTML = message;
    consoleDiv.appendChild(line);
    // Keep console manageable
    while (consoleDiv.children.length > 30) {
        consoleDiv.removeChild(consoleDiv.firstChild);
    }
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
}

function step() {
    // Pick a random sample from the XOR table
    const sample = data[Math.floor(Math.random() * data.length)];
    const result = net.train(sample.x1, sample.x2, sample.y);

    epoch++;
    
    // Update Canvas
    drawNetwork(result.weights, result.loss);
    
    // Update Status Bar
    document.getElementById('epochDisplay').textContent = epoch;
    document.getElementById('lossDisplay').textContent = result.loss.toFixed(6);
    
    // Log to Console every 5 epochs to avoid spam
    if (epoch % 10 === 0) {
        const w = result.weights;
        // Format weights nicely
        const wStr = `w11=${w[0].toFixed(3)} w12=${w[1].toFixed(3)} wo1=${w[4].toFixed(3)}`;
        const predStr = `Pred: ${result.out.toFixed(4)} | Target: ${sample.y}`;
        logToConsole(`<span class="weight">[Epoch ${epoch}]</span> ${predStr} | Loss: <span>${result.loss.toFixed(6)}</span> | ${wStr}`);
    }
    
    // Stop after 300 epochs (enough to learn XOR perfectly)
    if (epoch < 10000) {
        setTimeout(step, 3); 
    } else {
        logToConsole(`<span style="color: #3fb950;">DONE</span><br/>Final Weights: [${result.weights.map(w=>w.toFixed(3)).join(', ')}]`);
        // Test the network
        logToConsole(`<span style="color: #f0883e;">Testing:</span>`);
        
        let passed = true;

        data.forEach(d => {
            const f = net.forward(d.x1, d.x2);
            const prediction = Math.round(f.out);

            logToConsole(`  ${d.x1} XOR ${d.x2} = ${prediction} (raw: ${f.out.toFixed(4)})`);

            if (prediction !== d.y) {
                passed = false;
            }
        });

        if (passed) {
            logToConsole(`<span style="color: #3fb950;"> XOR Neural Network PASSED</span>`)
            count += 1;
        } else {
            logToConsole(`<span style="color: #b93f3f;"> XOR Neural Network FAILED</span>`)
            count += 0;
        }
    }
}
const data = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 0 }
];
function trainAndTest() {
    const net = new XORNetwork();

    for (let epoch = 0; epoch < 30000; epoch++) {
        const sample = data[Math.floor(Math.random() * data.length)];
        net.train(sample.x1, sample.x2, sample.y);
    }

    const results = data.map(d => ({
        ...d,
        out: net.forward(d.x1, d.x2).out
    }));

    const passed = results.every(r =>
        Math.round(r.out) === r.y
    );

    return { passed, results };
}

let passes = 0;

for (let run = 1; run <= 100; run++) {
    const { passed, results } = trainAndTest();

    if (passed) {
        passes++;
        console.log(`Run ${run}: PASS`);
    } else {
        console.log(`Run ${run}: FAIL`);
        results.forEach(r => {
            console.log(
                `${r.x1} XOR ${r.x2} = ${r.out.toFixed(4)} (expected ${r.y})`
            );
        });
    }
}

// console.log(`Passed ${passes}/100 runs`);

//Initial draw
const initialWeights = [net.w11, net.w12, net.w21, net.w22, net.wo1, net.wo2, net.bh1, net.bh2, net.bout];
drawNetwork(initialWeights, 1.0);
logToConsole('Starting real-time XOR training...');

//Start 
setTimeout(step, 2000);