// Dashboard Telemetry and Interaction
document.addEventListener("DOMContentLoaded", () => {
    let fraudSessionCount = 0;
    const presets = {
        genuine: {
            TransactionAmt: 59.00,
            TransactionDT: 86400,
            ProductCD: "W",
            card1: 13926,
            card2: 327,
            card3: 150,
            card4: "discover",
            card5: 142,
            card6: "credit",
            addr1: 315,
            addr2: 87,
            P_emaildomain: "gmail.com",
            R_emaildomain: "gmail.com",
            DeviceType: "desktop",
            DeviceInfo: "Windows"
        },
        fraud: {
            TransactionAmt: 250.00,
            TransactionDT: 90500,
            ProductCD: "C",
            card1: 9500,
            card2: 321,
            card3: 150,
            card4: "visa",
            card5: 226,
            card6: "credit",
            addr1: 299,
            addr2: 87,
            P_emaildomain: "anonymous.com",
            R_emaildomain: "anonymous.com",
            DeviceType: "mobile",
            DeviceInfo: "iOS Device"
        }
    };

    // Initialize UI Elements
    const systemStatusText = document.getElementById("system-status-text");
    const modelBadgeVersion = document.getElementById("model-badge-version");
    const valAccuracy = document.getElementById("val-accuracy");
    const valF1 = document.getElementById("val-f1");
    const valModelName = document.getElementById("val-model-name");
    const valFraudCount = document.getElementById("val-fraud-count");
    const hyperparamsContainer = document.getElementById("hyperparams-container");
    const presetSelect = document.getElementById("preset-select");
    const singlePredictForm = document.getElementById("single-predict-form");
    const predictBtn = document.getElementById("predict-btn");
    
    // Result box elements
    const predictionResultBox = document.getElementById("prediction-result-box");
    const resIconContainer = document.getElementById("res-icon-container");
    const resIcon = document.getElementById("res-icon");
    const resLabel = document.getElementById("res-label");
    const resProbabilityValue = document.getElementById("res-probability-value");
    const resVerdictText = document.getElementById("res-verdict-text");

    // CSV elements
    const dropZone = document.getElementById("drop-zone");
    const csvFileInput = document.getElementById("csv-file-input");
    const batchResultsCard = document.getElementById("batch-results-card");
    const batchStats = document.getElementById("batch-stats");
    const batchTableBody = document.getElementById("batch-table-body");
    const clearBatchBtn = document.getElementById("clear-batch-btn");

    // 1. Fetch Health Status
    const checkHealth = async () => {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (data.status === "healthy") {
                systemStatusText.innerText = "PIPELINE RUNNING";
                systemStatusText.parentElement.querySelector(".status-indicator").className = "status-indicator online";
                modelBadgeVersion.innerText = `MODEL: ${data.model_version.toUpperCase()}`;
            } else {
                systemStatusText.innerText = "DEGRADED STATE";
                systemStatusText.parentElement.querySelector(".status-indicator").className = "status-indicator offline";
            }
        } catch (e) {
            systemStatusText.innerText = "DISCONNECTED";
            systemStatusText.parentElement.querySelector(".status-indicator").className = "status-indicator offline";
        }
    };
    checkHealth();
    setInterval(checkHealth, 10000);

    // 2. Load Telemetry Data & Draw comparison charts
    const loadTelemetry = async () => {
        try {
            const res = await fetch("/api/telemetry");
            if (!res.ok) throw new Error("No telemetry file generated yet.");
            
            const data = await res.json();
            
            // Set KPIs
            valAccuracy.innerText = `${(data.metrics.accuracy * 100).toFixed(2)}%`;
            valF1.innerText = `${(data.best_f1_score * 100).toFixed(2)}%`;
            valModelName.innerText = `Champion: ${data.best_model_name}`;
            
            // Populate hyperparams
            const bestModelParams = data.all_models[data.best_model_name].params;
            hyperparamsContainer.innerHTML = "";
            Object.entries(bestModelParams).forEach(([param, val]) => {
                const div = document.createElement("div");
                div.className = "param-item";
                div.innerHTML = `<span>${param}</span><span class="param-val">${val}</span>`;
                hyperparamsContainer.appendChild(div);
            });

            // Draw Chart
            const ctx = document.getElementById("modelComparisonChart").getContext("2d");
            const models = Object.keys(data.all_models);
            const accuracyData = models.map(m => data.all_models[m].metrics.accuracy);
            const precisionData = models.map(m => data.all_models[m].metrics.precision);
            const recallData = models.map(m => data.all_models[m].metrics.recall);
            const f1Data = models.map(m => data.all_models[m].metrics.f1_score);

            new Chart(ctx, {
                type: "bar",
                data: {
                    labels: models,
                    datasets: [
                        { label: "Accuracy", data: accuracyData, backgroundColor: "rgba(94, 234, 212, 0.4)", borderColor: "#5eead4", borderWidth: 1 },
                        { label: "Precision", data: precisionData, backgroundColor: "rgba(56, 189, 248, 0.4)", borderColor: "#38bdf8", borderWidth: 1 },
                        { label: "Recall", data: recallData, backgroundColor: "rgba(245, 158, 11, 0.4)", borderColor: "#f59e0b", borderWidth: 1 },
                        { label: "F1-Score", data: f1Data, backgroundColor: "rgba(244, 63, 94, 0.5)", borderColor: "#f43f5e", borderWidth: 1.5 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            grid: { color: "rgba(255,255,255,0.05)" },
                            ticks: { color: "#94a3b8" }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: "#94a3b8" }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: "#cbd5e1" } }
                    }
                }
            });
        } catch (e) {
            console.log("Telemetry details not available. Waiting for training run...", e);
            // Default placeholder chart
            const ctx = document.getElementById("modelComparisonChart").getContext("2d");
            new Chart(ctx, {
                type: "bar",
                data: {
                    labels: ["Logistic Regression", "Random Forest", "XGBoost"],
                    datasets: [{ label: "F1-Score Placeholder", data: [0.75, 0.88, 0.91], backgroundColor: "rgba(56, 189, 248, 0.2)", borderColor: "#38bdf8", borderWidth: 1 }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: "#94a3b8" } } }
                }
            });
        }
    };
    loadTelemetry();

    // 3. Preset selector
    presetSelect.addEventListener("change", (e) => {
        const selectedPreset = presets[e.target.value];
        if (selectedPreset) {
            Object.keys(selectedPreset).forEach(key => {
                const el = document.getElementById(key);
                if (el) el.value = selectedPreset[key];
            });
        }
    });

    // 4. Single inference form submit
    singlePredictForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> RUNNING SECURITY CHECK...';
        
        const formData = new FormData(singlePredictForm);
        const payload = {};
        
        const categoricalFields = [
            "ProductCD", "card4", "card6", "P_emaildomain", 
            "R_emaildomain", "DeviceType", "DeviceInfo"
        ];
        
        // Parse numbers to floats, keep categories as strings
        for (const [key, value] of formData.entries()) {
            if (categoricalFields.includes(key)) {
                payload[key] = value;
            } else {
                payload[key] = parseFloat(value);
            }
        }

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Prediction request failed.");
            }
            
            const data = await res.json();
            
            // Show result block
            predictionResultBox.classList.remove("hidden");
            
            if (data.is_fraud === 1) {
                predictionResultBox.className = "result-box fraud-alert";
                resIcon.className = "fa-solid fa-triangle-exclamation";
                resLabel.innerText = "FRAUDULENT TRANSACTION FLAGGED";
                resVerdictText.innerText = "CRITICAL: The transaction shows high match with known fraudulent behavior. Action required.";
                fraudSessionCount++;
                valFraudCount.innerText = fraudSessionCount;
                valFraudCount.classList.add("text-red");
            } else {
                predictionResultBox.className = "result-box genuine-alert";
                resIcon.className = "fa-solid fa-circle-check";
                resLabel.innerText = "GENUINE TRANSACTION VERIFIED";
                resVerdictText.innerText = "SUCCESS: No fraudulent signatures were detected. The transaction is cleared.";
            }
            
            resProbabilityValue.innerText = `${(data.probability * 100).toFixed(4)}%`;
            
        } catch (err) {
            alert(`Prediction Error: ${err.message}`);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> RUN TRANSACTION CHECK';
        }
    });

    // 5. CSV Drag and Drop
    dropZone.addEventListener("click", () => csvFileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.endsWith(".csv")) {
            processCSV(files[0]);
        }
    });

    csvFileInput.addEventListener("change", (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            processCSV(files[0]);
        }
    });

    const processCSV = async (file) => {
        dropZone.innerHTML = `<i class="fa-solid fa-spinner fa-spin upload-icon"></i><p>Uploading and analyzing ${file.name}...</p>`;
        
        const fd = new FormData();
        fd.append("file", file);
        
        try {
            const res = await fetch("/predict_csv", {
                method: "POST",
                body: fd
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Batch prediction request failed.");
            }
            
            const data = await res.json();
            
            // Show stats and table
            batchResultsCard.classList.remove("hidden");
            batchStats.innerText = `Processed ${data.total_processed} records (${data.fraud_detected} fraud flagged)`;
            
            // Populate table
            batchTableBody.innerHTML = "";
            data.predictions.forEach(p => {
                const row = document.createElement("tr");
                const badgeClass = p.is_fraud === 1 ? "badge danger" : "badge safe";
                
                row.innerHTML = `
                    <td>Row ${p.row_index + 1}</td>
                    <td>${(p.probability * 100).toFixed(4)}%</td>
                    <td><span class="${badgeClass}">${p.label}</span></td>
                `;
                batchTableBody.appendChild(row);
                
                if (p.is_fraud === 1) {
                    fraudSessionCount++;
                }
            });
            
            valFraudCount.innerText = fraudSessionCount;
            if (fraudSessionCount > 0) {
                valFraudCount.classList.add("text-red");
            }
            
        } catch (err) {
            alert(`Batch Prediction Error: ${err.message}`);
        } finally {
            // Restore drop zone
            dropZone.innerHTML = `
                <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
                <p>Drag and drop a transaction CSV sample here or click to browse</p>
            `;
        }
    };

    clearBatchBtn.addEventListener("click", () => {
        batchResultsCard.classList.add("hidden");
        batchTableBody.innerHTML = "";
    });
});
