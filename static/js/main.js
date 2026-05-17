// Initialize Map
const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

let marker = L.marker([20, 0]).addTo(map);

// Input elements for coordinates
const latInput = document.getElementById('Latitude');
const lonInput = document.getElementById('Longitude');
const submitBtn = document.getElementById('submit-btn');

// Try to get user's current location
if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition((position) => {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        const userPos = [lat, lng];
        
        map.setView(userPos, 10);
        marker.setLatLng(userPos);
        
        if (latInput && lonInput) {
            latInput.value = lat.toFixed(6);
            lonInput.value = lng.toFixed(6);
            promptButton();
        }
    }, (error) => {
        console.log("Geolocation error or denied:", error.message);
    });
}

function promptButton() {
    submitBtn.classList.add('pulse-btn');
    submitBtn.innerText = 'Fetch Satellite Data & Assess Risk';
    const mapHint = document.querySelector('.map-hint');
    if (mapHint) {
        mapHint.innerText = "Location locked! Scroll down and click the glowing button to assess risk.";
        mapHint.style.color = 'var(--primary)';
        mapHint.style.fontWeight = 'bold';
    }
}

// Update map marker when inputs change
function updateMap() {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    if (!isNaN(lat) && !isNaN(lon)) {
        const newPos = [lat, lon];
        marker.setLatLng(newPos);
        map.setView(newPos, 13);
    }
}

latInput?.addEventListener('input', updateMap);
lonInput?.addEventListener('input', updateMap);

// Click on map to set inputs
map.on('click', (e) => {
    const { lat, lng } = e.latlng;
    if (latInput && lonInput) {
        latInput.value = lat.toFixed(6);
        lonInput.value = lng.toFixed(6);
        marker.setLatLng([lat, lng]);
        map.setView([lat, lng], 13);
        
        promptButton();
    }
});

async function fetchEnvironmentalData(lat, lng) {
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const transparencyNote = document.querySelector('.transparency-note');

    submitBtn.innerText = 'Connecting to Satellites...';
    
    try {
        const response = await fetch('/get_features', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ Latitude: lat, Longitude: lng })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const { features, api_log, location_context } = data;
            
            // Fill inputs
            for (const [key, value] of Object.entries(features)) {
                const input = document.getElementById(key);
                if (input) {
                    input.value = typeof value === 'number' ? value.toFixed(2) : value;
                    input.style.borderColor = 'var(--primary)';
                    setTimeout(() => input.style.borderColor = '', 1500);
                }
            }
            
            // Update Transparency Report with LIVE logs
            if (transparencyNote) {
                transparencyNote.innerHTML = `
                    <h3>Live AI Intelligence Report</h3>
                    <p><strong>📍 Coordinates:</strong> ${lat.toFixed(4)}, ${lng.toFixed(4)}</p>
                    <p><strong>📏 Elevation:</strong> ${api_log.elevation}</p>
                    <p><strong>🌧️ Rainfall:</strong> ${api_log.rainfall}</p>
                    <p><strong>🗺️ Terrain Mapping:</strong> ${location_context}</p>
                    <p class="map-hint" style="margin-top:10px; color: var(--primary)">* Data successfully fetched and locked.</p>
                `;
            }
            return true;
        }
    } catch (error) {
        console.error('Error fetching features:', error);
        return false;
    }
    return false;
}

document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    
    if (isNaN(lat) || isNaN(lon)) {
        alert("Please select a location on the map first.");
        return;
    }
    
    submitBtn.classList.remove('pulse-btn');
    submitBtn.disabled = true;
    
    // First fetch environmental data
    const fetchSuccess = await fetchEnvironmentalData(lat, lon);
    
    if (!fetchSuccess) {
        alert("Failed to fetch environmental data. Cannot proceed.");
        submitBtn.disabled = false;
        submitBtn.innerText = 'Try Again';
        return;
    }

    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const probText = document.getElementById('prob-text');
    const probBar = document.getElementById('prob-bar');
    
    // UI Loading State
    submitBtn.innerText = 'Running AI Prediction...';
    
    // Collect form data (which now has all environmental inputs filled)
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Show result
            resultContainer.style.display = 'block';
            resultText.innerText = result.result_text;
            
            // Update map to the prediction location
            updateMap();

            // Color coding and animations based on result
            probBar.classList.remove('danger-active', 'safe-active');
            if (result.prediction === 1) {
                resultText.style.color = 'var(--danger)';
                probBar.style.background = 'var(--danger)';
                probBar.classList.add('danger-active');
            } else {
                resultText.style.color = 'var(--success)';
                probBar.style.background = 'var(--success)';
                probBar.classList.add('safe-active');
            }
            
            // Update probability
            const probPercent = (result.probability * 100).toFixed(2);
            probText.innerText = `Risk Level: ${probPercent}%`;
            
            // Trigger bar animation
            setTimeout(() => {
                probBar.style.width = `${probPercent}%`;
            }, 50);

            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction.');
    } finally {
        submitBtn.innerText = 'Analysis Complete';
        setTimeout(() => {
            submitBtn.disabled = false;
            submitBtn.innerText = 'Run Another Assessment';
        }, 2000);
    }
});
