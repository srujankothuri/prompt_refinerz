// DOM elements
const promptForm = document.getElementById('promptForm');
const processingIndicator = document.getElementById('processing-indicator');
const resultsContainer = document.getElementById('results-container');
const bestPromptText = document.getElementById('best-prompt-text');
const bestScore = document.getElementById('best-score');
const bestStrengths = document.getElementById('best-strengths');
const bestWeaknesses = document.getElementById('best-weaknesses');
const copyBestPromptBtn = document.getElementById('copy-best-prompt');
const variationsContainer = document.getElementById('variations-container');

// Charts
let scoreChart = null;
let providerComparisonChart = null;

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    promptForm.addEventListener('submit', handleFormSubmit);
    copyBestPromptBtn.addEventListener('click', copyBestPrompt);
});

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Show processing indicator, hide results
    processingIndicator.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    
    // Get form data
    const originalPrompt = document.getElementById('original-prompt').value;
    const context = document.getElementById('context').value;
    const outputType = document.getElementById('output-type').value;
    const numVariations = parseInt(document.getElementById('num-variations').value);
    
    // Get selected providers
    const providerCheckboxes = document.querySelectorAll('input[name="providers"]:checked');
    const providers = Array.from(providerCheckboxes).map(cb => cb.value);
    
    // Validate inputs
    if (!originalPrompt.trim()) {
        showToast('Please enter a prompt');
        processingIndicator.classList.add('hidden');
        return;
    }
    
    if (providers.length === 0) {
        showToast('Please select at least one LLM provider');
        processingIndicator.classList.add('hidden');
        return;
    }
    
    // Prepare request data
    const requestData = {
        original_prompt: originalPrompt,
        context: context || null,
        desired_output_type: outputType,
        providers: providers,
        num_variations: numVariations
    };
    
    try {
        // Send API request
        const response = await fetch('/api/refine', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Hide processing indicator, show results
        processingIndicator.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error processing prompt:', error);
        showToast(`Error: ${error.message || 'Failed to process prompt'}`);
        processingIndicator.classList.add('hidden');
    }
}

// Display results
function displayResults(data) {
    // Display best prompt
    bestPromptText.textContent = data.best_prompt.prompt_text;
    bestScore.textContent = data.best_prompt.quality_score.toFixed(1);
    
    // Display strengths and weaknesses for best prompt
    bestStrengths.innerHTML = '';
    data.best_prompt.strengths.forEach(strength => {
        const li = document.createElement('li');
        li.textContent = strength;
        bestStrengths.appendChild(li);
    });
    
    bestWeaknesses.innerHTML = '';
    data.best_prompt.weaknesses.forEach(weakness => {
        const li = document.createElement('li');
        li.textContent = weakness;
        bestWeaknesses.appendChild(li);
    });

    // Display all variations
    variationsContainer.innerHTML = '';
    data.variations.forEach((variation, index) => {
        const card = document.createElement('div');
        card.className = `variation-card bg-gray-50 rounded-lg p-4 shadow-sm ${variation.provider}`; // Add provider class
        card.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <h3 class="text-lg font-medium text-gray-700">Variation ${index + 1} (${variation.provider})</h3>
                <span class="bg-indigo-100 text-indigo-800 text-sm font-medium px-2.5 py-0.5 rounded">
                    Score: ${variation.quality_score.toFixed(1)}
                </span>
            </div>
            <p class="text-gray-600 mb-2">${variation.prompt_text}</p>
            <div class="grid grid-cols-2 gap-2 text-sm">
                <div>
                    <h4 class="font-medium text-gray-700">Strengths</h4>
                    <ul class="list-disc pl-4 text-gray-600">
                        ${variation.strengths.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <h4 class="font-medium text-gray-700">Weaknesses</h4>
                    <ul class="list-disc pl-4 text-gray-600">
                        ${variation.weaknesses.map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
        variationsContainer.appendChild(card);
    });

    // Destroy existing charts if they exist
    if (scoreChart) scoreChart.destroy();
    if (providerComparisonChart) providerComparisonChart.destroy();

    // Render Score Chart (Bar chart of all variation scores)
    const scoreCtx = document.getElementById('scoreChart').getContext('2d');
    scoreChart = new Chart(scoreCtx, {
        type: 'bar',
        data: {
            labels: data.variations.map((_, i) => `Variation ${i + 1}`),
            datasets: [{
                label: 'Quality Score',
                data: data.variations.map(v => v.quality_score),
                backgroundColor: 'rgba(99, 102, 241, 0.6)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Score' }
                },
                x: { title: { display: true, text: 'Variations' } }
            },
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Prompt Variation Scores' }
            }
        }
    });

    // Render Provider Comparison Chart (Pie chart of average scores per provider)
    const providerScores = {};
    data.variations.forEach(v => {
        providerScores[v.provider] = providerScores[v.provider] || { total: 0, count: 0 };
        providerScores[v.provider].total += v.quality_score;
        providerScores[v.provider].count += 1;
    });
    const providerAvgScores = Object.keys(providerScores).map(provider => ({
        provider,
        avgScore: providerScores[provider].total / providerScores[provider].count
    }));

    const providerCtx = document.getElementById('providerComparisonChart').getContext('2d');
    providerComparisonChart = new Chart(providerCtx, {
        type: 'pie',
        data: {
            labels: providerAvgScores.map(p => p.provider === 'anthropic' ? 'Claude' : p.provider.charAt(0).toUpperCase() + p.provider.slice(1)), // Display "Claude" instead of "anthropic"
            datasets: [{
                data: providerAvgScores.map(p => p.avgScore),
                backgroundColor: [
                    'rgba(99, 102, 241, 0.8)',  // Claude
                    'rgba(167, 139, 250, 0.8)', // Together
                    'rgba(139, 92, 246, 0.8)'   // Gemini
                ],
                borderColor: '#fff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Average Score by Provider' }
            }
        }
    });
}

// Copy best prompt to clipboard
function copyBestPrompt() {
    const text = bestPromptText.textContent;
    navigator.clipboard.writeText(text)
        .then(() => showToast('Best prompt copied to clipboard!'))
        .catch(err => showToast('Failed to copy prompt: ' + err));
}

// Show toast notification
function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}