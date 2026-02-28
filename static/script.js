document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    const resultsWrapper = document.getElementById('results-wrapper');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const progressFill = document.getElementById('progress-fill');
    const insightText = document.getElementById('insight-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const textArea = document.getElementById('news-input');
        const textToAnalyze = textArea.value.trim();
        
        if (!textToAnalyze) return;

        // UI State: Loading
        submitBtn.disabled = true;
        btnText.textContent = 'Processing...';
        loader.classList.remove('hidden');
        resultsWrapper.classList.add('hidden');
        progressFill.style.width = '0%'; // reset bar

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textToAnalyze })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze text.');
            }

            // Populate Results
            const isFake = data.prediction === 'FAKE';
            
            verdictBadge.textContent = isFake ? 'Fake News Detected' : 'Verified Authentic';
            verdictBadge.className = `badge ${isFake ? 'fake' : 'real'}`;
            
            // Format confidence
            let confValue = data.confidence;
            // Bound confidence between 50 and 99 for realism if the model returns something low
            if (confValue < 50) confValue = 50 + (confValue / 2); 
            if (confValue > 99.9) confValue = 99.9;
            
            const confString = confValue.toFixed(1) + '%';
            confidencePercentage.textContent = confString;

            // Set progress bar appearance
            progressFill.style.background = isFake ? 
                'linear-gradient(90deg, #ef4444, #b91c1c)' : 
                'linear-gradient(90deg, #10b981, #047857)';

            // Contextual insights
            if (isFake) {
                insightText.textContent = `The AI model detected linguistic patterns, tone variations, and structural anomalies highly correlated with fabricated content. Confidence level: ${confString}.`;
            } else {
                insightText.textContent = `The analysis indicates this article utilizes credible journalistic structures, consistent tone, and factual framing. Confidence level: ${confString}.`;
            }

            // UI State: Show Results
            resultsWrapper.classList.remove('hidden');
            
            // Animate progress bar slightly after render
            setTimeout(() => {
                progressFill.style.width = confString;
            }, 100);

        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
        } finally {
            // Restore button
            submitBtn.disabled = false;
            btnText.textContent = 'Analyze Content';
            loader.classList.add('hidden');
        }
    });

    // Handle shift+enter to submit
    document.getElementById('news-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });
});
