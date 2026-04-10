import React, { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [theme, setTheme] = useState('dark');
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    energy: 150.0,
    shiftX: 0.0,
    shiftY: 0.0
  });
  
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Initialize theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(t => t === 'dark' ? 'light' : 'dark');
  };

  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          energy_mev: parseFloat(params.energy),
          shift_x_cm: parseFloat(params.shiftX),
          shift_y_cm: parseFloat(params.shiftY),
          seed: Math.floor(Math.random() * 10000)
        })
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError("Falha na simulação. Verifica se o backend Python (FastAPI) está a correr.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <button className="theme-toggle" onClick={toggleTheme}>
        {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
      </button>

      {/* Sidebar Controls */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <h1>Proton AI</h1>
          <p>2D Monte Carlo / U-Net Inference Engine</p>
        </div>

        <div className="control-group">
          <div className="control-item">
            <label>Beam Energy <span>{params.energy} MeV</span></label>
            <input 
              type="range" min="80" max="220" step="1" 
              value={params.energy}
              onChange={(e) => setParams({...params, energy: e.target.value})}
            />
          </div>

          <div className="control-item">
            <label>Depth Shift (Z) <span>{params.shiftX} cm</span></label>
            <input 
              type="range" min="-1.0" max="1.0" step="0.1" 
              value={params.shiftX}
              onChange={(e) => setParams({...params, shiftX: e.target.value})}
            />
          </div>

          <div className="control-item">
            <label>Lateral Shift (Y) <span>{params.shiftY} cm</span></label>
            <input 
              type="range" min="-1.0" max="1.0" step="0.1" 
              value={params.shiftY}
              onChange={(e) => setParams({...params, shiftY: e.target.value})}
            />
          </div>
        </div>

        <button 
          className="btn-primary" 
          onClick={handleSimulate} 
          disabled={loading}
        >
          {loading ? 'Simulating...' : 'Run Clinical Pipeline'}
        </button>

        {error && <div style={{color: '#ef4444', fontSize: '0.85rem', marginTop: '10px'}}>{error}</div>}
      </aside>

      {/* Main Results Area */}
      <main className="main-content">
        
        {/* Metrics Panel */}
        <section className="metrics-container glass-panel">
          <div className="metric-card">
            <h3>Range Error (Bragg Peak)</h3>
            <div className="value">
              {results ? `${results.metrics.range_error_mm.toFixed(2)} mm` : '--'}
            </div>
          </div>
          <div className="metric-card">
            <h3>Mean Penumbra Error</h3>
            <div className="value">
              {results ? `${results.metrics.penumbra_error_mm.toFixed(2)} mm` : '--'}
            </div>
          </div>
          <div className="metric-card">
            <h3>Mean Squared Error</h3>
            <div className="value">
              {results ? results.metrics.mse.toFixed(4) : '--'}
            </div>
          </div>
        </section>

        {/* Visualizations Panel */}
        <section className="glass-panel" style={{flex: 1, display: 'flex', flexDirection: 'column'}}>
          {loading ? (
            <div className="loading-overlay">
              <p>Orchestrating Monte Carlo Engine & AI Model...</p>
            </div>
          ) : results ? (
            <>
              {/* Heatmaps */}
              <div className="images-grid">
                <div className="image-card">
                  <img src={results.images.noisy} alt="Noisy Monte Carlo" />
                </div>
                <div className="image-card">
                  <img src={results.images.pred} alt="U-Net Denoised Prediction" />
                </div>
                <div className="image-card">
                  <img src={results.images.ref} alt="Ground Truth Simulation" />
                </div>
              </div>
              
              {/* Axial Profile */}
              <div className="profile-container">
                <img src={results.images.profile} alt="Central Axis Depth-Dose Profile" />
              </div>
              
              {/* CT Phantom Overlay indicator */}
              <div style={{marginTop: '15px', textAlign: 'center'}}>
                <img style={{width: '30%', borderRadius: '8px', opacity: 0.8}} src={results.images.phantom} alt="Anatomy" />
                <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px'}}>Current Anatomical Setup (shifted)</p>
              </div>
            </>
          ) : (
            <div className="loading-overlay" style={{animation: 'none'}}>
              <p>Configure parameters and click "Run Clinical Pipeline" to begin.</p>
            </div>
          )}
        </section>

      </main>
    </div>
  );
}

export default App;
