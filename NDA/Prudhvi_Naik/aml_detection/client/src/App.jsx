import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Shield, Activity, FileSearch, AlertTriangle, CheckCircle, ServerCrash, Cpu, Activity as ActivityIcon } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

// API base path
const API_BASE = 'http://localhost:8000';

const DashboardLayout = ({ children }) => {
  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans">
      <aside className="w-64 bg-[#1e293b] border-r border-slate-700 flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-slate-700">
          <Shield className="w-8 h-8 text-blue-500" />
          <span className="font-bold text-xl text-white tracking-tight">AML Engine</span>
        </div>
        <nav className="flex-1 p-4 space-y-2 mt-4">
          <Link to="/" className="flex items-center gap-3 p-3 rounded-lg bg-blue-500/10 text-blue-400 font-medium">
            <Activity className="w-5 h-5" /> Live Scoring
          </Link>
        </nav>
        <div className="p-6 border-t border-slate-700">
           <div className="flex items-center gap-2 text-xs text-slate-400">
             <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
             Multi-Model Ensemble Active
           </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-8">{children}</main>
    </div>
  );
};

const LiveScoring = () => {
  const [transactions, setTransactions] = useState([]);
  const [selectedTxn, setSelectedTxn] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/transactions`)
      .then(res => setTransactions(res.data.transactions))
      .catch(err => setError("Could not connect to API. Is backend running?"));
  }, []);

  const handlePredict = async (txnId) => {
    setSelectedTxn(txnId);
    if (!txnId) return;
    
    setLoading(true);
    setPrediction(null);
    setError(null);
    try {
      const res = await axios.post(`${API_BASE}/predict`, { transaction_id: txnId });
      setPrediction(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "text-red-500";
    if (score >= 40) return "text-amber-500";
    return "text-emerald-500";
  };

  const getBgColor = (score) => {
    if (score >= 80) return "bg-red-500/10 border-red-500/30";
    if (score >= 40) return "bg-amber-500/10 border-amber-500/30";
    return "bg-emerald-500/10 border-emerald-500/30";
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Fraud Detection Engine</h2>
          <p className="text-slate-400 mt-1">Real-time prediction using LightGBM, XGBoost, and CatBoost Ensemble.</p>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-6 flex items-center gap-3">
          <ServerCrash className="w-5 h-5" />
          {error}
        </div>
      )}

      <div className="bg-[#1e293b] border border-slate-700 p-6 rounded-2xl mb-8 shadow-xl">
        <label className="block text-sm font-medium text-slate-300 mb-2">Select a Transaction to Analyze</label>
        <select 
          className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-lg p-3 outline-none focus:border-blue-500 transition-colors"
          value={selectedTxn}
          onChange={(e) => handlePredict(e.target.value)}
        >
          <option value="">-- Choose Transaction --</option>
          {transactions.map(t => (
            <option key={t.transaction_id} value={t.transaction_id}>
              {t.transaction_id} (Ground Truth: {t.actual_is_aml ? `Fraud - ${t.actual_typology}` : 'Legitimate'})
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      )}

      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Risk Score Card */}
          <div className={`col-span-1 lg:col-span-1 rounded-2xl p-8 border backdrop-blur-sm flex flex-col items-center justify-center shadow-xl ${getBgColor(prediction.fraud_risk_score)}`}>
            {prediction.fraud_risk_score >= 80 ? <AlertTriangle className="w-16 h-16 text-red-500 mb-4" /> : <CheckCircle className="w-16 h-16 text-emerald-500 mb-4" />}
            <h3 className="text-slate-300 text-lg font-medium mb-2">Fraud Risk Score</h3>
            <div className={`text-6xl font-black tracking-tighter ${getScoreColor(prediction.fraud_risk_score)}`}>
              {prediction.fraud_risk_score.toFixed(1)}<span className="text-3xl">%</span>
            </div>
            <div className={`mt-4 px-4 py-1 rounded-full text-sm font-bold uppercase tracking-wider ${prediction.fraud_risk_score >= 80 ? 'bg-red-500 text-white' : prediction.fraud_risk_score >= 40 ? 'bg-amber-500 text-slate-900' : 'bg-emerald-500 text-white'}`}>
              {prediction.risk_category}
            </div>
            {prediction.actual_is_aml === 1 && (
              <p className="mt-6 text-sm text-slate-400">Ground Truth: Actual Fraud</p>
            )}
          </div>

          {/* Typology Chart */}
          <div className="col-span-1 lg:col-span-2 bg-[#1e293b] border border-slate-700 p-6 rounded-2xl shadow-xl flex flex-col">
            <h3 className="text-lg font-bold text-white mb-1">Typology Classification</h3>
            <p className="text-slate-400 text-sm mb-6">Probability distribution across money laundering typologies</p>
            <div className="flex-1 min-h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={prediction.typology_probabilities} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
                  <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
                  <YAxis dataKey="typology" type="category" width={150} stroke="#94a3b8" fontSize={12} />
                  <Tooltip cursor={{fill: '#334155'}} contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff'}} />
                  <Bar dataKey="probability" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                    {prediction.typology_probabilities.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.probability > 50 ? '#ef4444' : '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* SHAP Explainability */}
          <div className="col-span-1 lg:col-span-3 bg-[#1e293b] border border-slate-700 p-6 rounded-2xl shadow-xl">
             <div className="flex items-center gap-3 mb-6">
                <Cpu className="w-6 h-6 text-indigo-400" />
                <div>
                  <h3 className="text-lg font-bold text-white">SHAP Explainability (Why?)</h3>
                  <p className="text-slate-400 text-sm">Top features mathematically driving the risk score.</p>
                </div>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
               {prediction.key_risk_drivers.map((driver, idx) => (
                 <div key={idx} className="bg-[#0f172a] border border-slate-700 p-4 rounded-xl relative overflow-hidden">
                   <div className={`absolute top-0 left-0 w-full h-1 ${driver.impact > 0 ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
                   <p className="text-xs text-slate-400 font-mono mb-1 truncate" title={driver.feature}>{driver.feature}</p>
                   <p className="text-xl font-bold text-white mb-2">{driver.actual_value}</p>
                   <div className="flex items-center justify-between text-xs font-semibold">
                      <span className={driver.impact > 0 ? 'text-red-400' : 'text-emerald-400'}>
                        {driver.impact > 0 ? '+' : ''}{driver.impact.toFixed(2)} impact
                      </span>
                   </div>
                 </div>
               ))}
             </div>
          </div>
        </div>
      )}
    </div>
  );
};

const App = () => {
  return (
    <Router>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<LiveScoring />} />
        </Routes>
      </DashboardLayout>
    </Router>
  );
};

export default App;
