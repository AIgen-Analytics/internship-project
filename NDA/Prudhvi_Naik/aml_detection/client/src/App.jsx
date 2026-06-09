import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Shield, Activity, Share2, FileSearch, User, LayoutDashboard, Settings, AlertTriangle, CheckCircle, ZoomIn } from 'lucide-react';
import CytoscapeComponent from 'react-cytoscapejs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

// API base path (fallback for local dev)
const API_BASE = 'http://localhost:8000';

const Login = () => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <div className="p-8 bg-surface rounded-xl shadow-2xl text-center border border-slate-700 max-w-md w-full">
        <Shield className="w-16 h-16 text-primary mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-white mb-2">AML Intelligence Platform</h1>
        <button onClick={() => window.location.href = "/__catalyst/auth/login"} className="w-full bg-primary hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg mt-8 transition-colors">Sign In with Catalyst</button>
      </div>
    </div>
  );
};

const DashboardLayout = ({ children, user }) => {
  return (
    <div className="flex h-screen bg-background text-slate-200">
      <aside className="w-64 bg-surface border-r border-slate-800 flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-slate-800">
          <Shield className="w-8 h-8 text-primary" />
          <span className="font-bold text-lg text-white">AML Platform</span>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <Link to="/" className="flex items-center gap-3 p-3 rounded-lg hover:bg-slate-800"><LayoutDashboard className="w-5 h-5 text-slate-400" /> Overview</Link>
          <Link to="/transactions" className="flex items-center gap-3 p-3 rounded-lg hover:bg-slate-800"><Activity className="w-5 h-5 text-slate-400" /> Transactions</Link>
          <Link to="/typology" className="flex items-center gap-3 p-3 rounded-lg hover:bg-slate-800"><FileSearch className="w-5 h-5 text-slate-400" /> Typology Analysis</Link>
          <Link to="/network" className="flex items-center gap-3 p-3 rounded-lg hover:bg-slate-800"><Share2 className="w-5 h-5 text-slate-400" /> Network Graph</Link>
        </nav>
        <div className="p-4 border-t border-slate-800 flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center"><User className="w-5 h-5" /></div>
            <div>
              <p className="text-sm font-medium text-white">{user?.first_name || 'Admin'}</p>
              <p className="text-xs text-slate-400">{user?.role?.role_name || 'Investigator'}</p>
            </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-8">{children}</main>
    </div>
  );
};

const Overview = () => {
  const data = [{name: 'Mule Ring', value: 400}, {name: 'Layering', value: 300}, {name: 'Smurfing', value: 300}, {name: 'ATO', value: 200}];
  const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'];

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6 text-white">Dashboard Overview</h2>
      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="bg-surface p-6 rounded-xl border border-slate-800">
          <p className="text-slate-400 text-sm">Transactions (24h)</p>
          <h3 className="text-3xl font-bold text-white mt-2">1,204</h3>
        </div>
        <div className="bg-surface p-6 rounded-xl border border-slate-800 border-l-4 border-l-danger">
          <p className="text-slate-400 text-sm">High Risk Alerts</p>
          <h3 className="text-3xl font-bold text-danger mt-2">42</h3>
        </div>
        <div className="bg-surface p-6 rounded-xl border border-slate-800 border-l-4 border-l-warning">
          <p className="text-slate-400 text-sm">Review Queue</p>
          <h3 className="text-3xl font-bold text-warning mt-2">15</h3>
        </div>
      </div>
      <div className="bg-surface p-6 rounded-xl border border-slate-800 h-96">
        <h3 className="text-lg font-semibold text-white mb-4">Detected Typologies</h3>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={80} outerRadius={120} paddingAngle={5} dataKey="value">
              {data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const NetworkIntelligence = () => {
  const [elements, setElements] = useState([]);
  
  useEffect(() => {
    // Fetch mock network for TXN001
    axios.get(`${API_BASE}/network-analysis/TXN001`).then(res => {
      const graph = res.data.graph;
      const cyElements = [];
      graph.nodes.forEach(n => cyElements.push({ data: { id: n.id, label: n.id, type: n.type }}));
      graph.links.forEach((l, i) => cyElements.push({ data: { source: l.source, target: l.target, label: l.relation }}));
      setElements(cyElements);
    }).catch(err => console.log("Ensure backend is running locally for mock data", err));
  }, []);

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-3xl font-bold mb-6 text-white">Network Intelligence</h2>
      <div className="flex-1 bg-surface border border-slate-800 rounded-xl overflow-hidden relative">
        <div className="absolute top-4 left-4 z-10 bg-background/80 p-4 rounded-lg border border-slate-700 backdrop-blur-sm">
           <h4 className="text-white font-semibold mb-2">Detected Patterns</h4>
           <div className="flex items-center gap-2 text-danger text-sm"><AlertTriangle className="w-4 h-4"/> Mule Ring Indicator</div>
           <div className="flex items-center gap-2 text-warning text-sm mt-1"><ZoomIn className="w-4 h-4"/> Circular Transfer</div>
        </div>
        <CytoscapeComponent 
          elements={elements} 
          style={{ width: '100%', height: '100%' }}
          stylesheet={[
            { selector: 'node', style: { 'label': 'data(label)', 'background-color': '#3b82f6', 'color': '#fff' } },
            { selector: 'node[type="customer"]', style: { 'background-color': '#10b981', 'shape': 'diamond' } },
            { selector: 'node[type="transaction"]', style: { 'background-color': '#ef4444', 'shape': 'hexagon' } },
            { selector: 'edge', style: { 'width': 2, 'line-color': '#475569', 'target-arrow-color': '#475569', 'target-arrow-shape': 'triangle', 'label': 'data(label)', 'color': '#94a3b8', 'font-size': '10px' } }
          ]}
        />
      </div>
    </div>
  );
};

const Transactions = () => {
  return (
    <div>
       <h2 className="text-3xl font-bold mb-6 text-white">Live Transactions</h2>
       <div className="bg-surface border border-slate-800 rounded-xl p-6">
         <p className="text-slate-400">Waiting for Data Store connection...</p>
       </div>
    </div>
  );
}

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (window.catalyst) {
      window.catalyst.auth.isUserAuthenticated().then(res => {
        setIsAuthenticated(true);
        window.catalyst.auth.getCurrentUser().then(u => setUser(u));
      }).catch(err => {
        // Fallback for local testing
        setIsAuthenticated(true);
      });
    } else {
        setIsAuthenticated(true);
    }
  }, []);

  if (!isAuthenticated) return <Login />;

  return (
    <Router>
      <DashboardLayout user={user}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/typology" element={<div className="text-white text-2xl font-bold">Typology Analysis</div>} />
          <Route path="/network" element={<NetworkIntelligence />} />
        </Routes>
      </DashboardLayout>
    </Router>
  );
};

export default App;
