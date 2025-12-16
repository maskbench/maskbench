import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components';
import { Dashboard, NewExperiment, Results, Compare, Masked, History } from './pages';
import './styles/global.scss';

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Header />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new" element={<NewExperiment />} />
            <Route path="/results" element={<Results />} />
            <Route path="/results/:id" element={<Results />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/masked" element={<Masked />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
