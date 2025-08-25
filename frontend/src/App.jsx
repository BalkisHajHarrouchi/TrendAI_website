import React from 'react';
import Header from './components/Header';
import Home from './pages/Home';
import CryptoTrend from './pages/CryptoTrend';
import VideoGen from './pages/VideoGen';
import ImageGen from './pages/ImageGen';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-gray-800 text-gray-200 font-sans selection:bg-cyan-600 selection:text-white overflow-x-hidden">
      <Header />
      <main className="space-y-24 py-12">
        <section id="home">
          <Home />
        </section>
      </main>
    </div>
  );
}

export default App;
