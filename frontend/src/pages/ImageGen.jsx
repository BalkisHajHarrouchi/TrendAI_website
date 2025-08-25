import React from 'react';
import Loading from '../components/Loading';

export default function ImageGen() {
  return (
    <div className="max-w-4xl mx-auto p-10 bg-black/30 backdrop-blur-lg rounded-3xl shadow-lg border border-gray-700 text-center space-y-6">
      <h2 className="text-5xl font-semibold text-cyan-500 tracking-wide">Image Generation</h2>
      <p className="text-gray-300 text-lg leading-relaxed max-w-3xl mx-auto">
        Create stunning AI-generated images inspired by crypto themes with futuristic aesthetics.
      </p>
      <Loading />
    </div>
  );
}
