import React, { useState } from 'react';

const videoCategories = [
  'Crypto Explainers',
  'AI in Finance',
  'Blockchain Tutorials',
  'Market Analysis',
  'NFT Showcase',
];

const generationTypes = [
  'Animated',
  'Ultra Realistic',
  '3D Rendered',
  'Abstract Art',
  'Futuristic Cyberpunk',
];

const topTrends = [
  'Bitcoin Rally',
  'Ethereum Merge',
  'AI Market Boom',
  'DeFi Innovations',
  'Metaverse Growth',
];

export default function VideoGen() {
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [selectedGenTypes, setSelectedGenTypes] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [selectedTrends, setSelectedTrends] = useState([]);
  const [generatedVideo, setGeneratedVideo] = useState(null);

  const toggleSelection = (item, selectedList, setSelectedList) => {
    if (selectedList.includes(item)) {
      setSelectedList(selectedList.filter((i) => i !== item));
    } else {
      setSelectedList([...selectedList, item]);
    }
  };

  const generateVideo = () => {
    setGeneratedVideo('https://www.w3schools.com/html/mov_bbb.mp4');
  };

  return (
    <div className="max-w-5xl mx-auto p-8 bg-black/30 backdrop-blur-lg rounded-3xl shadow-lg border border-gray-700 space-y-12">
      <h2 className="text-4xl font-semibold text-cyan-500 tracking-wide text-center">Video Generator</h2>

      <section>
        <h3 className="text-xl font-medium text-gray-300 mb-4">Select Video Categories</h3>
        <div className="flex flex-wrap gap-4">
          {videoCategories.map((cat) => (
            <button
              key={cat}
              onClick={() => toggleSelection(cat, selectedCategories, setSelectedCategories)}
              className={`px-5 py-3 rounded-2xl border-2 cursor-pointer select-none transition-colors duration-300
                ${
                  selectedCategories.includes(cat)
                    ? 'bg-cyan-600 border-cyan-600 text-white'
                    : 'border-gray-600 text-gray-400 hover:border-cyan-500 hover:text-cyan-400'
                }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3 className="text-xl font-medium text-gray-300 mb-4">Select Generation Types</h3>
        <div className="flex flex-wrap gap-4">
          {generationTypes.map((type) => (
            <button
              key={type}
              onClick={() => toggleSelection(type, selectedGenTypes, setSelectedGenTypes)}
              className={`px-5 py-3 rounded-2xl border-2 cursor-pointer select-none transition-colors duration-300
                ${
                  selectedGenTypes.includes(type)
                    ? 'bg-purple-700 border-purple-700 text-white'
                    : 'border-gray-600 text-gray-400 hover:border-purple-600 hover:text-purple-400'
                }`}
            >
              {type}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3 className="text-xl font-medium text-gray-300 mb-3">Enter a Prompt (Optional)</h3>
        <textarea
          placeholder="Describe your video idea..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full p-4 rounded-3xl bg-gray-900 bg-opacity-50 text-white border border-gray-700 focus:border-cyan-500 outline-none resize-none shadow-md"
          rows={5}
        />
      </section>

      <section>
        <h3 className="text-xl font-medium text-gray-300 mb-4">Select Top Trends of the Day</h3>
        <div className="flex flex-wrap gap-4">
          {topTrends.map((trend) => (
            <button
              key={trend}
              onClick={() => toggleSelection(trend, selectedTrends, setSelectedTrends)}
              className={`px-5 py-3 rounded-2xl border-2 cursor-pointer select-none transition-colors duration-300
                ${
                  selectedTrends.includes(trend)
                    ? 'bg-emerald-600 border-emerald-600 text-white'
                    : 'border-gray-600 text-gray-400 hover:border-emerald-500 hover:text-emerald-400'
                }`}
            >
              {trend}
            </button>
          ))}
        </div>
      </section>

      <div className="text-center mt-6">
        <button
          onClick={generateVideo}
          className="px-14 py-4 bg-gradient-to-r from-cyan-600 to-purple-700 rounded-3xl font-semibold text-lg text-white hover:scale-105 transition-transform duration-300 shadow-md"
        >
          Generate Video
        </button>
      </div>

      {generatedVideo && (
        <section className="space-y-5 mt-10">
          <video
            src={generatedVideo}
            controls
            className="w-full rounded-3xl shadow-lg border border-cyan-600"
          />
          <div className="flex justify-center">
            <a
              href={generatedVideo}
              download="trendai_generated_video.mp4"
              className="px-10 py-3 border-2 border-cyan-600 rounded-3xl text-cyan-600 font-semibold
                hover:bg-cyan-600 hover:text-white transition-colors duration-300"
            >
              Download Video
            </a>
          </div>
        </section>
      )}
    </div>
  );
}
