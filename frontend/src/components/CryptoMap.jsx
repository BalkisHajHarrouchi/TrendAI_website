import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { useEffect, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const countryCoordinates = {
  US: [37.0902, -95.7129],
  JP: [36.2048, 138.2529],
  DE: [51.1657, 10.4515],
  BR: [-14.235, -51.9253],
  KR: [35.9078, 127.7669],
  FR: [46.6034, 1.8883],
  CA: [56.1304, -106.3468],
  IN: [20.5937, 78.9629],
  NG: [9.0820, 8.6753],
  AU: [-25.2744, 133.7751],
  RU: [61.5240, 105.3188],
  MX: [23.6345, -102.5528],
  GB: [55.3781, -3.4360],
  TR: [38.9637, 35.2433],
  IT: [41.8719, 12.5674]
};

const colorByTrend = {
  positive: 'rgba(34, 197, 94, 0.3)',
  neutral: 'rgba(202, 138, 4, 0.3)',
  negative: 'rgba(220, 38, 38, 0.3)'
};

const borderColorByTrend = {
  positive: 'rgba(34, 197, 94, 0.7)',
  neutral: 'rgba(202, 138, 4, 0.7)',
  negative: 'rgba(220, 38, 38, 0.7)'
};

const CryptoMap = () => {
  const [trends, setTrends] = useState([]);

  useEffect(() => {
    fetch('/trendData.json')
      .then(res => res.json())
      .then(data => setTrends(data))
      .catch(err => console.error('Failed to load trend data:', err));
  }, []);

  return (
    <div className="crypto-map-container">
      <h2 className="map-title">🌍 Trending Crypto Globally</h2>

      <div className="map-wrapper">
        <MapContainer
          center={[20, 0]}
          zoom={2}
          style={{ height: '100%', width: '100%' }}
          scrollWheelZoom={true}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
          />

          {trends.map((trend, idx) => {
            const pos = countryCoordinates[trend.isoCode];
            if (!pos) return null;

            return (
              <Marker
                key={idx}
                position={pos}
                icon={new L.DivIcon({
                  html: `
                    <div
                      style="
                        background-color: ${colorByTrend[trend.trend_label]};
                        border: 2.5px solid ${borderColorByTrend[trend.trend_label]};
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        box-shadow: 0 0 8px rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(4px);
                      "
                      class="marker-glow"
                    ></div>`,
                  className: '',
                  iconSize: [24, 24],
                  iconAnchor: [12, 12],
                  popupAnchor: [0, -12],
                })}
              >
                <Popup>
                  <strong>{trend.coin}</strong> from {trend.country}<br />
                  Platform: {trend.platform}<br />
                  Likes: {trend.likes}<br />
                  Trend: {trend.trend_label}<br />
                  Hashtags: {trend.hashtags.map((h, i) => (
                    <span key={i} style={{ marginRight: 6, color: '#0ff' }}>
                      #{h}
                    </span>
                  ))}
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
};

export default CryptoMap;
