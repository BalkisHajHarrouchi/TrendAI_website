// Banner3DSlider.jsx

import React from "react";
import "./Banner3DSlider.css";

export default function Banner3DSlider({
  images = [],          // can now include videos too
  title = "",
  subtitle = "",
  authorTitle = "",
  authorRole = "",
  authorText = "",
  modelImage = null     // set to null to hide the silhouette
}) {
  const showContent =
    Boolean(title || subtitle || authorTitle || authorRole || authorText || modelImage);

  return (
    <div className="banner relative">
      {/* 3D Slider */}
      <div
        className="slider"
        style={{ "--quantity": images.length }}
      >
        {images.map((src, i) => {
          const isVideo = /\.(mp4|webm|ogg)$/i.test(src);
          return (
            <div className="item" key={i} style={{ "--position": i + 1 }}>
              {isVideo ? (
                <video
                  className="media"
                  src={src}
                  playsInline
                  muted
                  loop
                  autoPlay
                  preload="metadata"
                />
              ) : (
                <img className="media" src={src} alt={`slide-${i + 1}`} />
              )}
            </div>
          );
        })}
      </div>

      {/* Content (you said you don't need text; this only renders if any text/model is provided) */}
      {showContent && (
        <div className="content">
          <h1 data-content={title}>{title}</h1>
          <div className="author">
            <h2>{authorTitle}</h2>
            <p><b>{authorRole}</b></p>
            <p>{authorText}</p>
            <p style={{ marginTop: 10, opacity: 0.8 }}>{subtitle}</p>
          </div>
          {!!modelImage && <div className="model" style={{ backgroundImage: `url(${modelImage})` }} />}
        </div>
      )}
    </div>
  );
}
