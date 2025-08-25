import React from 'react';

export default function Loading() {
  return (
    <div className="flex justify-center items-center h-64">
      <div className="w-12 h-12 border-4 border-cyan-600 border-t-transparent rounded-full animate-spin"></div>
    </div>
  );
}
