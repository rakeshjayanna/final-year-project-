import React from 'react';
import Upload from '../components/Upload';

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="card relative overflow-hidden p-8 md:p-10">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-24 -right-24 h-72 w-72 rounded-full bg-mango-200 blur-3xl opacity-60"></div>
          <div className="absolute -bottom-20 -left-20 h-72 w-72 rounded-full bg-amber-100 blur-3xl opacity-70"></div>
        </div>
        <div className="relative text-center">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-gray-900">
            Diagnose Mango Issues with AI
          </h1>
          <p className="mt-3 text-base md:text-lg text-gray-600 max-w-3xl mx-auto">
            Advanced dual-model system for mango analysis. Choose between <strong>Disease Detection</strong> (5 categories) 
            or <strong>Pesticide Detection</strong> (organic vs pesticide). Our CNN and SVM models work together to deliver 
            highly accurate predictions with detailed confidence scores.
          </p>
          <div className="mt-5 flex items-center justify-center gap-3">
            <a href="#upload" className="btn-primary">Start detection</a>
            <a href="/insights" className="btn-ghost">View model insights</a>
          </div>
        </div>
      </section>

      <div id="upload">
        <Upload />
      </div>
    </div>
  );
}
