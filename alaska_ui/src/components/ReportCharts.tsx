import React, { useEffect, useRef } from 'react';
import { AnalysisResults } from '../App';

// Chart.js is loaded from a script tag in index.html
declare const Chart: any;

interface ReportChartsProps {
  results: AnalysisResults;
}

const ReportCharts: React.FC<ReportChartsProps> = ({ results }) => {
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<any>(null);

  useEffect(() => {
    if (chartRef.current && results.featureImportance.length > 0) {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;
      
      const sortedFeatures = [...results.featureImportance].sort((a, b) => b.importance - a.importance);
      
      const labels = sortedFeatures.map(item => item.feature);
      const data = sortedFeatures.map(item => item.importance);

      chartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Relative Importance',
            data: data,
            backgroundColor: 'rgba(13, 71, 161, 0.6)',
            borderColor: 'rgba(13, 71, 161, 1)',
            borderWidth: 1
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            },
            title: {
              display: true,
              text: 'Top 5 Feature Importance for Crash Severity',
              font: {
                size: 16
              }
            }
          },
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Importance Score'
              }
            }
          }
        }
      });
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [results.featureImportance]);

  if (results.error) {
    return (
      <div className="p-8 text-center text-red-700 bg-red-100 rounded-lg h-full flex flex-col justify-center">
        <h3 className="text-xl font-semibold">Analysis Failed</h3>
        <p>{results.error}</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col lg:flex-row space-y-4 lg:space-y-0 lg:space-x-6">
      <div className="lg:w-1/2 h-1/2 lg:h-full p-4 border border-neutral-medium rounded-lg bg-white">
         <canvas ref={chartRef}></canvas>
      </div>
      <div className="lg:w-1/2 h-1/2 lg:h-full flex flex-col">
        <h3 className="text-lg font-semibold text-neutral-darker mb-2">Decision Rules</h3>
        <div className="flex-grow space-y-3 overflow-y-auto pr-2">
          {results.decisionRules.map((rule, index) => (
            <div key={index} className="bg-blue-50 border-l-4 border-brand-accent p-4 rounded-r-lg">
              <p className="text-sm text-neutral-dark">{rule}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ReportCharts;
