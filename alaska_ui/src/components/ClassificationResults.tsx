
import React from 'react';
import { AnalysisResults } from '../App';

interface ClassificationResultsProps {
  results: AnalysisResults;
}

const ClassificationResults: React.FC<ClassificationResultsProps> = ({ results }) => {
  const { classificationReport, confusionMatrix, classLabels, error } = results;

  if (error) {
    return (
      <div className="p-8 text-center text-red-700 bg-red-100 rounded-lg h-full flex flex-col justify-center">
        <h3 className="text-xl font-semibold">Analysis Failed</h3>
        <p>{error}</p>
      </div>
    );
  }

  const hasReport = classificationReport && classificationReport.length > 0;
  const hasMatrix = confusionMatrix && classLabels && confusionMatrix.length > 0 && classLabels.length > 0;

  if (!hasReport && !hasMatrix) {
    return (
        <div className="p-8 text-center text-gray-500 h-full flex flex-col justify-center">
            <h3 className="text-xl font-semibold">Classification Results</h3>
            <p>No classification data was generated from the analysis.</p>
        </div>
    );
  }

  return (
    <div className="h-full flex flex-col space-y-6 overflow-y-auto p-1">
      {hasReport && (
        <div>
          <h3 className="text-xl font-semibold text-neutral-darker mb-3">Classification Report</h3>
          <div className="overflow-x-auto border border-neutral-medium rounded-lg shadow-sm">
            <table className="min-w-full divide-y divide-neutral-medium">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Class</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precision</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recall</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1-Score</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Support</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {classificationReport.map((row) => (
                  <tr key={row.className} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.className}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 tabular-nums">{row.precision.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 tabular-nums">{row.recall.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 tabular-nums">{row['f1-score'].toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 tabular-nums">{row.support.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {hasMatrix && (
         <div>
            <h3 className="text-xl font-semibold text-neutral-darker mb-3">Confusion Matrix</h3>
            <div className="flex items-center justify-center">
                <span className="[writing-mode:vertical-lr] rotate-180 text-center uppercase text-xs font-medium text-gray-500 tracking-wider p-2">Actual</span>
                <div className="flex-grow border border-neutral-medium rounded-lg shadow-sm overflow-hidden">
                    <table className="min-w-full text-center text-sm">
                        <caption className="caption-top p-3 bg-gray-50 border-b border-neutral-medium uppercase text-xs font-medium text-gray-500 tracking-wider">Predicted</caption>
                        <thead>
                            <tr className="border-b border-neutral-medium bg-gray-50">
                                <th className="p-1 w-24"></th>
                                {classLabels?.map((label) => (
                                    <th key={label} scope="col" className="p-2 text-xs font-medium text-gray-500 whitespace-nowrap">{label}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white">
                            {confusionMatrix?.map((row, rowIndex) => (
                                <tr key={rowIndex} className="border-b last:border-b-0 border-neutral-medium">
                                    <th scope="row" className="p-2 font-medium text-gray-500 whitespace-nowrap text-xs bg-gray-50">{classLabels?.[rowIndex]}</th>
                                    {row.map((cell, cellIndex) => (
                                        <td key={cellIndex} className={`p-4 font-mono text-base ${rowIndex === cellIndex ? 'bg-blue-100 font-bold text-brand-primary' : 'text-neutral-darker'}`}>
                                            {cell.toLocaleString()}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

export default ClassificationResults;
