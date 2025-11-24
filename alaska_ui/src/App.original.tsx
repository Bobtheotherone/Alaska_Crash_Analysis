import React, { useState } from "react";

type SeverityBucket = {
  label: string;
  count: number;
};

type MetricStats = {
  mean: number;
  min: number;
  max: number;
};

type AnalysisResponse = {
  summary: {
    total_rows: number;
    numeric_columns: string[];
  };
  metrics: Record<string, MetricStats>;
  charts: {
    severity_distribution: SeverityBucket[];
  };
};

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please select a CSV file to upload.");
      return;
    }

    const form = new FormData();
    form.append("file", file);

    setIsLoading(true);
    try {
      const res = await fetch("/api/upload/", {
        method: "POST",
        body: form
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error((data as any).error || `Request failed with status ${res.status}`);
      }
      const data = (await res.json()) as AnalysisResponse;
      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unexpected error during upload.");
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <h1 className="text-xl font-semibold tracking-tight">
            Alaska Car Crash Analysis
          </h1>
          <p className="text-sm text-slate-500">
            Upload crash data and review summary metrics.
          </p>
        </div>
      </header>

      <main className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6">
        <section className="rounded-lg bg-white p-4 shadow-sm">
          <h2 className="mb-2 text-lg font-semibold">Upload crash dataset</h2>
          <form className="flex flex-col gap-3 md:flex-row md:items-center" onSubmit={handleSubmit}>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="rounded border border-slate-300 bg-white px-3 py-2 text-sm"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="inline-flex items-center justify-center rounded bg-sky-700 px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
            >
              {isLoading ? "Analyzing…" : "Upload and Analyze"}
            </button>
          </form>
          {error && (
            <p className="mt-3 text-sm text-red-600">
              {error}
            </p>
          )}
        </section>

        {result && (
          <section className="flex flex-col gap-6">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="text-sm font-medium text-slate-500">
                  Total rows
                </h3>
                <p className="mt-2 text-2xl font-semibold">
                  {result.summary.total_rows.toLocaleString()}
                </p>
              </div>
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="text-sm font-medium text-slate-500">
                  Numeric columns
                </h3>
                <p className="mt-2 text-base">
                  {result.summary.numeric_columns.join(", ") || "None detected"}
                </p>
              </div>
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="text-sm font-medium text-slate-500">
                  Severity buckets
                </h3>
                <p className="mt-2 text-base">
                  {result.charts.severity_distribution.length}
                </p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-3 text-sm font-semibold text-slate-700">
                  Severity distribution
                </h3>
                {result.charts.severity_distribution.length === 0 ? (
                  <p className="text-sm text-slate-500">
                    No severity column was detected in this dataset.
                  </p>
                ) : (
                  <ul className="space-y-1 text-sm">
                    {result.charts.severity_distribution.map((bucket) => (
                      <li
                        key={bucket.label}
                        className="flex items-center justify-between"
                      >
                        <span>{bucket.label}</span>
                        <span className="font-mono">
                          {bucket.count.toLocaleString()}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-3 text-sm font-semibold text-slate-700">
                  Numeric column statistics
                </h3>
                {Object.keys(result.metrics).length === 0 ? (
                  <p className="text-sm text-slate-500">
                    No numeric columns detected.
                  </p>
                ) : (
                  <table className="min-w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200">
                        <th className="py-1 pr-2 font-medium">Column</th>
                        <th className="py-1 pr-2 font-medium text-right">Mean</th>
                        <th className="py-1 pr-2 font-medium text-right">Min</th>
                        <th className="py-1 pr-2 font-medium text-right">Max</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.metrics).map(([name, stats]) => (
                        <tr key={name} className="border-b border-slate-100 last:border-0">
                          <td className="py-1 pr-2">{name}</td>
                          <td className="py-1 pr-2 text-right">{stats.mean.toFixed(2)}</td>
                          <td className="py-1 pr-2 text-right">{stats.min.toFixed(2)}</td>
                          <td className="py-1 pr-2 text-right">{stats.max.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default App;
