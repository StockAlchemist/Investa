import Dashboard from '../components/Dashboard';
import HoldingsList from '../components/HoldingsList';
import { fetchSummary, fetchHoldings } from '../lib/api';

export const revalidate = 0; // Disable static caching for real-time data

export default async function Home() {
  // Fetch data server-side
  const summaryData = await fetchSummary();
  const holdingsData = await fetchHoldings();

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-10 px-4 py-3">
        <h1 className="text-xl font-bold text-center text-gray-900 dark:text-white">Investa</h1>
      </div>

      {/* Content */}
      <div className="max-w-md mx-auto pt-4">
        <Dashboard summary={summaryData} />
        <HoldingsList holdings={holdingsData} />
      </div>

      {/* Bottom Nav (Visual only for now) */}
      <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3 flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
        <div className="flex flex-col items-center text-blue-600 dark:text-blue-400">
          <span className="text-xl">🏠</span>
          <span className="mt-1">Home</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xl">📊</span>
          <span className="mt-1">Markets</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xl">⚙️</span>
          <span className="mt-1">Settings</span>
        </div>
      </div>
    </main>
  );
}
