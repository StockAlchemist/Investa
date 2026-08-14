import Link from 'next/link';

export const metadata = {
  title: 'Page Not Found | Investa',
  description: 'The requested page could not be found.',
};

export default function NotFound() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6 text-center bg-background text-foreground">
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 text-primary text-2xl font-bold">
        404
      </div>
      <h1 className="text-2xl font-bold mb-2">Page Not Found</h1>
      <p className="text-muted-foreground max-w-md mb-8">
        The page you are looking for might have been moved or does not exist.
      </p>
      <Link
        href="/"
        aria-label="Return to Dashboard"
        className="px-6 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all shadow-sm"
      >
        Return to Dashboard
      </Link>
    </main>
  );
}
